"""Model Exporter and Ollama Deployer for automating Modelfile generation and remote model registration."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from engine.client import SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger("os_autofix.engine.deployer")
console = Console()


def generate_modelfile_content(
    base_model_or_gguf: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    stop_tokens: list[str] | None = None,
) -> str:
    """Construct dynamic Ollama Modelfile string."""
    sys_prompt = system_prompt or SYSTEM_PROMPT_TEMPLATE.strip()
    stops = stop_tokens or ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]

    lines: list[str] = [
        f"FROM {base_model_or_gguf}",
        f'SYSTEM """{sys_prompt}"""',
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
    ]

    for stop in stops:
        lines.append(f'PARAMETER stop "{stop}"')

    return "\n".join(lines) + "\n"


class OllamaDeployer:
    """Async client managing Ollama model packaging, registration, and version tagging."""

    def __init__(
        self,
        base_url: str = "http://10.0.0.25:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/").replace("/v1", "")
        self.timeout = timeout_seconds

    async def list_models(self) -> list[str]:
        """Fetch list of registered model tags on the target Ollama instance."""
        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
            except Exception as e:
                logger.error("Failed to query Ollama models at %s: %s", url, e)
                raise

    async def create_model_stream(
        self,
        model_name: str,
        modelfile_content: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Deploy model by streaming Modelfile directly to Ollama `/api/create` endpoint."""
        url = f"{self.base_url}/api/create"
        payload = {
            "name": model_name,
            "modelfile": modelfile_content,
            "stream": True,
        }

        logger.info("Deploying model '%s' to Ollama at %s...", model_name, url)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise RuntimeError(
                        f"Ollama create API returned status {response.status_code}: {error_text.decode('utf-8')}"
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        yield event
                    except json.JSONDecodeError:
                        continue

    async def deploy_model(
        self,
        model_name: str,
        base_model_or_gguf: str,
        output_modelfile_path: Path | str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> bool:
        """Full deployment workflow: generates Modelfile, creates model in Ollama, and asserts registration."""
        modelfile_content = generate_modelfile_content(
            base_model_or_gguf=base_model_or_gguf,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )

        if output_modelfile_path:
            m_path = Path(output_modelfile_path)
            m_path.parent.mkdir(parents=True, exist_ok=True)
            m_path.write_text(modelfile_content, encoding="utf-8")
            logger.info("Saved Modelfile to %s", m_path)

        console.print(
            f"[bold cyan]Deploying '{model_name}' to Ollama ({self.base_url})...[/bold cyan]"
        )

        try:
            # Stream status updates from Ollama
            async for event in self.create_model_stream(model_name, modelfile_content):
                status = event.get("status", "")
                if "error" in event:
                    raise RuntimeError(f"Ollama creation error: {event['error']}")
                if status:
                    logger.debug("Ollama deploy status: %s", status)

            # Verify model is available in /api/tags
            registered = await self.list_models()
            is_present = any(model_name in tag for tag in registered)

            if is_present:
                console.print(
                    f"[bold green]Successfully deployed and verified '{model_name}' on Ollama![/bold green]"
                )
                return True
            else:
                logger.warning(
                    "Model '%s' was deployed but did not appear in /api/tags immediately.",
                    model_name,
                )
                return True

        except Exception as e:
            logger.warning("Direct HTTP /api/create failed: %s. Attempting CLI fallback...", e)
            return await self._deploy_via_cli(model_name, modelfile_content)

    async def _deploy_via_cli(self, model_name: str, modelfile_content: str) -> bool:
        """Fallback deployment using local `ollama create` CLI command."""
        import shutil

        if shutil.which("ollama") is None:
            logger.warning(
                "Local 'ollama' CLI not found in PATH; cannot execute CLI fallback deployment."
            )
            return False

        temp_modelfile = Path(f"/tmp/Modelfile-{model_name.replace(':', '_')}")
        temp_modelfile.write_text(modelfile_content, encoding="utf-8")

        try:
            cmd = ["ollama", "create", model_name, "-f", str(temp_modelfile)]
            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            if proc.returncode == 0:
                console.print(
                    f"[bold green]CLI deployment succeeded for '{model_name}'![/bold green]"
                )
                return True
            else:
                err = stderr_b.decode("utf-8", errors="replace")
                logger.error("CLI deployment failed: %s", err)
                return False
        except FileNotFoundError:
            logger.warning("CLI deployment skipped: 'ollama' executable not found.")
            return False
        finally:
            if temp_modelfile.exists():
                temp_modelfile.unlink()

    async def delete_model(self, model_name: str) -> bool:
        """Delete a registered model tag from Ollama."""
        url = f"{self.base_url}/api/delete"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.request("DELETE", url, json={"name": model_name})
                return resp.status_code == 200
            except Exception as e:
                logger.warning("Failed to delete model '%s': %s", model_name, e)
                return False
