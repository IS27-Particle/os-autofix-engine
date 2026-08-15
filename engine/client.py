"""Async client supporting Ollama and Open-WebUI endpoints with automated retries and schema repair."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from config.settings import LLMConfig
from engine.action_schema import AgentAction, parse_action_response

logger = logging.getLogger("os_autofix.engine.client")

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous Linux Site Reliability Engineer and OS diagnostics agent.
Your objective is to diagnose and repair operating system-level faults in an isolated Linux environment (Ubuntu 24.04 / systemd).

RULES AND CONSTRAINTS:
1. You have root access inside the guest.
2. All commands MUST be strictly non-interactive (e.g. use `apt-get -y`, `systemctl`, `journalctl -n 50`, `cat`, `sed`, `ip`, `dig`).
3. Never launch blocking or interactive processes like `nano`, `top`, `vim`, or unflagged interactive installers.
4. Each command has a default timeout of 15 seconds.
5. Large outputs will be truncated at 2000 characters. Keep queries targeted.
6. When you determine the fault is completely fixed, output `"is_done": true` and `"command": ""` (or a final sanity check command).

You MUST ALWAYS respond with a SINGLE valid JSON object matching this schema:
```json
{
  "thought": "Detailed reasoning about current observations, hypothesis, and next action",
  "command": "The exact non-interactive bash command to execute",
  "timeout_seconds": 15,
  "is_done": false,
  "confidence": 0.95
}
```
"""


class LLMClientError(Exception):
    """Base exception for LLM client communication or parsing failures."""

    pass


class PolicyClient:
    """Async policy client supporting direct Ollama API, Open-WebUI, and OpenAI endpoints."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the underlying AsyncOpenAI and HTTP client instances."""
        base_url = self.config.active_endpoint
        api_key = self.config.active_api_key

        # Ensure trailing /v1 or appropriate path if not present for OpenAI client
        if not base_url.endswith("/v1") and self.config.backend in ("ollama", "openai"):
            if not base_url.endswith("/api"):
                openai_base = f"{base_url}/v1"
            else:
                openai_base = base_url
        else:
            openai_base = base_url

        headers: dict[str, str] = {}
        if self.config.backend == "open-webui" and self.config.open_webui_api_key:
            headers["Authorization"] = f"Bearer {self.config.open_webui_api_key}"

        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            headers=headers,
        )

        self.openai_client = AsyncOpenAI(
            base_url=openai_base,
            api_key=api_key or "EMPTY",
            http_client=self.http_client,
        )

    def create_initial_messages(
        self,
        scenario_description: str,
        initial_observation: str = "",
    ) -> list[dict[str, str]]:
        """Construct the initial conversation messages for a scenario episode."""
        user_content = f"FAULT SCENARIO:\n{scenario_description}"
        if initial_observation:
            user_content += f"\n\nINITIAL OBSERVATION:\n{initial_observation}"
        user_content += "\n\nPlease analyze the system and begin troubleshooting."

        return [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": user_content},
        ]

    async def get_next_action(
        self,
        messages: list[dict[str, str]],
        max_retries: int | None = None,
    ) -> tuple[AgentAction, str]:
        """Request next action with automatic schema repair, backoff, and JSON validation."""
        retries = max_retries if max_retries is not None else self.config.max_retries

        if self.config.mock_mode or self.config.backend == "mock":
            return self._mock_action(messages), "mock-completion"

        working_messages = list(messages)
        backoff = 1.0

        for attempt in range(1, retries + 1):
            try:
                logger.debug(
                    "Requesting action from backend '%s' (model: %s, attempt: %d/%d)...",
                    self.config.backend,
                    self.config.model_name,
                    attempt,
                    retries,
                )

                raw_content = await self._send_completion_request(working_messages)
                logger.debug("Raw model completion received: %s", raw_content[:150])

                try:
                    action = parse_action_response(raw_content)
                    return action, raw_content
                except (ValueError, ValidationError) as parse_err:
                    logger.warning(
                        "Output validation failed on attempt %d/%d: %s",
                        attempt,
                        retries,
                        parse_err,
                    )
                    if attempt == retries:
                        raise LLMClientError(
                            f"Model failed to produce valid AgentAction after {retries} attempts: {parse_err}\n"
                            f"Last raw output: {raw_content}"
                        ) from parse_err

                    # Feed schema error back into messages to guide self-correction
                    working_messages.append({"role": "assistant", "content": raw_content})
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"ERROR: Your previous response could not be parsed into the required JSON schema.\n"
                                f"Validation error: {parse_err}\n"
                                f"Please provide ONLY a valid JSON object matching the AgentAction schema strictly."
                            ),
                        }
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 1.5

            except Exception as e:
                if isinstance(e, LLMClientError):
                    raise
                logger.warning(
                    "Endpoint connection error on attempt %d/%d: %s", attempt, retries, e
                )
                if attempt == retries:
                    raise LLMClientError(
                        f"Failed communicating with endpoint '{self.config.active_endpoint}': {e}"
                    ) from e
                await asyncio.sleep(backoff)
                backoff *= 1.5

        raise LLMClientError("Exceeded maximum retries without producing a valid action.")

    async def _send_completion_request(self, messages: list[dict[str, str]]) -> str:
        """Dispatch chat completion to OpenAI-compatible endpoint or fallback."""
        start_t = time.monotonic()
        try:
            response = await self.openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=self.config.model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                response_format={"type": "json_object"},  # type: ignore[arg-type]
            )
            duration = time.monotonic() - start_t
            try:
                from monitoring.metrics import LLM_LATENCY_SECONDS

                LLM_LATENCY_SECONDS.observe(
                    duration,
                    model_tag=self.config.model_name,
                    backend=self.config.backend,
                )
            except Exception:
                pass
            return response.choices[0].message.content or ""
        except Exception as e:
            # If standard OpenAI /v1 format fails and backend is Ollama, try native /api/chat
            if self.config.backend == "ollama":
                logger.debug("Falling back to native Ollama /api/chat endpoint...")
                res = await self._call_native_ollama(messages)
                duration = time.monotonic() - start_t
                try:
                    from monitoring.metrics import LLM_LATENCY_SECONDS

                    LLM_LATENCY_SECONDS.observe(
                        duration,
                        model_tag=self.config.model_name,
                        backend=self.config.backend,
                    )
                except Exception:
                    pass
                return res
            raise e

    async def _call_native_ollama(self, messages: list[dict[str, str]]) -> str:
        """Direct call to native Ollama /api/chat endpoint with JSON format enforce."""
        base = self.config.ollama_base_url.replace("/v1", "")
        url = f"{base}/api/chat"

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }

        resp = await self.http_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _mock_action(self, messages: list[dict[str, str]]) -> AgentAction:
        """Deterministic mock policy for offline development and testing."""
        full_text = " ".join(m["content"] for m in messages).lower()
        last_msg = messages[-1]["content"].lower() if messages else ""

        if "systemd_dns" in full_text or "dns" in full_text or "resolv" in full_text:
            if (
                "systemctl restart systemd-resolved" in full_text
                or "nameserver 1.1.1.1" in full_text
                or "exit code: 0" in last_msg
            ):
                return AgentAction(
                    thought="DNS service restarted, ensuring stub resolv conf and service active.",
                    command="ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf 2>/dev/null; systemctl restart systemd-resolved",
                    is_done=True,
                    confidence=0.98,
                )
            return AgentAction(
                thought="Detected DNS failure. Restarting systemd-resolved.",
                command="systemctl restart systemd-resolved",
                is_done=False,
                confidence=0.9,
            )
        elif "routing" in full_text or "route" in full_text or "gateway" in full_text:
            return AgentAction(
                thought="Detected routing table corruption. Re-adding default route.",
                command="ip route replace default via 10.0.0.1 dev eth0 || dhclient",
                is_done=True,
                confidence=0.95,
            )
        elif "dpkg" in full_text or "lock" in full_text or "package" in full_text:
            return AgentAction(
                thought="Detected apt/dpkg lock. Removing stale lockfiles.",
                command="rm -f /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock && dpkg --configure -a",
                is_done=True,
                confidence=0.95,
            )
        elif "permission" in full_text or "sudoers" in full_text or "sshd" in full_text:
            return AgentAction(
                thought="Detected permission misconfiguration. Restoring sudoers to 0440.",
                command="chmod 0440 /etc/sudoers 2>/dev/null; chmod 0600 /etc/ssh/sshd_config 2>/dev/null",
                is_done=True,
                confidence=0.95,
            )
        else:
            return AgentAction(
                thought="Performing initial diagnostics.",
                command="uname -a",
                is_done=True,
                confidence=0.5,
            )

    async def close(self) -> None:
        """Close underlying HTTP connections."""
        await self.http_client.aclose()
