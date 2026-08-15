"""Open-WebUI Pipeline filter and interactive troubleshooting stream provider."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from config.settings import get_default_config
from engine.orchestrator import Orchestrator
from sandbox.incus_sandbox import IncusSandbox
from scenarios.registry import get_scenario
from trainer.trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger("os_autofix.integrations.open_webui")


class Pipeline:
    """Open-WebUI Pipe pipeline enabling interactive OS fault troubleshooting from the web chat interface."""

    class Valves:
        """Configurable valves exposed in Open-WebUI Admin UI."""

        OLLAMA_ENDPOINT: str = "http://10.0.0.25:11434/v1"
        DEFAULT_MODEL: str = "qwen2.5-coder:7b"
        SANDBOX_TYPE: str = "container"
        MAX_STEPS: int = 6

    def __init__(self) -> None:
        self.type = "pipe"
        self.id = "os_autofix_pipeline"
        self.name = "OS-AutoFix Engine Pipeline"
        self.valves = self.Valves()

    async def on_startup(self) -> None:
        """Called by Open-WebUI on pipeline startup."""
        logger.info("OS-AutoFix Pipeline started for Open-WebUI.")

    async def on_shutdown(self) -> None:
        """Called by Open-WebUI on pipeline shutdown."""
        logger.info("OS-AutoFix Pipeline shut down.")

    def _match_scenario(self, prompt: str) -> str:
        """Find best scenario match from user prompt keywords."""
        prompt_lower = prompt.lower()
        if "docker" in prompt_lower or "sock" in prompt_lower:
            return "docker_socket"
        if "zfs" in prompt_lower or "mount" in prompt_lower:
            return "zfs_mount"
        if "firewall" in prompt_lower or "iptables" in prompt_lower:
            return "iptables_lockout"
        if "dns" in prompt_lower or "resolved" in prompt_lower or "domain" in prompt_lower:
            return "systemd_dns"
        if "route" in prompt_lower or "gateway" in prompt_lower or "ip route" in prompt_lower:
            return "network_routing"
        if "apt" in prompt_lower or "dpkg" in prompt_lower or "lock" in prompt_lower:
            return "package_corruption"
        if "sudo" in prompt_lower or "perm" in prompt_lower:
            return "file_permissions"
        return "systemd_dns"

    async def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream real-time diagnostic agent execution into Open-WebUI chat."""
        scenario_name = self._match_scenario(user_message)
        scenario = get_scenario(scenario_name)

        yield "### 🛠️ OS-AutoFix Autonomous Diagnostic Engine\n\n"
        yield f"**Matched Scenario**: `{scenario.name}` ({scenario.category})\n"
        yield f"**Target Instance**: `Incus ({self.valves.SANDBOX_TYPE})`\n\n"
        yield "---\n"

        cfg = get_default_config()
        cfg.incus.instance_type = "container" if self.valves.SANDBOX_TYPE == "container" else "vm"
        cfg.llm.model_name = self.valves.DEFAULT_MODEL
        cfg.llm.ollama_base_url = self.valves.OLLAMA_ENDPOINT

        buffer = TrajectoryBuffer()
        sandbox = IncusSandbox(
            instance_name=f"webui-sb-{scenario_name.replace('_', '-')}",
            is_vm=(self.valves.SANDBOX_TYPE == "vm"),
        )
        orchestrator = Orchestrator(
            config=cfg,
            trajectory_buffer=buffer,
            custom_sandbox_factory=(lambda _: sandbox),
        )

        yield "🚀 **Initializing ephemeral sandbox & injecting fault...**\n\n"
        yield f"🔍 **Fault State Target**: {scenario.description}\n\n"
        yield "#### 🤖 Agent Action Stream:\n"

        try:
            traj = await orchestrator.run_single_episode(scenario=scenario)

            for step in traj.steps:
                yield f"**Step {step.step_index}**\n"
                yield f"> 💭 *Thought*: {step.thought}\n\n"
                yield f"```bash\n$ {step.command}\n```\n"
                yield f"<details><summary>Output ({len(step.stdout)} chars)</summary>\n\n```\n{step.stdout[:500]}\n```\n</details>\n\n"

            if traj.success:
                yield "\n### ✅ Status: RESOLVED\n"
                yield f"System restored in **{traj.duration_seconds:.2f}s** over **{len(traj.steps)} step(s)**.\n"
            else:
                yield "\n### ❌ Status: FAILED / UNRESOLVED\n"
                yield f"Diagnostics: {traj.verification_message}\n"
        finally:
            await sandbox.cleanup()
            yield "\n*Ephemeral sandbox destroyed. Host resources reclaimed.*\n"
