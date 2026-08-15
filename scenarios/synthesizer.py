"""LLM-driven synthetic OS diagnostic scenario generator with dynamic sandbox pre-flight validation."""

from __future__ import annotations

import ast
import importlib.util
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from config.settings import EngineConfig
from engine.client import PolicyClient
from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario
from scenarios.registry import register_scenario

logger = logging.getLogger("os_autofix.scenarios.synthesizer")

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert Linux Kernel and Site Reliability Engineering systems architect.
Your task is to create a complete, executable Python class defining an autonomous OS fault troubleshooting scenario for Ubuntu 24.04 (systemd).

REQUIREMENTS:
1. Inherit from `BaseScenario` (from `scenarios.base_scenario import BaseScenario`).
2. Class MUST define:
   - `name`: unique snake_case string (e.g. `pam_auth_lockout`, `limits_nofile_exhaustion`, `systemd_timer_deadlock`).
   - `description`: clear symptom explanation.
   - `category`: e.g. "Security / Authentication", "Resource Limits", "System Daemons".
   - `difficulty`: "easy", "medium", or "hard".
   - `max_steps`: integer between 4 and 10.
   - `async def setup(self, sandbox: BaseSandbox) -> bool`: Prepares baseline configuration and verification markers.
   - `async def inject_fault(self, sandbox: BaseSandbox) -> bool`: Injects the failure.
   - `async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]`: Checks if the system is healthy.
   - `async def reference_fix(self, sandbox: BaseSandbox) -> bool`: Executes the known correct remediation commands.
3. All commands executed via `await sandbox.execute(...)` must be strictly non-interactive (apt -y, systemctl, chmod, sed, echo, etc.).
4. Return ONLY valid Python code inside a markdown code fence (```python ... ```).
"""


class ScenarioSynthesizer:
    """Synthesizes novel OS diagnostic scenarios using teacher LLMs and validates them inside Incus sandboxes."""

    def __init__(
        self,
        config: EngineConfig,
        client: PolicyClient | None = None,
        output_dir: Path | str = "scenarios/synthetic",
    ) -> None:
        self.config = config
        self.client = client or PolicyClient(config.llm)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_python_code(self, response: str) -> str:
        """Extract python code block from LLM markdown response."""
        match = re.search(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def compile_scenario_class(self, code_str: str) -> type[BaseScenario]:
        """Dynamically compile code string and extract the BaseScenario subclass."""
        # Syntax check
        ast.parse(code_str)

        module_name = f"dynamic_scenario_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if not spec:
            raise RuntimeError("Failed creating module spec for dynamic scenario")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        exec(code_str, module.__dict__)

        scenario_cls: type[BaseScenario] | None = None
        for obj in module.__dict__.values():
            if isinstance(obj, type) and issubclass(obj, BaseScenario) and obj is not BaseScenario:
                scenario_cls = obj
                break

        if not scenario_cls:
            raise ValueError("No valid BaseScenario subclass found in synthesized code")

        return scenario_cls

    async def generate_scenario_code(self, topic: str | None = None) -> str:
        """Prompt teacher LLM to generate novel Linux failure definition."""
        user_prompt = "Generate a new Linux troubleshooting scenario."
        if topic:
            user_prompt += f" Topic: {topic}."
        else:
            user_prompt += " Examples: PAM security lockout, /etc/security/limits.conf file descriptor starvation, or corrupted cron timer unit."

        messages = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if self.config.llm.mock_mode or self.config.llm.backend == "mock":
            return self._generate_fallback_synthetic_code(topic or "limits_nofile")

        try:
            raw_content = await self.client._send_completion_request(messages)
            return self.extract_python_code(raw_content)
        except Exception as e:
            logger.warning("Teacher LLM generation failed: %s. Using fallback template.", e)
            return self._generate_fallback_synthetic_code(topic or "pam_auth_lockout")

    def _generate_fallback_synthetic_code(self, name_seed: str) -> str:
        """Deterministic synthetic scenario template for offline testing."""
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name_seed.lower()) or "synthetic_limits"
        class_name = "".join(part.capitalize() for part in clean_name.split("_")) + "Scenario"

        return f'''"""Auto-generated synthetic scenario: {clean_name}"""

from __future__ import annotations
import logging
from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.{clean_name}")

class {class_name}(BaseScenario):
    name: str = "{clean_name}"
    description: str = "Resource limit configuration error preventing process file descriptor allocation."
    category: str = "Resource Limits"
    difficulty: str = "medium"
    max_steps: int = 6

    async def setup(self, sandbox: BaseSandbox) -> bool:
        await sandbox.execute("mkdir -p /etc/security")
        await sandbox.execute("echo '* soft nofile 65535' > /etc/security/limits.conf")
        return True

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        await sandbox.execute("echo '* soft nofile 0' > /etc/security/limits.conf")
        await sandbox.execute("echo 'FAULT_ACTIVE=1' > /tmp/nofile_fault.flag")
        return True

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        res = await sandbox.execute("cat /etc/security/limits.conf 2>/dev/null")
        if "nofile 0" in res.stdout:
            return False, "File descriptor limit is set to 0 (starvation)."
        return True, "Resource limits verified functional."

    async def reference_fix(self, sandbox: BaseSandbox) -> bool:
        await sandbox.execute("echo '* soft nofile 65535' > /etc/security/limits.conf")
        await sandbox.execute("rm -f /tmp/nofile_fault.flag")
        return True
'''

    async def validate_preflight(
        self,
        scenario: BaseScenario,
        sandbox: BaseSandbox,
    ) -> tuple[bool, str]:
        """Execute 3-phase sandbox validation: Clean -> Fault Injected -> Remediated."""
        logger.info("Running pre-flight validation on synthesized scenario '%s'...", scenario.name)

        # 1. Baseline setup check
        await sandbox.setup()
        await scenario.setup(sandbox)
        baseline_ok, baseline_msg = await scenario.verify(sandbox)
        if not baseline_ok:
            return (
                False,
                f"Phase 1 Failed: Scenario did not pass baseline verification after setup ({baseline_msg})",
            )

        # 2. Fault injection check
        await scenario.inject_fault(sandbox)
        fault_verified, fault_msg = await scenario.verify(sandbox)
        if fault_verified:
            return (
                False,
                f"Phase 2 Failed: Scenario still passed verification immediately after fault injection ({fault_msg})",
            )

        # 3. Remediation check via reference_fix
        if hasattr(scenario, "reference_fix") and callable(scenario.reference_fix):
            await scenario.reference_fix(sandbox)
            fix_ok, fix_msg = await scenario.verify(sandbox)
            if not fix_ok:
                return (
                    False,
                    f"Phase 3 Failed: Reference fix failed to resolve scenario ({fix_msg})",
                )

        return True, "All 3 pre-flight verification phases passed."

    async def synthesize(
        self,
        count: int = 1,
        topic: str | None = None,
        sandbox_factory: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Synthesize, validate, save, and register N new scenarios."""
        results: list[dict[str, Any]] = []

        for i in range(1, count + 1):
            logger.info("Synthesizing scenario %d/%d (topic: %s)...", i, count, topic or "general")
            code = await self.generate_scenario_code(topic=topic)

            try:
                scenario_cls = self.compile_scenario_class(code)
                scenario_instance = scenario_cls()

                # Pre-flight sandbox validation if factory provided
                if sandbox_factory:
                    sb = sandbox_factory(f"autofix-synth-{scenario_instance.name}")
                    try:
                        ok, msg = await self.validate_preflight(scenario_instance, sb)
                        if not ok:
                            logger.warning(
                                "Synthesized scenario '%s' failed pre-flight: %s",
                                scenario_instance.name,
                                msg,
                            )
                            results.append(
                                {"name": scenario_instance.name, "valid": False, "error": msg}
                            )
                            continue
                    finally:
                        await sb.cleanup()

                # Save validated scenario file
                file_path = self.output_dir / f"{scenario_instance.name}.py"
                file_path.write_text(code, encoding="utf-8")

                # Register scenario
                register_scenario(scenario_cls)
                logger.info(
                    "Successfully synthesized and registered scenario '%s' -> %s",
                    scenario_instance.name,
                    file_path,
                )

                results.append(
                    {
                        "name": scenario_instance.name,
                        "category": scenario_instance.category,
                        "difficulty": scenario_instance.difficulty,
                        "file_path": str(file_path),
                        "valid": True,
                    }
                )
            except Exception as e:
                logger.error("Failed compiling/validating synthesized scenario: %s", e)
                results.append({"valid": False, "error": str(e)})

        return results
