"""Remediation specialist agent for surgical mutation and configuration fixing in OS environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import EngineConfig
from engine.agents.triage_agent import TriageFinding
from engine.client import PolicyClient
from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.engine.agents.remediation")

REMEDIATION_SYSTEM_PROMPT = """You are the Lead SRE Remediation Specialist Agent.
Your goal is to surgically execute the necessary configuration fixes, file edits, and service restarts to resolve the system fault identified by the Triage Agent.

RULES:
1. ONLY execute minimal, necessary mutations to resolve the specified root cause.
2. Avoid unnecessary system wide restarts or package removals.
3. In each step, return valid JSON:
   {
     "thought": "Rationale for the mutation",
     "command": "remediation shell command",
     "is_done": false,
     "confidence": 0.0 to 1.0
   }
4. When remediation commands have all been executed, set "is_done": true.
"""


@dataclass
class RemediationResult:
    """Report produced by the Remediation Agent."""

    success_attempted: bool
    executed_commands: list[str] = field(default_factory=list)
    mutations_summary: str = ""
    errors_encountered: list[str] = field(default_factory=list)


class RemediationAgent:
    """Surgical execution agent that consumes triage findings and executes precise state mutations."""

    def __init__(self, config: EngineConfig, client: PolicyClient | None = None) -> None:
        self.config = config
        self.client = client or PolicyClient(config.llm)

    async def remediate(
        self,
        sandbox: BaseSandbox,
        triage_finding: TriageFinding,
        max_steps: int = 5,
    ) -> RemediationResult:
        """Execute remediation loop based on triage findings."""
        logger.info(
            "Remediation Agent: Starting remediation for root cause '%s'...",
            triage_finding.root_cause,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": REMEDIATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TRIAGE REPORT:\n"
                    f"Root Cause: {triage_finding.root_cause}\n"
                    f"Affected Daemons: {', '.join(triage_finding.affected_daemons)}\n"
                    f"Evidence:\n" + "\n".join(triage_finding.evidence) + "\n\n"
                    "Execute surgical remediation commands now."
                ),
            },
        ]

        executed_commands: list[str] = []
        errors: list[str] = []

        for _step in range(1, max_steps + 1):
            if self.config.llm.mock_mode or self.config.llm.backend == "mock":
                # Deterministic mock remediation commands
                if "systemd-resolved" in triage_finding.affected_daemons:
                    cmd = "systemctl restart systemd-resolved"
                elif "iptables" in triage_finding.affected_daemons:
                    cmd = "iptables -F"
                elif "docker" in triage_finding.affected_daemons:
                    cmd = "chmod 0660 /var/run/docker.sock"
                else:
                    cmd = "systemctl restart systemd"

                await sandbox.execute(cmd)
                executed_commands.append(cmd)
                break

            try:
                action, raw = await self.client.get_next_action(messages)
            except Exception as e:
                logger.warning("Remediation Agent LLM query failed: %s", e)
                errors.append(f"LLM communication error: {e}")
                break

            cmd = action.command.strip()
            if cmd:
                exec_res = await sandbox.execute(cmd, timeout_seconds=15)
                executed_commands.append(cmd)
                if exec_res.exit_code != 0:
                    errors.append(
                        f"Command `{cmd}` exited with {exec_res.exit_code}: {exec_res.stderr}"
                    )

                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": f"[EXIT CODE: {exec_res.exit_code}]\nSTDOUT:\n{exec_res.combined_output}",
                    }
                )

            if action.is_done:
                break

        return RemediationResult(
            success_attempted=len(executed_commands) > 0,
            executed_commands=executed_commands,
            mutations_summary=f"Executed {len(executed_commands)} remediation commands.",
            errors_encountered=errors,
        )
