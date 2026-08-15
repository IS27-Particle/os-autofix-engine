"""Triage specialist agent for non-destructive root cause investigation in OS environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import EngineConfig
from engine.client import PolicyClient
from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.engine.agents.triage")

TRIAGE_SYSTEM_PROMPT = """You are the Lead SRE Triage Specialist Agent.
Your goal is to safely diagnose the root cause of an operating system failure in Ubuntu 24.04 without making ANY state-modifying changes.

RULES:
1. ONLY execute read-only inspection commands (e.g., `journalctl`, `systemctl status`, `cat`, `ls -la`, `ip route`, `iptables -L -n -v`, `ps aux`, `ss -tulpn`, `dmesg`).
2. NEVER run state modifying commands (no rm, no chmod, no systemctl start/restart, no apt, no kill).
3. In each step, return valid JSON with:
   {
     "thought": "Analysis of current findings and next read-only probe",
     "command": "inspection command to run",
     "is_done": false,
     "confidence": 0.0 to 1.0
   }
4. When you have identified the exact root cause, set "is_done": true, and formulate your final conclusion.
"""


@dataclass
class TriageFinding:
    """Structured diagnostic report produced by the Triage Agent."""

    root_cause: str
    affected_daemons: list[str] = field(default_factory=list)
    blast_radius: str = "local"  # "local", "network", "critical"
    evidence: list[str] = field(default_factory=list)
    suggested_strategy: str = ""


class TriageAgent:
    """Read-only diagnostic agent that probes the OS environment to isolate failure mechanisms."""

    def __init__(self, config: EngineConfig, client: PolicyClient | None = None) -> None:
        self.config = config
        self.client = client or PolicyClient(config.llm)

    async def diagnose(
        self,
        sandbox: BaseSandbox,
        symptom_description: str,
        max_probes: int = 4,
    ) -> TriageFinding:
        """Run read-only inspection loop to identify root cause."""
        logger.info(
            "Triage Agent: Starting diagnostic investigation on '%s'...", symptom_description
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"INITIAL SYMPTOMS:\n{symptom_description}\nBegin your read-only inspection.",
            },
        ]

        evidence: list[str] = []
        affected_daemons: set[str] = set()

        for step in range(1, max_probes + 1):
            if self.config.llm.mock_mode or self.config.llm.backend == "mock":
                # Deterministic mock triage finding
                evidence.append("Inspected logs and configuration markers")
                desc_lower = symptom_description.lower()
                if any(
                    k in desc_lower for k in ["dns", "domain", "resolution", "hostname", "resolved"]
                ):
                    affected_daemons.add("systemd-resolved")
                elif "iptables" in desc_lower or "firewall" in desc_lower:
                    affected_daemons.add("iptables")
                elif "docker" in desc_lower or "socket" in desc_lower or "sock" in desc_lower:
                    affected_daemons.add("docker")
                elif "zfs" in desc_lower or "mount" in desc_lower or "dataset" in desc_lower:
                    affected_daemons.add("zfs")
                else:
                    affected_daemons.add("systemd")
                break

            try:
                action, raw = await self.client.get_next_action(messages)
            except Exception as e:
                logger.warning("Triage Agent LLM query failed: %s", e)
                evidence.append(f"LLM communication error: {e}")
                break

            # Enforce read-only constraint
            cmd = action.command.strip()
            if any(
                forbidden in cmd
                for forbidden in [
                    "rm ",
                    "chmod ",
                    "chown ",
                    "sed -i",
                    "systemctl restart",
                    "systemctl start",
                    "kill ",
                ]
            ):
                cmd = f"echo 'BLOCKED_MUTATION: {cmd}'"

            exec_res = await sandbox.execute(cmd, timeout_seconds=10)
            obs = exec_res.combined_output
            evidence.append(f"Probe {step} (`{cmd}`): {obs[:120]}")

            if "systemd-resolved" in cmd or "systemd-resolved" in obs:
                affected_daemons.add("systemd-resolved")
            if "docker" in cmd or "docker" in obs:
                affected_daemons.add("docker")
            if "iptables" in cmd or "iptables" in obs:
                affected_daemons.add("iptables")

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"[OUTPUT]:\n{obs}"})

            if action.is_done:
                break

        return TriageFinding(
            root_cause=f"Diagnostic isolation of {symptom_description}",
            affected_daemons=list(affected_daemons) or ["system"],
            blast_radius="network"
            if any("network" in d or "iptables" in d or "dns" in d for d in affected_daemons)
            else "local",
            evidence=evidence,
            suggested_strategy="Perform targeted remediation on affected daemons.",
        )
