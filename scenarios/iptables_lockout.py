"""Diagnostic scenario for outbound firewall lockouts blocking DNS and HTTP traffic."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.iptables_lockout")


class IPTablesLockoutScenario(BaseScenario):
    """Diagnose and remove outbound firewall DROP rules blocking DNS and Web traffic."""

    name: str = "iptables_lockout"
    description: str = (
        "Outbound network traffic is completely blocked by restrictive firewall / iptables rules. "
        "DNS queries and HTTP/HTTPS requests to external hosts timeout."
    )
    category: str = "Networking / Security"
    difficulty: str = "medium"
    max_steps: int = 8

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Ensure clean iptables baseline state."""
        logger.info("Setting up baseline iptables state for %s...", self.name)
        cmds = [
            "mkdir -p /etc/iptables",
            "iptables -F OUTPUT 2>/dev/null || true",
            "iptables -P OUTPUT ACCEPT 2>/dev/null || true",
        ]
        for cmd in cmds:
            await sandbox.execute(cmd)
        return True

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Inject restrictive DROP rules on the OUTPUT chain."""
        logger.info("Injecting outbound firewall drop rules...")
        cmds = [
            "iptables -I OUTPUT 1 -p udp --dport 53 -j DROP 2>/dev/null || true",
            "iptables -I OUTPUT 1 -p tcp --dport 53 -j DROP 2>/dev/null || true",
            "iptables -I OUTPUT 1 -p tcp --dport 80 -j DROP 2>/dev/null || true",
            "iptables -I OUTPUT 1 -p tcp --dport 443 -j DROP 2>/dev/null || true",
            "echo 'FIREWALL_LOCKED=1' > /etc/iptables/lock_status.conf",
        ]
        for cmd in cmds:
            await sandbox.execute(cmd)
        return True

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify that OUTPUT chain has no active DROP rules and policy is ACCEPT."""
        res_rules = await sandbox.execute(
            "iptables -S OUTPUT 2>/dev/null || iptables -L OUTPUT -n 2>/dev/null"
        )
        if "-j DROP" in res_rules.stdout or "DROP" in res_rules.stdout:
            return (
                False,
                f"Firewall verification failed: Active DROP rules detected in OUTPUT chain:\n{res_rules.stdout.strip()}",
            )

        res_flag = await sandbox.execute("cat /etc/iptables/lock_status.conf 2>/dev/null")
        if "FIREWALL_LOCKED=1" in res_flag.stdout and "-j DROP" in res_rules.stdout:
            return False, "Firewall lock state flag remains active."

        return True, "Firewall rules cleared. Outbound network traffic permitted."
