"""Scenario: Broken dpkg/apt lockfiles and package manager corruption."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.package_corruption")


class PackageCorruptionScenario(BaseScenario):
    """Introduces stale/corrupted dpkg and apt lockfiles and asserts package manager health."""

    name = "package_corruption"
    description = "APT package manager is locked or failing. System administrators cannot run updates or install tools."
    category = "Package Management"
    difficulty = "easy"
    max_steps = 6

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Verify baseline package manager functionality."""
        logger.info("Setting up baseline package management state for %s...", self.name)
        res = await sandbox.execute("dpkg --audit && apt-get check")
        return res.exit_code == 0

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Inject stale dpkg and apt lockfiles and simulate interrupted frontend."""
        logger.info("Injecting package manager lock fault...")
        fault_script = """
# Create stale lock files with restrictive flags
touch /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock
chmod 000 /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock 2>/dev/null || true

# Injected dummy interrupted state file
mkdir -p /var/lib/dpkg/updates
echo "Package: autofix-dummy" > /var/lib/dpkg/updates/0001
"""
        res = await sandbox.execute(fault_script)
        return res.exit_code == 0

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify that dpkg audit and apt-get check succeed cleanly."""
        logger.info("Verifying package manager health in %s...", self.name)
        verify_script = """
# Test 1: dpkg audit
dpkg --audit
if [ $? -ne 0 ]; then
    echo "DPKG_AUDIT_FAILED"
    exit 1
fi

# Test 2: apt-get check
apt-get check > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "APT_CHECK_FAILED"
    exit 2
fi

echo "PACKAGE_MANAGER_HEALTHY"
exit 0
"""
        res = await sandbox.execute(verify_script, timeout_seconds=10)
        if res.exit_code == 0 and "PACKAGE_MANAGER_HEALTHY" in res.stdout:
            return True, "Package manager health verified (dpkg audit clean, apt-get check OK)."

        return (
            False,
            f"Package manager health check failed (code {res.exit_code}): {res.combined_output}",
        )
