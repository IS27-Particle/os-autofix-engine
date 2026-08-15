"""Scenario: Critical daemon configuration permission lockouts."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.file_permissions")


class FilePermissionsScenario(BaseScenario):
    """Corrupts critical security and daemon configuration permissions and asserts service restoration."""

    name = "file_permissions"
    description = "Critical system daemon configurations (/etc/sudoers, /etc/ssh/sshd_config) have corrupted permissions causing authentication and service failures."
    category = "Security / Permissions"
    difficulty = "easy"
    max_steps = 6

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Ensure sudo and ssh daemon configs exist."""
        logger.info("Setting up baseline permission state for %s...", self.name)
        res = await sandbox.execute(
            "touch /etc/sudoers && (test -f /etc/ssh/sshd_config || touch /etc/ssh/sshd_config)"
        )
        return res.exit_code == 0

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Break sudoers permissions (world writable) and lockout sshd_config."""
        logger.info("Injecting permission faults into critical files...")
        fault_script = """
# Sudo refuses to execute when sudoers is world-writable (0777)
chmod 0777 /etc/sudoers

# SSH daemon rejects unreadable config
if [ -f /etc/ssh/sshd_config ]; then
    chmod 0000 /etc/ssh/sshd_config
fi
"""
        res = await sandbox.execute(fault_script)
        return res.exit_code == 0

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify that sudoers and sshd configurations have valid, safe permissions."""
        logger.info("Verifying file permissions in %s...", self.name)
        verify_script = """
# Check sudoers permissions (must not be world-writable; standard is 0440)
SUDOERS_PERM=$(stat -c "%a" /etc/sudoers)
if [ "$SUDOERS_PERM" = "777" ] || [ "$SUDOERS_PERM" = "666" ] || [ "$SUDOERS_PERM" = "775" ]; then
    echo "SUDOERS_INSECURE_PERMS: $SUDOERS_PERM"
    exit 1
fi

# Verify sudo syntax check
sudo -V > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "SUDO_VALIDATION_FAILED"
    exit 2
fi

# Check sshd config readability if present
if [ -f /etc/ssh/sshd_config ]; then
    if ! [ -r /etc/ssh/sshd_config ]; then
        echo "SSHD_CONFIG_UNREADABLE"
        exit 3
    fi
fi

echo "PERMISSIONS_VALID"
exit 0
"""
        res = await sandbox.execute(verify_script, timeout_seconds=8)
        if res.exit_code == 0 and "PERMISSIONS_VALID" in res.stdout:
            return True, "Critical daemon and security file permissions verified successfully."

        return (
            False,
            f"Permission verification failed (code {res.exit_code}): {res.combined_output}",
        )
