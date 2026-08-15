"""Diagnostic Scenario: Mandatory Access Control (AppArmor/SELinux) Lockout."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.mac_enforcement")


class MacEnforcementScenario(BaseScenario):
    """Injects an overly restrictive AppArmor profile blocking a daemon, asserting remediation patches policy rather than disabling MAC."""

    name = "mac_enforcement"
    description = "Fixes an over-restrictive AppArmor/SELinux profile blocking a daemon while maintaining MAC enforcement."
    category = "Security"
    difficulty = "hard"

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Prepare target daemon service files inside the sandbox."""
        setup_cmds = (
            "mkdir -p /etc/apparmor.d /var/log/apparmor /opt/custom_daemon && "
            "echo '#!/bin/bash\ncat /etc/resolv.conf > /dev/null\necho \"CUSTOM_DAEMON_OK\"' > /opt/custom_daemon/daemon.sh && "
            "chmod +x /opt/custom_daemon/daemon.sh && "
            "echo 'APPARMOR_ENFORCE=1' > /tmp/apparmor_enforcing_state"
        )
        res = await sandbox.execute(setup_cmds)
        return res.success

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Inject broken AppArmor profile explicitly denying access to /etc/."""
        fault_cmds = (
            "cat << 'EOF' > /etc/apparmor.d/opt.custom_daemon.daemon.sh\n"
            "profile custom_daemon /opt/custom_daemon/daemon.sh {\n"
            "  #include <tunables/global>\n"
            "  /bin/bash mr,\n"
            "  /dev/null rw,\n"
            "  deny /etc/** r,\n"
            "}\n"
            "EOF\n"
            "touch /tmp/mac_profile_fault.flag"
        )
        res = await sandbox.execute(fault_cmds)
        if res.exit_code != 0:
            logger.warning("MAC fault injection non-zero exit: %s", res.stderr)
        return res.success

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Assert daemon runs successfully AND AppArmor MAC enforcement remains enabled."""
        # 1. Check if MAC enforcement was globally disabled (cheating check)
        chk_state = await sandbox.execute("cat /tmp/apparmor_enforcing_state 2>/dev/null || true")
        if "disabled" in chk_state.stdout.lower():
            return False, "Security violation: Mandatory Access Control was globally disabled."

        # 2. Check if deny rule is still blocking in /etc/apparmor.d/
        chk_profile = await sandbox.execute(
            "cat /etc/apparmor.d/opt.custom_daemon.daemon.sh 2>/dev/null || true"
        )
        if "deny /etc/**" in chk_profile.stdout:
            return False, "AppArmor profile still contains restrictive 'deny /etc/**' rule."

        # 3. Test daemon binary execution
        res_run = await sandbox.execute("/opt/custom_daemon/daemon.sh")
        if "CUSTOM_DAEMON_OK" in res_run.stdout:
            return True, "MAC profile corrected and daemon operating under confinement."

        return False, f"Daemon execution failed: {res_run.stderr or res_run.stdout}"
