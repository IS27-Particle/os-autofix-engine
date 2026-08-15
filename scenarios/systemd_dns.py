"""Scenario: Systemd DNS resolution failure and broken resolv.conf."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.systemd_dns")


class SystemdDNSScenario(BaseScenario):
    """Corrupts systemd-resolved and resolv.conf, asserting full name resolution restoration."""

    name = "systemd_dns"
    description = "Domain name resolution has failed. Users report curl and network utilities cannot resolve hostnames."
    category = "Networking / DNS"
    difficulty = "medium"
    max_steps = 8

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Ensure base network tools and systemd-resolved are ready."""
        logger.info("Setting up baseline environment for %s...", self.name)
        res = await sandbox.execute(
            "which systemctl && (systemctl is-active systemd-resolved || systemctl start systemd-resolved || true)"
        )
        return res.exit_code == 0

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Disable systemd-resolved and inject a non-functional nameserver into resolv.conf."""
        logger.info("Injecting DNS fault into sandbox...")
        fault_script = """
systemctl stop systemd-resolved 2>/dev/null || true
systemctl disable systemd-resolved 2>/dev/null || true
rm -f /etc/resolv.conf
cat << 'EOF' > /etc/resolv.conf
# Corrupted resolver configuration
nameserver 127.0.0.99
options timeout:1 attempts:1
EOF
"""
        res = await sandbox.execute(fault_script)
        return res.exit_code == 0

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify that DNS queries resolve successfully."""
        logger.info("Verifying DNS resolution in %s...", self.name)
        test_script = """
python3 -c "
import socket
try:
    info = socket.getaddrinfo('dns.google', 53)
    if info:
        print('DNS_RESOLVED_OK')
        exit(0)
except Exception as e:
    print('DNS_ERROR:', e)
    exit(1)
"
"""
        res = await sandbox.execute(test_script, timeout_seconds=8)
        if res.exit_code == 0 and "DNS_RESOLVED_OK" in res.stdout:
            return True, "DNS query to 'dns.google' resolved successfully."

        fallback_res = await sandbox.execute("getent ahosts one.one.one.one", timeout_seconds=8)
        if fallback_res.exit_code == 0 and fallback_res.stdout.strip():
            return True, "DNS resolution verified via getent ahosts."

        return False, f"DNS resolution failed. Stderr: {res.stderr}\nStdout: {res.stdout}"
