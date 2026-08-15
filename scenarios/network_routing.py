"""Scenario: Default network gateway and routing table misconfiguration."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger("os_autofix.scenarios.network_routing")


class NetworkRoutingScenario(BaseScenario):
    """Corrupts default gateway routing tables and asserts route/gateway connectivity."""

    name = "network_routing"
    description = "Default network gateway route is broken. Outbound packet routing is failing across all interfaces."
    category = "Networking / Routing"
    difficulty = "medium"
    max_steps = 8

    async def setup(self, sandbox: BaseSandbox) -> bool:
        """Capture and preserve the original valid gateway for verification."""
        logger.info("Setting up baseline network routing state for %s...", self.name)
        save_gw_script = """
GW=$(ip route show default | awk '{print $3}' | head -n 1)
IFACE=$(ip route show default | awk '{print $5}' | head -n 1)
if [ -n "$GW" ]; then
    echo "$GW" > /root/.orig_gw
    echo "$IFACE" > /root/.orig_iface
fi
"""
        res = await sandbox.execute(save_gw_script)
        return res.exit_code == 0

    async def inject_fault(self, sandbox: BaseSandbox) -> bool:
        """Delete default gateway route and insert an unreachable blackhole gateway."""
        logger.info("Injecting network routing fault...")
        fault_script = """
# Remove all active default routes
while ip route del default 2>/dev/null; do :; done

# Inject bogus unreachable route
ip route add default via 192.0.2.254 metric 10 2>/dev/null || true
"""
        res = await sandbox.execute(fault_script)
        return res.exit_code == 0

    async def verify(self, sandbox: BaseSandbox) -> tuple[bool, str]:
        """Verify that a valid default gateway route exists and can route packets."""
        logger.info("Verifying network routing in %s...", self.name)
        verify_script = """
# Check if default route exists
DEFAULT_ROUTE=$(ip route show default | grep -v '192.0.2.254')
if [ -z "$DEFAULT_ROUTE" ]; then
    echo "NO_VALID_DEFAULT_ROUTE"
    exit 1
fi

# Check route resolution to public IP
ip route get 1.1.1.1 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ROUTE_GET_FAILED"
    exit 2
fi

# Verify ping to gateway if saved
if [ -f /root/.orig_gw ]; then
    GW=$(cat /root/.orig_gw)
    if [ -n "$GW" ]; then
        ping -c 1 -W 2 "$GW" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "GATEWAY_UNREACHABLE"
            exit 3
        fi
    fi
fi

echo "ROUTING_HEALTHY"
exit 0
"""
        res = await sandbox.execute(verify_script, timeout_seconds=8)
        if res.exit_code == 0 and "ROUTING_HEALTHY" in res.stdout:
            return True, "Default network routing and gateway connectivity successfully verified."

        return (
            False,
            f"Routing verification failed (exit code {res.exit_code}): {res.combined_output}",
        )
