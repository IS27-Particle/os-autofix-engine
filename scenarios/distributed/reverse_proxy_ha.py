"""Distributed High-Availability Reverse Proxy scenario (Keepalived + HAProxy with VIP failover)."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.distributed.base_distributed import BaseDistributedScenario

logger = logging.getLogger("os_autofix.scenarios.distributed.haproxy")


class ReverseProxyHAScenario(BaseDistributedScenario):
    """High-Availability Keepalived VRRP + HAProxy dual load balancer failover scenario."""

    name: str = "reverse_proxy_ha"
    description: str = (
        "Dual HAProxy/Keepalived load balancer pair fronting backend web nodes. "
        "The Master load balancer (lb-1) has a crashed VRRP daemon, and Backup (lb-2) is failing to acquire the VIP (10.0.0.200)."
    )
    category: str = "High Availability / Load Balancing"
    difficulty: str = "medium"
    max_steps: int = 8
    required_nodes: list[str] = ["lb-1", "lb-2", "backend-1", "backend-2"]

    async def setup_topology(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Configure baseline keepalived and haproxy state."""
        if "lb-1" in nodes:
            await nodes["lb-1"].execute("mkdir -p /etc/keepalived /etc/haproxy")
            await nodes["lb-1"].execute(
                "echo 'vrrp_instance VI_1 { state MASTER virtual_ipaddress { 10.0.0.200 } }' > /etc/keepalived/keepalived.conf"
            )
            await nodes["lb-1"].execute("echo 'KEEPALIVED_RUNNING=1' > /tmp/keepalived.status")
        if "lb-2" in nodes:
            await nodes["lb-2"].execute("mkdir -p /etc/keepalived /etc/haproxy")
            await nodes["lb-2"].execute(
                "echo 'vrrp_instance VI_1 { state BACKUP virtual_ipaddress { 10.0.0.200 } }' > /etc/keepalived/keepalived.conf"
            )
            await nodes["lb-2"].execute("echo 'KEEPALIVED_RUNNING=1' > /tmp/keepalived.status")
        for b_name in ["backend-1", "backend-2"]:
            if b_name in nodes:
                await nodes[b_name].execute("echo 'HTTP_BACKEND_OK=1' > /tmp/backend.status")
        return True

    async def inject_fault(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Inject crashed VRRP on lb-1 and VIP binding lockout on lb-2."""
        if "lb-1" in nodes:
            await nodes["lb-1"].execute("rm -f /tmp/keepalived.status")
            await nodes["lb-1"].execute("echo 'VRRP_CRASHED=1' > /tmp/ha_fault.flag")

        if "lb-2" in nodes:
            # Misconfigured priority prevents VIP takeover
            await nodes["lb-2"].execute("echo 'VIP_TAKEOVER_FAILED=1' > /tmp/ha_fault.flag")
        return True

    async def verify(self, nodes: dict[str, BaseSandbox]) -> tuple[bool, str]:
        """Verify either lb-1 or lb-2 is actively holding the VIP and routing traffic."""
        lb1_healthy = False
        lb2_healthy = False

        if "lb-1" in nodes:
            res_lb1_flag = await nodes["lb-1"].execute("cat /tmp/ha_fault.flag 2>/dev/null")
            res_lb1_stat = await nodes["lb-1"].execute("cat /tmp/keepalived.status 2>/dev/null")
            if (
                "VRRP_CRASHED" not in res_lb1_flag.stdout
                and "KEEPALIVED_RUNNING=1" in res_lb1_stat.stdout
            ):
                lb1_healthy = True

        if "lb-2" in nodes:
            res_lb2_flag = await nodes["lb-2"].execute("cat /tmp/ha_fault.flag 2>/dev/null")
            if "VIP_TAKEOVER_FAILED" not in res_lb2_flag.stdout:
                lb2_healthy = True

        if not lb1_healthy and not lb2_healthy:
            return (
                False,
                "Neither load balancer (lb-1, lb-2) is healthy and able to serve the Virtual IP (10.0.0.200).",
            )

        return (
            True,
            "High-Availability reverse proxy cluster is healthy with active VIP failover capability.",
        )
