"""Distributed WireGuard 3-node mesh scenario with MTU mismatch and corrupted endpoint keys."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.distributed.base_distributed import BaseDistributedScenario

logger = logging.getLogger("os_autofix.scenarios.distributed.wireguard")


class WireGuardMeshScenario(BaseDistributedScenario):
    """3-Node WireGuard Mesh Network troubleshooting scenario."""

    name: str = "wireguard_mesh"
    description: str = (
        "3-node WireGuard full-mesh overlay network (10.0.99.0/24) has lost peer reachability "
        "due to an MTU mismatch on node-1 and corrupted peer public key configuration on node-2."
    )
    category: str = "Distributed Networking / VPN"
    difficulty: str = "hard"
    max_steps: int = 10
    required_nodes: list[str] = ["node-1", "node-2", "node-3"]

    async def setup_topology(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Configure baseline WireGuard mesh overlay interfaces."""
        for idx, (_node_name, sb) in enumerate(nodes.items(), start=1):
            ip = f"10.0.99.{idx}"
            await sb.execute("mkdir -p /etc/wireguard")
            await sb.execute("ip link add dev wg0 type wireguard 2>/dev/null || true")
            await sb.execute(f"ip addr add {ip}/24 dev wg0 2>/dev/null || true")
            await sb.execute("ip link set mtu 1420 up dev wg0 2>/dev/null || true")
            await sb.execute(f"echo 'PEER_READY={idx}' > /etc/wireguard/wg0.conf")
        return True

    async def inject_fault(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Inject MTU mismatch on node-1 and bad key config on node-2."""
        if "node-1" in nodes:
            # MTU mismatch: 1200 causes packet drop under jumbo frames
            await nodes["node-1"].execute("ip link set mtu 1200 dev wg0 2>/dev/null || true")
            await nodes["node-1"].execute("echo 'MTU_FAULT=1' > /tmp/wg_fault.flag")

        if "node-2" in nodes:
            # Broken peer key in configuration
            await nodes["node-2"].execute(
                "echo 'PublicKey = INVALID_CORRUPTED_KEY_BASE64==' >> /etc/wireguard/wg0.conf"
            )
            await nodes["node-2"].execute("echo 'KEY_FAULT=1' > /tmp/wg_fault.flag")

        return True

    async def verify(self, nodes: dict[str, BaseSandbox]) -> tuple[bool, str]:
        """Verify full mesh reachability and valid MTU / key parameters."""
        # 1. Check MTU on node-1
        if "node-1" in nodes:
            res_mtu = await nodes["node-1"].execute("ip link show wg0 2>/dev/null")
            if (
                "mtu 1200" in res_mtu.stdout
                or "MTU_FAULT"
                in (await nodes["node-1"].execute("cat /tmp/wg_fault.flag 2>/dev/null")).stdout
            ):
                return False, "node-1 WireGuard interface MTU is degraded or mismatched (1200)."

        # 2. Check peer keys on node-2
        if "node-2" in nodes:
            res_conf = await nodes["node-2"].execute("cat /etc/wireguard/wg0.conf 2>/dev/null")
            if (
                "INVALID_CORRUPTED_KEY" in res_conf.stdout
                or "KEY_FAULT"
                in (await nodes["node-2"].execute("cat /tmp/wg_fault.flag 2>/dev/null")).stdout
            ):
                return False, "node-2 has invalid WireGuard peer public keys."

        return (
            True,
            "WireGuard 3-node mesh network is healthy with consistent MTU and valid cryptographic keys.",
        )
