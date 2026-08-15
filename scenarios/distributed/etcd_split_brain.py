"""Distributed 3-node etcd / Raft cluster network partition and split-brain recovery scenario."""

from __future__ import annotations

import logging

from sandbox.base import BaseSandbox
from scenarios.distributed.base_distributed import BaseDistributedScenario

logger = logging.getLogger("os_autofix.scenarios.distributed.etcd")


class EtcdSplitBrainScenario(BaseDistributedScenario):
    """3-Node etcd / Raft cluster network partition and quorum recovery scenario."""

    name: str = "etcd_split_brain"
    description: str = (
        "3-node etcd cluster (etcd-1, etcd-2, etcd-3) has lost consensus quorum. "
        "An asymmetrical iptables firewall DROP rule on etcd-1 isolates it from peer heartbeats on port 2380."
    )
    category: str = "Distributed Consensus / Raft"
    difficulty: str = "hard"
    max_steps: int = 8
    required_nodes: list[str] = ["etcd-1", "etcd-2", "etcd-3"]

    async def setup_topology(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Initialize mock etcd cluster member markers and peers."""
        for idx, (node_name, sb) in enumerate(nodes.items(), start=1):
            await sb.execute("mkdir -p /var/lib/etcd /etc/etcd")
            await sb.execute(
                f"echo 'ETCD_NAME={node_name}\nETCD_INITIAL_CLUSTER=etcd-1=http://10.0.0.11:2380,etcd-2=http://10.0.0.12:2380,etcd-3=http://10.0.0.13:2380' > /etc/etcd/etcd.conf"
            )
            await sb.execute(
                f"echo 'MEMBER_ID=etcd-0{idx}\nLEADER=etcd-2' > /var/lib/etcd/member_state.txt"
            )
        return True

    async def inject_fault(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Inject firewall drop on etcd-1 peer port 2380 to break Raft consensus."""
        if "etcd-1" in nodes:
            await nodes["etcd-1"].execute(
                "iptables -A INPUT -p tcp --dport 2380 -j DROP 2>/dev/null || true"
            )
            await nodes["etcd-1"].execute(
                "echo 'ETCD_PARTITION_ACTIVE=1' > /tmp/etcd_partition.flag"
            )
            await nodes["etcd-1"].execute(
                "echo 'MEMBER_ID=etcd-01\nLEADER=NONE' > /var/lib/etcd/member_state.txt"
            )
        return True

    async def verify(self, nodes: dict[str, BaseSandbox]) -> tuple[bool, str]:
        """Verify firewall is unblocked and all nodes see active cluster quorum."""
        if "etcd-1" in nodes:
            res_flag = await nodes["etcd-1"].execute("cat /tmp/etcd_partition.flag 2>/dev/null")
            res_rules = await nodes["etcd-1"].execute("iptables -L INPUT -n 2>/dev/null")
            if "ETCD_PARTITION_ACTIVE" in res_flag.stdout or "DROP" in res_rules.stdout:
                return (
                    False,
                    "etcd-1 has an active firewall DROP rule blocking Raft peer port 2380.",
                )

            res_state = await nodes["etcd-1"].execute(
                "cat /var/lib/etcd/member_state.txt 2>/dev/null"
            )
            if "LEADER=NONE" in res_state.stdout:
                return False, "etcd-1 is not reconverged to cluster consensus."

        return (
            True,
            "etcd 3-node cluster has full consensus quorum and unpartitioned Raft peer networking.",
        )
