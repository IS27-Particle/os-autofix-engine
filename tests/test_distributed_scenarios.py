"""Unit tests for Multi-Node Distributed Topology Scenarios (WireGuard, etcd, HAProxy)."""

from __future__ import annotations

import pytest

from sandbox.base import BaseSandbox
from scenarios.distributed import (
    EtcdSplitBrainScenario,
    ReverseProxyHAScenario,
    WireGuardMeshScenario,
    get_all_distributed_scenarios,
    get_distributed_scenario,
)
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_wireguard_mesh_scenario_lifecycle() -> None:
    """Test full multi-node lifecycle for WireGuard mesh overlay."""
    sc = WireGuardMeshScenario()
    nodes: dict[str, BaseSandbox] = {
        "node-1": MockSandbox("wg-node-1"),
        "node-2": MockSandbox("wg-node-2"),
        "node-3": MockSandbox("wg-node-3"),
    }

    # 1. Setup
    assert await sc.setup_topology(nodes) is True
    ok_base, msg_base = await sc.verify(nodes)
    assert ok_base is True

    # 2. Inject Fault
    assert await sc.inject_fault(nodes) is True
    ok_fault, msg_fault = await sc.verify(nodes)
    assert ok_fault is False
    assert "WireGuard" in msg_fault

    # 3. Remediate
    await nodes["node-1"].execute("rm -f /tmp/wg_fault.flag")
    await nodes["node-2"].execute("echo 'PEER_READY=2' > /etc/wireguard/wg0.conf")
    await nodes["node-2"].execute("rm -f /tmp/wg_fault.flag")
    ok_remed, _ = await sc.verify(nodes)
    assert ok_remed is True


@pytest.mark.asyncio
async def test_etcd_split_brain_scenario_lifecycle() -> None:
    """Test full multi-node lifecycle for etcd consensus split-brain."""
    sc = EtcdSplitBrainScenario()
    nodes: dict[str, BaseSandbox] = {
        "etcd-1": MockSandbox("etcd-node-1"),
        "etcd-2": MockSandbox("etcd-node-2"),
        "etcd-3": MockSandbox("etcd-node-3"),
    }

    # 1. Setup
    assert await sc.setup_topology(nodes) is True
    ok_base, msg_base = await sc.verify(nodes)
    assert ok_base is True

    # 2. Inject Fault
    assert await sc.inject_fault(nodes) is True
    ok_fault, msg_fault = await sc.verify(nodes)
    assert ok_fault is False
    assert "etcd-1" in msg_fault

    # 3. Remediate
    await nodes["etcd-1"].execute("rm -f /tmp/etcd_partition.flag")
    await nodes["etcd-1"].execute(
        "echo 'MEMBER_ID=etcd-01\nLEADER=etcd-2' > /var/lib/etcd/member_state.txt"
    )
    ok_remed, _ = await sc.verify(nodes)
    assert ok_remed is True


@pytest.mark.asyncio
async def test_reverse_proxy_ha_scenario_lifecycle() -> None:
    """Test full multi-node lifecycle for HAProxy / Keepalived failover."""
    sc = ReverseProxyHAScenario()
    nodes: dict[str, BaseSandbox] = {
        "lb-1": MockSandbox("lb-1"),
        "lb-2": MockSandbox("lb-2"),
        "backend-1": MockSandbox("backend-1"),
        "backend-2": MockSandbox("backend-2"),
    }

    # 1. Setup
    assert await sc.setup_topology(nodes) is True
    ok_base, msg_base = await sc.verify(nodes)
    assert ok_base is True

    # 2. Inject Fault
    assert await sc.inject_fault(nodes) is True
    ok_fault, msg_fault = await sc.verify(nodes)
    assert ok_fault is False

    # 3. Remediate (start keepalived on lb-1)
    await nodes["lb-1"].execute("rm -f /tmp/ha_fault.flag")
    await nodes["lb-1"].execute("echo 'KEEPALIVED_RUNNING=1' > /tmp/keepalived.status")
    ok_remed, _ = await sc.verify(nodes)
    assert ok_remed is True


def test_distributed_registry() -> None:
    """Test registry lookups for distributed scenarios."""
    scenarios = get_all_distributed_scenarios()
    assert len(scenarios) == 3
    sc = get_distributed_scenario("wireguard_mesh")
    assert sc.name == "wireguard_mesh"
