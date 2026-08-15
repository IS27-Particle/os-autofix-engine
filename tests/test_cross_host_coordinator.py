"""Unit tests for the Distributed Cross-Host Remediation Engine."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.federation.cluster_raft import ClusterRaftNode
from engine.federation.cross_host_coordinator import CrossHostRemediationEngine
from sandbox.base import BaseSandbox
from scenarios.distributed.etcd_split_brain import EtcdSplitBrainScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_cross_host_remediation_workflow() -> None:
    """Test multi-node remediation coordination with Raft distributed locks."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    raft_node = ClusterRaftNode(node_id="test-orch-1")

    engine = CrossHostRemediationEngine(config=cfg, raft_node=raft_node)
    scenario = EtcdSplitBrainScenario()

    nodes: dict[str, BaseSandbox] = {
        "etcd-1": MockSandbox("etcd-node-1"),
        "etcd-2": MockSandbox("etcd-node-2"),
        "etcd-3": MockSandbox("etcd-node-3"),
    }

    # Setup topology and inject fault
    await scenario.setup_topology(nodes)
    await scenario.inject_fault(nodes)
    assert (await scenario.verify(nodes))[0] is False

    # Execute coordinated cross-host remediation
    res = await engine.remediate_distributed_topology(scenario, nodes)

    assert res.success is True
    assert res.lock_acquired is True
    assert res.reverted_on_failure is False
    assert len(res.node_results) == 3


@pytest.mark.asyncio
async def test_cross_host_remediation_lock_conflict() -> None:
    """Test aborting remediation when resource is locked by another orchestrator."""
    cfg = EngineConfig()
    raft_node = ClusterRaftNode(node_id="orch-1")

    # Manually hold lock under different node
    raft_node.acquire_lock("lock:topology:etcd_split_brain", ttl_seconds=100.0)
    raft_node.distributed_locks["lock:topology:etcd_split_brain"].holder_node_id = "orch-2"

    engine = CrossHostRemediationEngine(config=cfg, raft_node=raft_node)
    scenario = EtcdSplitBrainScenario()
    nodes: dict[str, BaseSandbox] = {"etcd-1": MockSandbox("sb-1")}

    res = await engine.remediate_distributed_topology(scenario, nodes)
    assert res.success is False
    assert res.lock_acquired is False
    assert "locked by another orchestrator" in res.details
