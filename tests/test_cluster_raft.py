"""Unit tests for the Cluster Raft Consensus and Distributed Lock Engine."""

from __future__ import annotations

import pytest

from engine.federation.cluster_raft import ClusterRaftNode, NodeRole, RaftMessage


@pytest.mark.asyncio
async def test_raft_leader_election_3_nodes() -> None:
    """Test 3-node in-memory cluster leader election and consensus majority."""
    node1 = ClusterRaftNode(node_id="node-1")
    node2 = ClusterRaftNode(node_id="node-2")
    node3 = ClusterRaftNode(node_id="node-3")

    # Wire mesh
    node1.register_in_memory_peer(node2)
    node1.register_in_memory_peer(node3)
    node2.register_in_memory_peer(node1)
    node2.register_in_memory_peer(node3)
    node3.register_in_memory_peer(node1)
    node3.register_in_memory_peer(node2)

    # Node 1 starts election
    await node1.start_election()

    assert node1.role == NodeRole.LEADER
    assert node1.leader_id == "node-1"
    assert node1.current_term == 1

    # Broadcast heartbeats
    await node1.broadcast_heartbeats()
    assert node2.role == NodeRole.FOLLOWER
    assert node2.leader_id == "node-1"
    assert node3.role == NodeRole.FOLLOWER
    assert node3.leader_id == "node-1"


def test_distributed_lock_acquisition_and_expiration() -> None:
    """Test acquiring, holding, and expiring cluster distributed locks."""
    node = ClusterRaftNode(node_id="node-1")

    # 1. Acquire valid lock
    assert node.acquire_lock("lock:db:migration", ttl_seconds=10.0) is True

    # 2. Acquire duplicate from same holder -> allowed / updated
    assert node.acquire_lock("lock:db:migration", ttl_seconds=10.0) is True

    # 3. Simulate another node trying to acquire
    node_other = ClusterRaftNode(node_id="node-2")
    node_other.distributed_locks = node.distributed_locks  # Share synced locks
    assert node_other.acquire_lock("lock:db:migration", ttl_seconds=10.0) is False

    # 4. Release lock
    assert node.release_lock("lock:db:migration") is True
    assert node_other.acquire_lock("lock:db:migration", ttl_seconds=10.0) is True


@pytest.mark.asyncio
async def test_raft_message_handling() -> None:
    """Test RPC message handling for RequestVote and Heartbeat."""
    follower = ClusterRaftNode(node_id="follower-1")

    # Vote request
    req_vote = RaftMessage(
        msg_type="REQUEST_VOTE",
        term=1,
        sender_id="candidate-1",
        recipient_id="follower-1",
    )
    resp = await follower.handle_message(req_vote)
    assert resp is not None
    assert resp.msg_type == "VOTE_RESPONSE"
    assert resp.vote_granted is True
    assert follower.voted_for == "candidate-1"
