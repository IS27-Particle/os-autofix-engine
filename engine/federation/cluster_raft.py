"""Lightweight Asynchronous Raft Consensus Engine and Distributed Lock Manager for Multi-Host Federation."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("os_autofix.engine.federation.raft")


class NodeRole(str, Enum):
    """Raft consensus role."""

    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"


@dataclass
class RaftMessage:
    """Raft inter-node RPC payload."""

    msg_type: str  # "REQUEST_VOTE", "VOTE_RESPONSE", "HEARTBEAT", "HEARTBEAT_ACK"
    term: int
    sender_id: str
    recipient_id: str
    vote_granted: bool = False
    leader_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedLock:
    """Cluster-wide distributed lock for coordinated multi-node remediation."""

    resource_key: str
    holder_node_id: str
    acquired_at: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        return time.time() > (self.acquired_at + self.ttl_seconds)


class ClusterRaftNode:
    """Asynchronous Raft node with distributed locking and cluster election."""

    def __init__(
        self,
        node_id: str,
        peers: list[str] | None = None,
        heartbeat_interval: float = 0.5,
        election_timeout_range: tuple[float, float] = (1.5, 3.0),
    ) -> None:
        self.node_id = node_id
        self.peers = peers or []
        self.heartbeat_interval = heartbeat_interval
        self.election_timeout_range = election_timeout_range

        self.role = NodeRole.FOLLOWER
        self.current_term = 0
        self.voted_for: str | None = None
        self.leader_id: str | None = None
        self.last_heartbeat_received = time.monotonic()

        self.distributed_locks: dict[str, DistributedLock] = {}
        self._running = False
        self._votes_received: set[str] = set()
        self._peer_nodes: dict[
            str, ClusterRaftNode
        ] = {}  # In-memory peer mesh for local simulation

    def register_in_memory_peer(self, peer: ClusterRaftNode) -> None:
        """Register direct peer reference for in-memory cluster federation and testing."""
        if peer.node_id != self.node_id:
            self._peer_nodes[peer.node_id] = peer
            if peer.node_id not in self.peers:
                self.peers.append(peer.node_id)

    def _reset_election_timeout(self) -> float:
        return random.uniform(*self.election_timeout_range)

    async def handle_message(self, msg: RaftMessage) -> RaftMessage | None:
        """Process incoming Raft consensus RPC."""
        # 1. Update term if message has higher term
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None

        # 2. Process message types
        if msg.msg_type == "REQUEST_VOTE":
            granted = False
            if msg.term >= self.current_term and (
                self.voted_for is None or self.voted_for == msg.sender_id
            ):
                granted = True
                self.voted_for = msg.sender_id
                self.last_heartbeat_received = time.monotonic()

            return RaftMessage(
                msg_type="VOTE_RESPONSE",
                term=self.current_term,
                sender_id=self.node_id,
                recipient_id=msg.sender_id,
                vote_granted=granted,
            )

        elif msg.msg_type == "HEARTBEAT":
            if msg.term >= self.current_term:
                self.role = NodeRole.FOLLOWER
                self.leader_id = msg.leader_id
                self.last_heartbeat_received = time.monotonic()

                # Sync lock states from leader
                if "locks" in msg.payload:
                    for k, l_data in msg.payload["locks"].items():
                        self.distributed_locks[k] = DistributedLock(
                            resource_key=l_data["resource_key"],
                            holder_node_id=l_data["holder_node_id"],
                            acquired_at=l_data["acquired_at"],
                            ttl_seconds=l_data["ttl_seconds"],
                        )

            return RaftMessage(
                msg_type="HEARTBEAT_ACK",
                term=self.current_term,
                sender_id=self.node_id,
                recipient_id=msg.sender_id,
            )

        elif msg.msg_type == "VOTE_RESPONSE":
            if (
                self.role == NodeRole.CANDIDATE
                and msg.vote_granted
                and msg.term == self.current_term
            ):
                self._votes_received.add(msg.sender_id)
                cluster_size = len(self.peers) + 1
                majority = (cluster_size // 2) + 1
                if len(self._votes_received) >= majority:
                    self._become_leader()

        return None

    def _become_leader(self) -> None:
        """Transition node state to cluster leader."""
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id
        logger.info(
            "[%s] Won election for Term %d. Transitioned to LEADER.",
            self.node_id,
            self.current_term,
        )

    async def start_election(self) -> None:
        """Initiate new leader election campaign."""
        self.current_term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.node_id
        self._votes_received = {self.node_id}
        self.last_heartbeat_received = time.monotonic()

        logger.info("[%s] Initiating election for Term %d...", self.node_id, self.current_term)

        cluster_size = len(self.peers) + 1
        if cluster_size == 1:
            self._become_leader()
            return

        for peer_id in self.peers:
            msg = RaftMessage(
                msg_type="REQUEST_VOTE",
                term=self.current_term,
                sender_id=self.node_id,
                recipient_id=peer_id,
            )
            # Dispatch to in-memory peer or network endpoint
            if peer_id in self._peer_nodes:
                peer = self._peer_nodes[peer_id]
                resp = await peer.handle_message(msg)
                if resp:
                    await self.handle_message(resp)

    async def broadcast_heartbeats(self) -> None:
        """Broadcast periodic append-entries heartbeats to all followers."""
        if self.role != NodeRole.LEADER:
            return

        # Clean expired locks
        self._cleanup_expired_locks()

        locks_payload = {
            k: {
                "resource_key": lock_obj.resource_key,
                "holder_node_id": lock_obj.holder_node_id,
                "acquired_at": lock_obj.acquired_at,
                "ttl_seconds": lock_obj.ttl_seconds,
            }
            for k, lock_obj in self.distributed_locks.items()
            if not lock_obj.is_expired
        }

        for peer_id in self.peers:
            msg = RaftMessage(
                msg_type="HEARTBEAT",
                term=self.current_term,
                sender_id=self.node_id,
                recipient_id=peer_id,
                leader_id=self.node_id,
                payload={"locks": locks_payload},
            )
            if peer_id in self._peer_nodes:
                await self._peer_nodes[peer_id].handle_message(msg)

    def _cleanup_expired_locks(self) -> None:
        """Purge locks that have exceeded their TTL."""
        expired = [k for k, lock_obj in self.distributed_locks.items() if lock_obj.is_expired]
        for k in expired:
            del self.distributed_locks[k]

    def acquire_lock(self, resource_key: str, ttl_seconds: float = 30.0) -> bool:
        """Acquire a cluster-wide distributed lock (must be granted by leader or local state)."""
        self._cleanup_expired_locks()
        if resource_key in self.distributed_locks:
            current_lock = self.distributed_locks[resource_key]
            if not current_lock.is_expired and current_lock.holder_node_id != self.node_id:
                logger.warning(
                    "[%s] Lock '%s' already held by '%s'",
                    self.node_id,
                    resource_key,
                    current_lock.holder_node_id,
                )
                return False

        lock = DistributedLock(
            resource_key=resource_key,
            holder_node_id=self.node_id,
            acquired_at=time.time(),
            ttl_seconds=ttl_seconds,
        )
        self.distributed_locks[resource_key] = lock
        logger.info(
            "[%s] Acquired distributed lock '%s' (TTL=%.1fs)",
            self.node_id,
            resource_key,
            ttl_seconds,
        )
        return True

    def release_lock(self, resource_key: str) -> bool:
        """Release an acquired distributed lock."""
        if resource_key in self.distributed_locks:
            if self.distributed_locks[resource_key].holder_node_id == self.node_id:
                del self.distributed_locks[resource_key]
                logger.info("[%s] Released distributed lock '%s'", self.node_id, resource_key)
                return True
        return False

    async def run(self, max_ticks: int | None = None) -> None:
        """Main asynchronous Raft consensus loop."""
        self._running = True
        timeout = self._reset_election_timeout()
        ticks = 0

        while self._running:
            if max_ticks and ticks >= max_ticks:
                break
            ticks += 1

            if self.role == NodeRole.LEADER:
                await self.broadcast_heartbeats()
                await asyncio.sleep(self.heartbeat_interval)
            else:
                elapsed = time.monotonic() - self.last_heartbeat_received
                if elapsed > timeout:
                    logger.warning(
                        "[%s] Heartbeat timeout (%.2fs > %.2fs). Starting election...",
                        self.node_id,
                        elapsed,
                        timeout,
                    )
                    await self.start_election()
                    timeout = self._reset_election_timeout()
                await asyncio.sleep(0.1)

    def stop(self) -> None:
        """Halt consensus engine."""
        self._running = False

    def get_cluster_status(self) -> dict[str, Any]:
        """Summary of current node and cluster consensus status."""
        self._cleanup_expired_locks()
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "term": self.current_term,
            "leader_id": self.leader_id,
            "peer_count": len(self.peers),
            "active_locks_count": len(self.distributed_locks),
            "active_locks": [
                {
                    "resource": lock_obj.resource_key,
                    "holder": lock_obj.holder_node_id,
                    "ttl": lock_obj.ttl_seconds,
                }
                for lock_obj in self.distributed_locks.values()
            ],
        }
