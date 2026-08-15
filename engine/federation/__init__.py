"""Federation, Raft consensus, and cross-host remediation package."""

from engine.federation.cluster_raft import (
    ClusterRaftNode,
    DistributedLock,
    NodeRole,
    RaftMessage,
)
from engine.federation.cross_host_coordinator import (
    CrossHostRemediationEngine,
    CrossHostRemediationResult,
)

__all__ = [
    "ClusterRaftNode",
    "NodeRole",
    "RaftMessage",
    "DistributedLock",
    "CrossHostRemediationEngine",
    "CrossHostRemediationResult",
]
