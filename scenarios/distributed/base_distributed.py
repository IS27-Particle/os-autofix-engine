"""Abstract base class for multi-node distributed topology scenarios."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.scenarios.distributed.base")


class BaseDistributedScenario(ABC):
    """Abstract harness for scenarios involving multi-instance networks and clustered workloads."""

    name: str
    description: str
    category: str
    difficulty: str
    max_steps: int
    required_nodes: list[str]  # e.g., ["node-1", "node-2", "node-3"]

    def __init__(self) -> None:
        if not hasattr(self, "name"):
            raise NotImplementedError("Distributed scenario must specify 'name'")
        if not hasattr(self, "required_nodes"):
            self.required_nodes = ["node-1", "node-2", "node-3"]

    def get_prompt(self) -> str:
        """Instruction prompt presented to the agent for distributed troubleshooting."""
        return (
            f"You are an expert SRE troubleshooting a distributed multi-node cluster scenario: '{self.name}'.\n"
            f"SYMPTOM DESCRIPTION:\n{self.description}\n\n"
            f"CLUSTER NODES: {', '.join(self.required_nodes)}\n"
            f"Your objective is to diagnose and restore health across all cluster nodes.\n"
            f"In your command JSON, specify the target node by prefixing with `@<node_name>` if multi-node execution is supported, "
            f"or execute cluster coordination commands."
        )

    @abstractmethod
    async def setup_topology(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Prepare baseline cluster configuration, networking, and services across all nodes."""
        pass

    @abstractmethod
    async def inject_fault(self, nodes: dict[str, BaseSandbox]) -> bool:
        """Inject network partitions, MTU mismatches, or daemon failures across target nodes."""
        pass

    @abstractmethod
    async def verify(self, nodes: dict[str, BaseSandbox]) -> tuple[bool, str]:
        """Assert multi-node cluster health, quorum, reachability, or VIP failover."""
        pass
