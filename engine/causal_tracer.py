"""Autonomous Causal Fault Graph and Root-Cause Tracer for OS-level Failure Diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.engine.causal_tracer")


@dataclass
class CausalNode:
    """Represents a component or state within the OS causal dependency graph."""

    node_id: str
    label: str
    node_type: str  # "service", "socket", "file", "network", "security"
    state: str  # "healthy", "degraded", "failed"
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """Directed dependency or causality edge between components."""

    source_id: str  # Downstream affected component
    target_id: str  # Upstream root/dependency component
    relation: str  # "depends_on", "binds_to", "requires_config", "blocked_by"


@dataclass
class RootCauseHypothesis:
    """Ranked root cause candidate with Bayesian-weighted confidence score."""

    node_id: str
    label: str
    node_type: str
    confidence_score: float  # [0.0 - 1.0]
    downstream_impact_count: int
    summary: str


class CausalGraph:
    """Directed Acyclic Graph (DAG) modeling system dependencies and failure propagation."""

    def __init__(self) -> None:
        self.nodes: dict[str, CausalNode] = {}
        self.edges: list[CausalEdge] = []

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str = "service",
        state: str = "healthy",
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CausalNode:
        node = CausalNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            state=state,
            error_message=error_message,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, relation: str = "depends_on") -> None:
        self.edges.append(CausalEdge(source_id=source_id, target_id=target_id, relation=relation))

    def find_root_causes(self) -> list[RootCauseHypothesis]:
        """Compute ranked root-cause hypotheses by analyzing dependency topology and failure propagation."""
        failed_nodes = {
            nid: n for nid, n in self.nodes.items() if n.state in ("failed", "degraded")
        }
        if not failed_nodes:
            return []

        # Calculate downstream dependents count (how many components depend on this node)
        dependents: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        dependencies: dict[str, set[str]] = {nid: set() for nid in self.nodes}

        for edge in self.edges:
            # edge: source depends on target (target is upstream dependency)
            if edge.target_id in dependents and edge.source_id in self.nodes:
                dependents[edge.target_id].add(edge.source_id)
            if edge.source_id in dependencies and edge.target_id in self.nodes:
                dependencies[edge.source_id].add(edge.target_id)

        hypotheses: list[RootCauseHypothesis] = []

        for nid, node in failed_nodes.items():
            # Check if this failure is caused by an upstream failed dependency
            upstream_failed = [dep for dep in dependencies.get(nid, set()) if dep in failed_nodes]
            downstream_count = len(dependents.get(nid, set()))

            # Higher confidence if node has no upstream failed dependencies (it is the root trigger)
            if not upstream_failed:
                confidence = 0.85 + min(0.14, downstream_count * 0.05)
                summary = (
                    f"Root Trigger: {node.label} failed with zero failed dependencies. "
                    f"Impacting {downstream_count} downstream component(s)."
                )
            else:
                confidence = max(0.10, 0.60 - (len(upstream_failed) * 0.20))
                summary = (
                    f"Downstream Symptom: {node.label} failure likely induced by upstream dependency "
                    f"[{', '.join(upstream_failed)}]."
                )

            hypotheses.append(
                RootCauseHypothesis(
                    node_id=nid,
                    label=node.label,
                    node_type=node.node_type,
                    confidence_score=round(confidence, 2),
                    downstream_impact_count=downstream_count,
                    summary=summary,
                )
            )

        hypotheses.sort(key=lambda h: (h.confidence_score, h.downstream_impact_count), reverse=True)
        return hypotheses

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram string."""
        lines = ["graph TD"]
        for nid, n in self.nodes.items():
            color_style = (
                ":::failed"
                if n.state == "failed"
                else ":::degraded"
                if n.state == "degraded"
                else ""
            )
            lines.append(f'    {nid}["{n.label} ({n.node_type})"]{color_style}')

        for e in self.edges:
            lines.append(f"    {e.source_id} -->|{e.relation}| {e.target_id}")

        lines.append("    classDef failed fill:#e74c3c,stroke:#c0392b,color:#fff;")
        lines.append("    classDef degraded fill:#f39c12,stroke:#d35400,color:#fff;")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export serialized dictionary summary."""
        hypotheses = self.find_root_causes()
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "root_cause_count": len(hypotheses),
            "top_hypothesis": hypotheses[0].__dict__ if hypotheses else None,
            "hypotheses": [h.__dict__ for h in hypotheses],
            "nodes": [n.__dict__ for n in self.nodes.values()],
            "edges": [e.__dict__ for e in self.edges],
        }


class CausalTracer:
    """Gathers runtime OS signals and builds an actionable Causal Dependency Graph."""

    async def trace_sandbox(self, sandbox: BaseSandbox) -> CausalGraph:
        """Inspect sandbox environment and infer the causal failure topology."""
        graph = CausalGraph()

        # 1. Inspect failed units
        res_failed = await sandbox.execute(
            "systemctl list-units --state=failed --no-legend 2>/dev/null || true"
        )
        failed_units: set[str] = set()
        for line in res_failed.stdout.splitlines():
            parts = line.strip().split()
            if parts:
                unit = parts[0]
                failed_units.add(unit)
                graph.add_node(
                    node_id=unit,
                    label=unit,
                    node_type="service",
                    state="failed",
                    error_message=f"Unit {unit} reported failed state",
                )

        # 2. Inspect DNS / /etc/resolv.conf
        res_dns = await sandbox.execute("cat /etc/resolv.conf 2>/dev/null || true")
        if "127.0.0.53" in res_dns.stdout or "nameserver" in res_dns.stdout:
            graph.add_node("config_resolv_conf", "/etc/resolv.conf", "file", "healthy")
        else:
            graph.add_node(
                "config_resolv_conf",
                "/etc/resolv.conf",
                "file",
                "failed",
                error_message="Missing nameserver or empty resolv.conf",
            )

        if "systemd-resolved.service" in failed_units:
            graph.add_edge(
                "config_resolv_conf", "systemd-resolved.service", relation="generated_by"
            )

        # 3. Inspect open sockets
        await sandbox.execute("ss -tulpn 2>/dev/null || true")
        if "docker" in res_failed.stdout or "dockerd" in failed_units:
            graph.add_node(
                "socket_docker",
                "/var/run/docker.sock",
                "socket",
                "failed",
                error_message="Docker socket inaccessible",
            )
            graph.add_edge("docker.service", "socket_docker", relation="binds_to")

        # 4. If no failure nodes were added, add default system node
        if not graph.nodes:
            graph.add_node("system_root", "Linux Host Root", "system", "healthy")

        return graph
