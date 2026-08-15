"""Unit tests for the Causal Fault Graph and Root-Cause Tracer."""

from __future__ import annotations

import pytest

from engine.causal_tracer import CausalGraph, CausalTracer
from tests.conftest import MockSandbox


def test_causal_graph_root_cause_ranking() -> None:
    """Test identifying root triggers vs downstream symptoms in causal dependency graph."""
    graph = CausalGraph()

    # Build dependency chain: nginx -> uwsgi -> database_socket -> missing_mount
    graph.add_node("nginx.service", "Nginx Web Server", "service", "failed")
    graph.add_node("uwsgi.service", "uWSGI App Daemon", "service", "failed")
    graph.add_node("db_socket", "/run/mysql/mysqld.sock", "socket", "failed")
    graph.add_node("mount_storage", "/mnt/data", "file", "failed")

    # Edges: source depends on target (target is upstream root)
    graph.add_edge("nginx.service", "uwsgi.service", relation="depends_on")
    graph.add_edge("uwsgi.service", "db_socket", relation="binds_to")
    graph.add_edge("db_socket", "mount_storage", relation="requires_mount")

    hypotheses = graph.find_root_causes()
    assert len(hypotheses) == 4

    # Top root cause should be mount_storage (0 upstream failures, 1+ downstream dependents)
    top = hypotheses[0]
    assert top.node_id == "mount_storage"
    assert top.confidence_score >= 0.85
    assert "Root Trigger" in top.summary

    # Nginx should have lowest confidence as a downstream symptom
    assert hypotheses[-1].node_id == "nginx.service"
    assert "Downstream Symptom" in hypotheses[-1].summary


def test_causal_graph_mermaid_export() -> None:
    """Test Mermaid DAG flowchart generation."""
    graph = CausalGraph()
    graph.add_node("dns_service", "systemd-resolved", "service", "failed")
    graph.add_node("resolv_file", "/etc/resolv.conf", "file", "failed")
    graph.add_edge("resolv_file", "dns_service", relation="generated_by")

    mermaid = graph.to_mermaid()
    assert "graph TD" in mermaid
    assert 'dns_service["systemd-resolved (service)"]:::failed' in mermaid
    assert "resolv_file -->|generated_by| dns_service" in mermaid


@pytest.mark.asyncio
async def test_causal_tracer_sandbox_inspection() -> None:
    """Test building causal graph from sandbox system commands."""
    sandbox = MockSandbox("tracer-test")
    tracer = CausalTracer()

    graph = await tracer.trace_sandbox(sandbox)
    assert len(graph.nodes) >= 1
    dict_summary = graph.to_dict()
    assert "node_count" in dict_summary
