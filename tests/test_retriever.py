"""Unit tests for the Offline Hybrid Documentation & Runbook Retriever."""

from __future__ import annotations

from pathlib import Path

from knowledge.retriever import HybridRetriever


def test_markdown_chunking(tmp_path: Path) -> None:
    """Test splitting Markdown runbooks into structured section chunks."""
    doc_path = tmp_path / "test_runbook.md"
    doc_path.write_text(
        "# My Test Daemon Runbook\n\n"
        "## Overview\nThis daemon manages network routing.\n\n"
        "## Common Root Causes\nGateway routes lost.\n\n"
        "## Remediation Commands\n`ip route add default via 10.0.0.1`\n",
        encoding="utf-8",
    )

    retriever = HybridRetriever(runbooks_dir=tmp_path)
    chunks = retriever.chunk_markdown(doc_path)

    assert len(chunks) == 3
    assert chunks[0].section == "Overview"
    assert chunks[1].section == "Common Root Causes"
    assert chunks[2].section == "Remediation Commands"
    assert "10.0.0.1" in chunks[2].content


def test_hybrid_search_bm25_retrieval() -> None:
    """Test indexing pre-populated runbooks and querying with BM25."""
    retriever = HybridRetriever()
    count = retriever.index_all()
    assert count >= 5

    # Query DNS
    dns_res = retriever.consult_runbook("DNS resolution failed in systemd-resolved", top_k=2)
    assert len(dns_res) > 0
    assert any("DNS" in r.title or "systemd_dns" in r.source_file for r in dns_res)
    assert dns_res[0].score > 0

    # Query ZFS
    zfs_res = retriever.consult_runbook("zpool dataset mount failed", top_k=2)
    assert len(zfs_res) > 0
    assert any("ZFS" in r.title or "zfs_storage" in r.source_file for r in zfs_res)


def test_manpage_search_fallback() -> None:
    """Test search_manpages helper."""
    retriever = HybridRetriever()
    man_out = retriever.search_manpages("nonexistent_fake_daemon_xyz123")
    assert "No manual entry found" in man_out
