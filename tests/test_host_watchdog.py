"""Unit tests for the Host Self-Healing Watchdog Daemon."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.host_watchdog import HostWatchdogDaemon
from tests.conftest import MockSandbox


def test_journal_pattern_parsing() -> None:
    """Test matching journal logs to inferred fault scenarios."""
    daemon = HostWatchdogDaemon()

    # 1. DNS Anomaly
    ev1 = daemon.parse_journal_line(
        "systemd[1]: Failed to start systemd-resolved.service: Unit is broken."
    )
    assert ev1 is not None
    assert ev1.anomaly_type == "DNS_FAILURE"
    assert ev1.inferred_scenario == "systemd_dns"

    # 2. Docker socket lockout
    ev2 = daemon.parse_journal_line("dockerd[902]: /var/run/docker.sock permission denied")
    assert ev2 is not None
    assert ev2.anomaly_type == "DOCKER_LOCKOUT"
    assert ev2.inferred_scenario == "docker_socket"

    # 3. Normal log line (no match)
    ev3 = daemon.parse_journal_line("kernel: Linux version 6.8.0-generic (buildd@canonical)")
    assert ev3 is None


@pytest.mark.asyncio
async def test_shadow_validation_dry_run() -> None:
    """Test shadow container dry-run execution without host mutation."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    daemon = HostWatchdogDaemon(
        config=cfg,
        dry_run=True,
        min_safety_score=0.85,
        sandbox_factory=mock_factory,
    )

    ev = daemon.parse_journal_line(
        "systemd-resolved[12]: DNS server failure, nameserver refused query"
    )
    assert ev is not None

    report = await daemon.execute_shadow_validation(ev)
    assert report.shadow_verified is True
    assert report.host_applied is False  # Dry-run protected
    assert "DRY-RUN mode" in report.notes
    assert report.safety_score >= 0.85
