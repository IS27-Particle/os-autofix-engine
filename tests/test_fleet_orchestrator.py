"""Unit tests for the Progressive Canary Fleet Rollout Manager."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.fleet_orchestrator import FleetRolloutOrchestrator
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_fleet_progressive_canary_success() -> None:
    """Test full multi-tier canary rollout across a fleet of 5 instances."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    orchestrator = FleetRolloutOrchestrator(
        config=cfg,
        sandbox_factory=mock_factory,
        error_threshold=0.05,
        tiers=[0.20, 0.60, 1.00],
    )
    scenario = SystemdDNSScenario()

    res = await orchestrator.execute_fleet_rollout(
        scenario=scenario,
        fleet_size=5,
        patch_command="echo 'nameserver 1.1.1.1' > /etc/resolv.conf",
    )

    assert res.rollout_id.startswith("rollout-")
    assert res.total_fleet_size == 5
    assert res.final_status == "SUCCESS"
    assert len(res.tiers_executed) == 3
    assert all(t.passed for t in res.tiers_executed)
    assert len(res.rolled_back_nodes) == 0


@pytest.mark.asyncio
async def test_fleet_error_threshold_freeze_and_atomic_rollback() -> None:
    """Test halting rollout and executing atomic rollback when error rate exceeds threshold."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    orchestrator = FleetRolloutOrchestrator(
        config=cfg,
        sandbox_factory=mock_factory,
        error_threshold=0.02,  # Strict threshold
        tiers=[0.20, 0.60, 1.00],
    )
    scenario = SystemdDNSScenario()

    # Apply broken patch that fails verifier
    res = await orchestrator.execute_fleet_rollout(
        scenario=scenario,
        fleet_size=5,
        patch_command="echo 'nameserver 127.0.0.99' > /etc/resolv.conf",
    )

    assert res.final_status == "FROZEN_ROLLED_BACK"
    assert len(res.tiers_executed) == 1
    assert res.tiers_executed[0].passed is False
    assert res.tiers_executed[0].error_rate > 0.02
    assert len(res.rolled_back_nodes) > 0
