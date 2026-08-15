"""Unit tests for the Autonomous Chaos Engineering Daemon and MTTR tracking."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.chaos_daemon import ChaosDaemon
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_chaos_daemon_single_experiment() -> None:
    """Test running a single autonomous chaos experiment with canary sandbox."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    daemon = ChaosDaemon(
        config=cfg,
        fleet_size=2,
        rate_minutes=0.1,
        duration_hours=0.01,
        sandbox_factory=mock_factory,
    )

    scenario = SystemdDNSScenario()
    res = await daemon.run_single_experiment(scenario=scenario)

    assert res.experiment_id.startswith("chaos-")
    assert res.scenario_name == "systemd_dns"
    assert res.injected is True
    assert res.recovered is True
    assert res.mttr_seconds > 0.0
    assert res.safety_score == 1.0


def test_chaos_daemon_metrics_summary() -> None:
    """Test calculation of aggregate MTTR and recovery rate metrics."""
    cfg = EngineConfig()
    daemon = ChaosDaemon(config=cfg)

    # Empty history
    assert daemon.get_summary_metrics()["total_experiments"] == 0

    # Populated history
    from engine.chaos_daemon import ChaosExperimentResult

    daemon.history.append(
        ChaosExperimentResult(
            experiment_id="exp-1",
            scenario_name="dns",
            instance_id="canary-1",
            injected=True,
            recovered=True,
            mttr_seconds=12.5,
            safety_score=1.0,
        )
    )
    daemon.history.append(
        ChaosExperimentResult(
            experiment_id="exp-2",
            scenario_name="routing",
            instance_id="canary-2",
            injected=True,
            recovered=False,
            mttr_seconds=30.0,
            safety_score=0.9,
        )
    )

    summary = daemon.get_summary_metrics()
    assert summary["total_experiments"] == 2
    assert summary["recoveries"] == 1
    assert summary["recovery_rate"] == 0.5
    assert summary["mean_mttr_seconds"] == 12.5
    assert summary["avg_safety_score"] == 0.95
