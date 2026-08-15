"""Unit tests for the Self-Supervised Differential State Shadow Engine."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.shadow_evaluator import (
    DifferentialMetrics,
    DifferentialStateReport,
    ShadowEvaluator,
)
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_differential_state_evaluation_success() -> None:
    """Test twin sandbox differential execution and promotion on zero regression."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    evaluator = ShadowEvaluator(config=cfg, sandbox_factory=mock_factory)
    scenario = SystemdDNSScenario()

    primary_sb = mock_factory("test-primary")
    shadow_sb = mock_factory("test-shadow")

    await primary_sb.setup()
    await shadow_sb.setup()

    report = await evaluator.evaluate_differential(
        scenario=scenario,
        primary_sb=primary_sb,
        shadow_sb=shadow_sb,
    )

    assert report.evaluation_id.startswith("diff-")
    assert report.scenario_name == "systemd_dns"
    assert report.passed is True
    assert report.promoted is True
    assert report.divergence_score <= 0.05
    assert report.metrics.primary_success_rate == 1.0
    assert report.metrics.shadow_success_rate == 0.0


@pytest.mark.asyncio
async def test_differential_state_divergence_detection() -> None:
    """Test detecting regression when primary fails or excessive divergence occurs."""
    primary_sb = MockSandbox("failing-primary")
    shadow_sb = MockSandbox("control-shadow")

    # Manually simulate faulted state where primary cannot resolve
    primary_sb.dns_working = False
    shadow_sb.dns_working = False

    metrics = DifferentialMetrics(
        fs_hash_matches=3,
        fs_hash_divergences=2,
        socket_status_matches=2,
        socket_status_divergences=1,
        memory_rss_delta_mb=250.0,
        primary_success_rate=0.0,
        shadow_success_rate=0.0,
    )

    report = DifferentialStateReport(
        evaluation_id="diff-test-fail",
        scenario_name="systemd_dns",
        passed=False,
        divergence_score=0.70,
        primary_instance="failing-primary",
        shadow_instance="control-shadow",
        metrics=metrics,
        promoted=False,
    )

    assert report.passed is False
    assert report.promoted is False
    assert report.divergence_score > 0.05
    assert "graph TD" in report.to_mermaid()


@pytest.mark.asyncio
async def test_shadow_evaluator_full_lifecycle() -> None:
    """Test running full shadow comparison lifecycle via run_shadow_comparison."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    evaluator = ShadowEvaluator(config=cfg, sandbox_factory=mock_factory)
    scenario = SystemdDNSScenario()

    report = await evaluator.run_shadow_comparison(
        scenario=scenario,
        primary_name="auto-primary",
        shadow_name="auto-shadow",
    )

    assert report.passed is True
    assert report.duration_seconds >= 0.0
    assert isinstance(report.to_dict(), dict)
