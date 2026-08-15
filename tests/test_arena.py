"""Unit tests for Model Arena ELO rating tournament system."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import EngineConfig
from engine.arena import ModelArena
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryStep


def test_elo_update_formula() -> None:
    """Test standard ELO update calculation for wins, losses, and draws."""
    arena = ModelArena()

    # Case 1: Model A wins against equal rating
    new_a, new_b = arena.compute_elo_update(1200.0, 1200.0, 1.0, 0.0, k_factor=32.0)
    assert new_a == 1216.0
    assert new_b == 1184.0

    # Case 2: Draw between equal ratings
    draw_a, draw_b = arena.compute_elo_update(1200.0, 1200.0, 0.5, 0.5, k_factor=32.0)
    assert draw_a == 1200.0
    assert draw_b == 1200.0

    # Case 3: Underdog (1000) beats Favorite (1400)
    underdog_a, fav_b = arena.compute_elo_update(1000.0, 1400.0, 1.0, 0.0, k_factor=32.0)
    assert underdog_a > 1028.0  # Large gain
    assert fav_b < 1372.0


def test_determine_winner_criteria() -> None:
    """Test multi-tier victory conditions (success, step efficiency, latency)."""
    arena = ModelArena()

    # Tier 1: Success vs Failure
    traj_win = EpisodeTrajectory(
        scenario_name="dns",
        instance_id="sb-win",
        steps=[TrajectoryStep(1, "", "", "", 15, "", "", 0, 1.0, True, "")],
        success=True,
        total_reward=1.0,
        duration_seconds=5.0,
    )
    traj_loss = EpisodeTrajectory(
        scenario_name="dns",
        instance_id="sb-loss",
        steps=[],
        success=False,
        total_reward=0.0,
        duration_seconds=5.0,
    )
    score_a, score_b, winner, _ = arena.determine_winner(traj_win, traj_loss)
    assert score_a == 1.0
    assert score_b == 0.0

    # Tier 2: Step efficiency
    traj_fast_steps = EpisodeTrajectory(
        scenario_name="dns",
        instance_id="sb-fast",
        steps=[TrajectoryStep(1, "", "", "", 15, "", "", 0, 1.0, True, "")],
        success=True,
        total_reward=1.0,
        duration_seconds=5.0,
    )
    traj_slow_steps = EpisodeTrajectory(
        scenario_name="dns",
        instance_id="sb-slow",
        steps=[
            TrajectoryStep(1, "", "", "", 15, "", "", 0, 0.0, False, ""),
            TrajectoryStep(2, "", "", "", 15, "", "", 0, 1.0, True, ""),
        ],
        success=True,
        total_reward=1.0,
        duration_seconds=5.0,
    )
    score_a, score_b, winner, _ = arena.determine_winner(traj_fast_steps, traj_slow_steps)
    assert score_a == 1.0
    assert score_b == 0.0


@pytest.mark.asyncio
async def test_arena_tournament_execution(tmp_path: Path) -> None:
    """Test full arena tournament run with mock sandbox factory."""
    from scenarios.base_scenario import BaseScenario

    ratings_file = tmp_path / "arena_ratings.json"
    cfg = EngineConfig()
    cfg.llm.mock_mode = True
    arena = ModelArena(config=cfg, ratings_file=ratings_file)

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    scenarios: list[BaseScenario] = [SystemdDNSScenario()]
    summary = await arena.run_tournament(
        model_a="qwen-base",
        model_b="os-fixer-v1",
        scenarios=scenarios,
        rounds=1,
        sandbox_factory=mock_factory,
    )

    assert summary.total_matches == 1
    assert len(summary.matches) == 1
    assert ratings_file.exists()
    assert summary.final_elo_a > 0
    assert summary.final_elo_b > 0
