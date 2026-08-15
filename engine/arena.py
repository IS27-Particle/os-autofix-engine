"""Model Arena and ELO Rating tournament evaluation harness for comparing agent checkpoints."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import EngineConfig, get_default_config
from engine.orchestrator import Orchestrator
from scenarios.base_scenario import BaseScenario
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryBuffer

logger = logging.getLogger("os_autofix.engine.arena")

DEFAULT_RATINGS_FILE = Path("reports/arena_ratings.json")
DEFAULT_BASE_ELO = 1200.0


@dataclass
class MatchResult:
    """Outcome of a single head-to-head match between Model A and Model B on a scenario."""

    scenario_name: str
    round_idx: int
    model_a: str
    model_b: str
    score_a: float  # 1.0 (win), 0.5 (draw), 0.0 (loss)
    score_b: float
    traj_a_success: bool
    traj_b_success: bool
    traj_a_steps: int
    traj_b_steps: int
    traj_a_duration: float
    traj_b_duration: float
    winner: str  # model_a, model_b, or "tie"
    reason: str


@dataclass
class ArenaSummary:
    """Summary metrics of an Arena evaluation tournament."""

    model_a: str
    model_b: str
    total_matches: int
    wins_a: int
    wins_b: int
    draws: int
    initial_elo_a: float
    initial_elo_b: float
    final_elo_a: float
    final_elo_b: float
    matches: list[MatchResult] = field(default_factory=list)


class ModelArena:
    """Autonomous Model Arena executing paired head-to-head evaluations with ELO rating tracking."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        k_factor: float = 32.0,
        ratings_file: Path | str = DEFAULT_RATINGS_FILE,
    ) -> None:
        self.config = config or get_default_config()
        self.k_factor = k_factor
        self.ratings_file = Path(ratings_file)
        self.ratings = self.load_ratings()

    def load_ratings(self) -> dict[str, float]:
        """Load persistent ELO ratings from JSON store or initialize default ratings."""
        if self.ratings_file.exists():
            try:
                data = json.loads(self.ratings_file.read_text(encoding="utf-8"))
                return {k: float(v) for k, v in data.items()}
            except Exception as e:
                logger.warning("Failed loading ratings from %s: %s", self.ratings_file, e)
        return {}

    def save_ratings(self) -> Path:
        """Save current ELO ratings dictionary to JSON store."""
        self.ratings_file.parent.mkdir(parents=True, exist_ok=True)
        self.ratings_file.write_text(json.dumps(self.ratings, indent=2), encoding="utf-8")
        return self.ratings_file

    def get_rating(self, model_name: str) -> float:
        """Retrieve current rating for model, defaulting to base ELO 1200."""
        return self.ratings.get(model_name, DEFAULT_BASE_ELO)

    @staticmethod
    def compute_elo_update(
        rating_a: float,
        rating_b: float,
        score_a: float,
        score_b: float,
        k_factor: float = 32.0,
    ) -> tuple[float, float]:
        """Calculate updated ELO scores following standard logistic rating formulas."""
        expected_a = 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
        expected_b = 1.0 / (1.0 + math.pow(10, (rating_a - rating_b) / 400.0))

        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * (score_b - expected_b)

        return round(new_rating_a, 2), round(new_rating_b, 2)

    def determine_winner(
        self,
        traj_a: EpisodeTrajectory,
        traj_b: EpisodeTrajectory,
    ) -> tuple[float, float, str, str]:
        """Determine match scores (score_a, score_b, winner, reason) based on multi-tier victory criteria."""
        # Tier 1: Verification Success
        if traj_a.success and not traj_b.success:
            return 1.0, 0.0, traj_a.instance_id, "Model A resolved fault; Model B failed."
        if traj_b.success and not traj_a.success:
            return 0.0, 1.0, traj_b.instance_id, "Model B resolved fault; Model A failed."
        if not traj_a.success and not traj_b.success:
            return 0.5, 0.5, "tie", "Both models failed to resolve the scenario."

        # Tier 2: Step Efficiency (if both succeeded)
        steps_a = len(traj_a.steps)
        steps_b = len(traj_b.steps)
        if steps_a < steps_b:
            return (
                1.0,
                0.0,
                traj_a.instance_id,
                f"Model A required fewer steps ({steps_a} vs {steps_b}).",
            )
        if steps_b < steps_a:
            return (
                0.0,
                1.0,
                traj_b.instance_id,
                f"Model B required fewer steps ({steps_b} vs {steps_a}).",
            )

        # Tier 3: Execution Latency (if steps equal)
        if traj_a.duration_seconds < traj_b.duration_seconds - 0.5:
            return (
                1.0,
                0.0,
                traj_a.instance_id,
                f"Model A faster execution ({traj_a.duration_seconds:.2f}s vs {traj_b.duration_seconds:.2f}s).",
            )
        if traj_b.duration_seconds < traj_a.duration_seconds - 0.5:
            return (
                0.0,
                1.0,
                traj_b.instance_id,
                f"Model B faster execution ({traj_b.duration_seconds:.2f}s vs {traj_a.duration_seconds:.2f}s).",
            )

        return (
            0.5,
            0.5,
            "tie",
            "Both models succeeded with identical steps and comparable duration.",
        )

    async def run_tournament(
        self,
        model_a: str,
        model_b: str,
        scenarios: list[BaseScenario] | tuple[BaseScenario, ...],
        rounds: int = 1,
        instance_type: str = "container",
        sandbox_factory: Any | None = None,
    ) -> ArenaSummary:
        """Run paired tournament matches and update persistent ELO ratings."""
        logger.info(
            "Starting Arena Tournament: '%s' vs '%s' (%d scenarios, %d rounds)...",
            model_a,
            model_b,
            len(scenarios),
            rounds,
        )

        init_elo_a = self.get_rating(model_a)
        init_elo_b = self.get_rating(model_b)
        curr_elo_a = init_elo_a
        curr_elo_b = init_elo_b

        matches: list[MatchResult] = []
        wins_a = 0
        wins_b = 0
        draws = 0

        for r in range(1, rounds + 1):
            for sc in scenarios:
                logger.info("Arena Round %d: Scenario '%s'", r, sc.name)

                # Execute Model A
                cfg_a = get_default_config()
                cfg_a.llm.model_name = model_a
                cfg_a.incus.instance_type = (
                    "container" if instance_type.lower() == "container" else "vm"
                )
                orch_a = Orchestrator(
                    cfg_a,
                    TrajectoryBuffer(),
                    custom_sandbox_factory=sandbox_factory,
                )
                traj_a = await orch_a.run_single_episode(sc)

                # Execute Model B
                cfg_b = get_default_config()
                cfg_b.llm.model_name = model_b
                cfg_b.incus.instance_type = (
                    "container" if instance_type.lower() == "container" else "vm"
                )
                orch_b = Orchestrator(
                    cfg_b,
                    TrajectoryBuffer(),
                    custom_sandbox_factory=sandbox_factory,
                )
                traj_b = await orch_b.run_single_episode(sc)

                # Score match
                score_a, score_b, win_id, reason = self.determine_winner(traj_a, traj_b)
                winner_name = model_a if score_a == 1.0 else (model_b if score_b == 1.0 else "tie")

                if score_a == 1.0:
                    wins_a += 1
                elif score_b == 1.0:
                    wins_b += 1
                else:
                    draws += 1

                # Update ELO
                curr_elo_a, curr_elo_b = self.compute_elo_update(
                    curr_elo_a, curr_elo_b, score_a, score_b, self.k_factor
                )

                matches.append(
                    MatchResult(
                        scenario_name=sc.name,
                        round_idx=r,
                        model_a=model_a,
                        model_b=model_b,
                        score_a=score_a,
                        score_b=score_b,
                        traj_a_success=traj_a.success,
                        traj_b_success=traj_b.success,
                        traj_a_steps=len(traj_a.steps),
                        traj_b_steps=len(traj_b.steps),
                        traj_a_duration=traj_a.duration_seconds,
                        traj_b_duration=traj_b.duration_seconds,
                        winner=winner_name,
                        reason=reason,
                    )
                )

        self.ratings[model_a] = curr_elo_a
        self.ratings[model_b] = curr_elo_b
        self.save_ratings()

        return ArenaSummary(
            model_a=model_a,
            model_b=model_b,
            total_matches=len(matches),
            wins_a=wins_a,
            wins_b=wins_b,
            draws=draws,
            initial_elo_a=init_elo_a,
            initial_elo_b=init_elo_b,
            final_elo_a=curr_elo_a,
            final_elo_b=curr_elo_b,
            matches=matches,
        )
