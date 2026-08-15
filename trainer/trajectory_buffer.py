"""Thread-safe and async-safe trajectory recording buffer with TRL and Unsloth dataset exporters."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("os_autofix.trainer.buffer")


@dataclass
class TrajectoryStep:
    """A single decision step within an environment episode."""

    step_index: int
    state_observation: str
    thought: str
    command: str
    timeout_seconds: int
    stdout: str
    stderr: str
    exit_code: int
    reward: float
    done: bool
    raw_model_completion: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrajectory:
    """A complete episode trajectory containing sequential agent steps and episode outcome."""

    scenario_name: str
    instance_id: str
    steps: list[TrajectoryStep]
    success: bool
    total_reward: float
    duration_seconds: float
    verification_message: str = ""
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode trajectory to a standard dictionary."""
        return asdict(self)


class TrajectoryBuffer:
    """Concurrent buffer storing agent trajectories for offline RL (GRPO/DPO) and SFT dataset generation."""

    def __init__(self) -> None:
        self._trajectories: list[EpisodeTrajectory] = []
        self._lock = asyncio.Lock()

    async def add_trajectory(self, trajectory: EpisodeTrajectory) -> None:
        """Append an episode trajectory safely in an async context."""
        async with self._lock:
            self._trajectories.append(trajectory)
            logger.info(
                "Recorded trajectory for scenario '%s' (success=%s, steps=%d, reward=%.2f)",
                trajectory.scenario_name,
                trajectory.success,
                len(trajectory.steps),
                trajectory.total_reward,
            )

    def add_trajectory_sync(self, trajectory: EpisodeTrajectory) -> None:
        """Synchronous trajectory addition for non-async callers."""
        self._trajectories.append(trajectory)

    def get_all(self) -> list[EpisodeTrajectory]:
        """Retrieve all recorded trajectories."""
        return list(self._trajectories)

    def get_successful(self) -> list[EpisodeTrajectory]:
        """Filter trajectories that successfully resolved the fault scenario."""
        return [t for t in self._trajectories if t.success]

    def get_failed(self) -> list[EpisodeTrajectory]:
        """Filter trajectories that failed to resolve the fault scenario."""
        return [t for t in self._trajectories if not t.success]

    def clear(self) -> None:
        """Clear all trajectories from buffer."""
        self._trajectories.clear()

    @property
    def size(self) -> int:
        """Return count of trajectories currently buffered."""
        return len(self._trajectories)

    def export_raw_jsonl(self, output_path: Path | str) -> int:
        """Export all trajectories in raw JSONL format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with path.open("w", encoding="utf-8") as f:
            for traj in self._trajectories:
                f.write(json.dumps(traj.to_dict()) + "\n")
                count += 1

        logger.info("Exported %d raw trajectories to %s", count, path)
        return count

    def export_trl_grpo(self, output_path: Path | str) -> int:
        """Export dataset for Hugging Face TRL GRPO (Group Relative Policy Optimization).

        Outputs prompt, single/multi-step completions, and verified scalar rewards.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with path.open("w", encoding="utf-8") as f:
            for traj in self._trajectories:
                if not traj.steps:
                    continue

                initial_prompt = (
                    f"Diagnose and resolve the following OS issue: {traj.scenario_name}"
                )

                completion_payload = {
                    "actions": [
                        {
                            "thought": s.thought,
                            "command": s.command,
                            "is_done": s.done,
                        }
                        for s in traj.steps
                    ]
                }

                record = {
                    "prompt": initial_prompt,
                    "completion": json.dumps(completion_payload),
                    "reward": traj.total_reward,
                    "success": traj.success,
                    "scenario": traj.scenario_name,
                    "steps_count": len(traj.steps),
                    "episode_id": traj.episode_id,
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info("Exported %d GRPO samples to %s", count, path)
        return count

    def export_trl_dpo(self, output_path: Path | str) -> int:
        """Export dataset for Hugging Face TRL Direct Preference Optimization (DPO).

        Pairs successful trajectories (chosen) against failed trajectories (rejected)
        for identical scenarios.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        successful_by_scenario: dict[str, list[EpisodeTrajectory]] = {}
        failed_by_scenario: dict[str, list[EpisodeTrajectory]] = {}

        for traj in self._trajectories:
            if traj.success:
                successful_by_scenario.setdefault(traj.scenario_name, []).append(traj)
            else:
                failed_by_scenario.setdefault(traj.scenario_name, []).append(traj)

        count = 0
        with path.open("w", encoding="utf-8") as f:
            for scenario, chosen_list in successful_by_scenario.items():
                rejected_list = failed_by_scenario.get(scenario, [])
                for chosen in chosen_list:
                    prompt = f"Diagnose and resolve the following OS issue: {scenario}"
                    chosen_formatted = [
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {"thought": s.thought, "command": s.command, "is_done": s.done}
                            ),
                        }
                        for s in chosen.steps
                    ]

                    if rejected_list:
                        for rejected in rejected_list:
                            rejected_formatted = [
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "thought": s.thought,
                                            "command": s.command,
                                            "is_done": s.done,
                                        }
                                    ),
                                }
                                for s in rejected.steps
                            ]
                            dpo_sample = {
                                "prompt": prompt,
                                "chosen": chosen_formatted,
                                "rejected": rejected_formatted,
                                "scenario": scenario,
                            }
                            f.write(json.dumps(dpo_sample) + "\n")
                            count += 1
                    else:
                        synthetic_rejected = [
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "thought": "I will reboot blindly.",
                                        "command": "reboot",
                                        "is_done": True,
                                    }
                                ),
                            }
                        ]
                        dpo_sample = {
                            "prompt": prompt,
                            "chosen": chosen_formatted,
                            "rejected": synthetic_rejected,
                            "scenario": scenario,
                        }
                        f.write(json.dumps(dpo_sample) + "\n")
                        count += 1

        logger.info("Exported %d DPO preference pairs to %s", count, path)
        return count

    def export_unsloth_sharegpt(self, output_path: Path | str, successful_only: bool = True) -> int:
        """Export dataset in ShareGPT format optimized for Unsloth multi-turn instruction fine-tuning."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        target_trajectories = self.get_successful() if successful_only else self._trajectories

        with path.open("w", encoding="utf-8") as f:
            for traj in target_trajectories:
                conversations: list[dict[str, str]] = []

                conversations.append(
                    {
                        "from": "human",
                        "value": f"Troubleshoot and resolve this OS issue:\nScenario: {traj.scenario_name}",
                    }
                )

                for step in traj.steps:
                    action_json = json.dumps(
                        {
                            "thought": step.thought,
                            "command": step.command,
                            "timeout_seconds": step.timeout_seconds,
                            "is_done": step.done,
                        },
                        indent=2,
                    )
                    conversations.append(
                        {
                            "from": "gpt",
                            "value": action_json,
                        }
                    )

                    if not step.done:
                        obs = f"[EXIT CODE: {step.exit_code}]\nSTDOUT:\n{step.stdout}"
                        if step.stderr:
                            obs += f"\nSTDERR:\n{step.stderr}"
                        conversations.append(
                            {
                                "from": "human",
                                "value": obs,
                            }
                        )

                sharegpt_record = {
                    "id": traj.episode_id,
                    "conversations": conversations,
                    "metadata": {
                        "scenario": traj.scenario_name,
                        "success": traj.success,
                        "total_reward": traj.total_reward,
                    },
                }
                f.write(json.dumps(sharegpt_record) + "\n")
                count += 1

        logger.info("Exported %d Unsloth/ShareGPT conversations to %s", count, path)
        return count
