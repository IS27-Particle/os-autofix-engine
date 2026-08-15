"""Trainer package for recording, formatting, and executing SFT and GRPO policy training."""

from trainer.train_grpo import compute_trajectory_reward, load_grpo_dataset, train_grpo
from trainer.train_sft import load_sharegpt_dataset, train_sft
from trainer.trajectory_buffer import (
    EpisodeTrajectory,
    TrajectoryBuffer,
    TrajectoryStep,
)

__all__ = [
    "EpisodeTrajectory",
    "TrajectoryBuffer",
    "TrajectoryStep",
    "train_sft",
    "train_grpo",
    "compute_trajectory_reward",
    "load_sharegpt_dataset",
    "load_grpo_dataset",
]
