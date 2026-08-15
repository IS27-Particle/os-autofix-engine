"""Unit tests for SFT, GRPO training pipelines, and multi-component reward calculations."""

from __future__ import annotations

import json
from pathlib import Path

from engine.deployer import generate_modelfile_content
from trainer.train_grpo import compute_trajectory_reward, load_grpo_dataset, train_grpo
from trainer.train_sft import load_sharegpt_dataset, train_sft


def test_compute_trajectory_reward_components() -> None:
    """Test multi-component reward evaluation for success, step penalties, and JSON schema compliance."""
    # 1. Success with clean structured JSON
    valid_json_completion = json.dumps(
        {
            "thought": "Restarting DNS resolver",
            "command": "systemctl restart systemd-resolved",
            "timeout_seconds": 15,
            "is_done": True,
        }
    )
    reward_success = compute_trajectory_reward(
        completion_text=valid_json_completion,
        success=True,
        steps_count=1,
        step_penalty=0.05,
        schema_bonus=0.2,
        success_reward=1.0,
    )
    # Expected: 1.0 (success) - 0.05 (1 step) + 0.2 (schema) = 1.15
    assert reward_success == 1.15

    # 2. Failure with invalid JSON
    reward_fail_bad_json = compute_trajectory_reward(
        completion_text="I don't know what to do",
        success=False,
        steps_count=3,
        step_penalty=0.05,
        schema_bonus=0.2,
        success_reward=1.0,
    )
    # Expected: 0.0 (fail) - 0.15 (3 steps) - 0.1 (bad schema) = -0.25
    assert reward_fail_bad_json == -0.25

    # 3. Multi-step action payload parsing
    multi_step_json = json.dumps(
        {
            "actions": [
                {"thought": "Step 1", "command": "ip route", "is_done": False},
                {"thought": "Step 2", "command": "dhclient", "is_done": True},
            ]
        }
    )
    reward_multi = compute_trajectory_reward(
        completion_text=multi_step_json,
        success=True,
        steps_count=2,
    )
    # Expected: 1.0 - 0.10 + 0.2 = 1.10
    assert reward_multi == 1.10


def test_load_sharegpt_dataset_parsing(tmp_path: Path) -> None:
    """Test loading and validating ShareGPT/Unsloth conversation JSONL datasets."""
    dataset_file = tmp_path / "test_sharegpt.jsonl"
    samples = [
        {
            "id": "sample-1",
            "conversations": [
                {"from": "human", "value": "Fix DNS"},
                {
                    "from": "gpt",
                    "value": '{"thought": "Restart", "command": "systemctl restart systemd-resolved", "is_done": true}',
                },
            ],
        },
        {
            "prompt": "Fix Route",
            "completion": '{"thought": "Fix", "command": "ip route", "is_done": true}',
        },
    ]

    with dataset_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    loaded = load_sharegpt_dataset(dataset_file)
    assert len(loaded) == 2
    assert "conversations" in loaded[0]
    assert "conversations" in loaded[1]


def test_load_grpo_dataset_parsing(tmp_path: Path) -> None:
    """Test loading and dynamic reward assignment for GRPO rollout datasets."""
    dataset_file = tmp_path / "test_grpo.jsonl"
    samples = [
        {
            "prompt": "Fix DNS",
            "completion": '{"thought": "Restart", "command": "systemctl restart systemd-resolved", "is_done": true}',
            "success": True,
            "steps_count": 1,
        },
    ]

    with dataset_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    loaded = load_grpo_dataset(dataset_file)
    assert len(loaded) == 1
    assert "reward" in loaded[0]
    assert loaded[0]["reward"] > 0.5


def test_modelfile_generation() -> None:
    """Test dynamic Modelfile construction with system prompts and stop tokens."""
    modelfile = generate_modelfile_content(
        base_model_or_gguf="/models/qwen-custom.gguf",
        temperature=0.2,
        top_p=0.9,
    )
    assert "FROM /models/qwen-custom.gguf" in modelfile
    assert "PARAMETER temperature 0.2" in modelfile
    assert 'PARAMETER stop "<|im_end|>"' in modelfile
    assert "SYSTEM" in modelfile


def test_train_sft_dry_run(tmp_path: Path) -> None:
    """Test SFT training pipeline in dry-run mode."""
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Investigate issue"},
                    {
                        "from": "gpt",
                        "value": '{"thought": "Diagnose", "command": "uname -a", "is_done": false}',
                    },
                ]
            }
        )
        + "\n"
    )

    out_dir = tmp_path / "sft_out"
    meta = train_sft(
        dataset_path=dataset_file,
        model_name="qwen2.5-coder:7b",
        output_dir=out_dir,
        epochs=1,
        export_gguf=True,
        dry_run=True,
    )

    assert meta["status"] == "success"
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "training_meta.json").exists()
    assert "gguf_path" in meta


def test_train_grpo_dry_run(tmp_path: Path) -> None:
    """Test GRPO policy optimization pipeline in dry-run mode."""
    dataset_file = tmp_path / "dataset_grpo.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "prompt": "Fix broken networking",
                "completion": '{"thought": "Re-add route", "command": "ip route add default", "is_done": true}',
                "success": True,
                "steps_count": 1,
            }
        )
        + "\n"
    )

    out_dir = tmp_path / "grpo_out"
    meta = train_grpo(
        dataset_path=dataset_file,
        model_name="qwen2.5-coder:7b",
        output_dir=out_dir,
        epochs=1,
        dry_run=True,
    )

    assert meta["status"] == "success"
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "grpo_meta.json").exists()
    assert meta["average_reward"] > 0
