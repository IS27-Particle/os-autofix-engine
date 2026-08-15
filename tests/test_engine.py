"""Unit tests for engine components: action parsing, client retry loops, trajectory exporters, and orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import EngineConfig
from engine.action_schema import extract_json_block, parse_action_response
from engine.orchestrator import Orchestrator
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryBuffer, TrajectoryStep


def test_agent_action_parsing_valid() -> None:
    """Test standard valid JSON payload parsing."""
    raw_json = '{"thought": "Check system status", "command": "systemctl status", "timeout_seconds": 10, "is_done": false}'
    action = parse_action_response(raw_json)
    assert action.thought == "Check system status"
    assert action.command == "systemctl status"
    assert action.timeout_seconds == 10
    assert action.is_done is False


def test_agent_action_parsing_markdown_fences() -> None:
    """Test extracting JSON from markdown code fences."""
    raw = """Here is the command to run:
```json
{
  "thought": "Restart DNS",
  "command": "systemctl restart systemd-resolved",
  "timeout_seconds": 15,
  "is_done": true
}
```
Hope this helps!"""
    action = parse_action_response(raw)
    assert action.thought == "Restart DNS"
    assert action.command == "systemctl restart systemd-resolved"
    assert action.is_done is True


def test_agent_action_parsing_invalid() -> None:
    """Test syntax and schema validation errors."""
    with pytest.raises(ValueError):
        parse_action_response("Not a JSON object at all")

    with pytest.raises(ValidationError):
        parse_action_response('{"command": "ls -la"}')


def test_extract_json_block_helper() -> None:
    """Test extracting JSON substrings from mixed output."""
    raw = 'Some text before {"thought": "test", "command": ""} some text after'
    extracted = extract_json_block(raw)
    assert extracted == '{"thought": "test", "command": ""}'


@pytest.mark.asyncio
async def test_trajectory_buffer_exporters(tmp_path: Path) -> None:
    """Test exporting datasets to TRL GRPO, TRL DPO, Unsloth ShareGPT, and raw JSONL."""
    buffer = TrajectoryBuffer()

    step1 = TrajectoryStep(
        step_index=1,
        state_observation="DNS resolution failed",
        thought="Restart resolved",
        command="systemctl restart systemd-resolved",
        timeout_seconds=15,
        stdout="Restarted",
        stderr="",
        exit_code=0,
        reward=0.95,
        done=True,
    )

    traj_success = EpisodeTrajectory(
        scenario_name="systemd_dns",
        instance_id="inst-1",
        steps=[step1],
        success=True,
        total_reward=0.95,
        duration_seconds=3.2,
    )

    traj_fail = EpisodeTrajectory(
        scenario_name="systemd_dns",
        instance_id="inst-2",
        steps=[step1],
        success=False,
        total_reward=-0.5,
        duration_seconds=5.0,
    )

    await buffer.add_trajectory(traj_success)
    await buffer.add_trajectory(traj_fail)

    assert buffer.size == 2
    assert len(buffer.get_successful()) == 1
    assert len(buffer.get_failed()) == 1

    # 1. Raw JSONL
    raw_path = tmp_path / "raw.jsonl"
    count_raw = buffer.export_raw_jsonl(raw_path)
    assert count_raw == 2
    assert raw_path.exists()

    # 2. TRL GRPO
    grpo_path = tmp_path / "grpo.jsonl"
    count_grpo = buffer.export_trl_grpo(grpo_path)
    assert count_grpo == 2
    with grpo_path.open() as f:
        line = json.loads(f.readline())
        assert "prompt" in line
        assert "completion" in line
        assert "reward" in line

    # 3. TRL DPO
    dpo_path = tmp_path / "dpo.jsonl"
    count_dpo = buffer.export_trl_dpo(dpo_path)
    assert count_dpo >= 1
    with dpo_path.open() as f:
        line = json.loads(f.readline())
        assert "chosen" in line
        assert "rejected" in line

    # 4. Unsloth ShareGPT
    unsloth_path = tmp_path / "unsloth.jsonl"
    count_unsloth = buffer.export_unsloth_sharegpt(unsloth_path, successful_only=True)
    assert count_unsloth == 1
    with unsloth_path.open() as f:
        line = json.loads(f.readline())
        assert "conversations" in line
        assert line["conversations"][0]["from"] == "human"


@pytest.mark.asyncio
async def test_orchestrator_single_episode_workflow(engine_config: EngineConfig) -> None:
    """Test full orchestrator single episode lifecycle with mock sandbox and mock LLM."""
    mock_sb_instance = MockSandbox("orchestrator-testbox")
    buffer = TrajectoryBuffer()

    orchestrator = Orchestrator(
        config=engine_config,
        trajectory_buffer=buffer,
        custom_sandbox_factory=lambda name: mock_sb_instance,
    )

    scenario = SystemdDNSScenario()
    traj = await orchestrator.run_single_episode(scenario, episode_idx=1)

    assert traj.scenario_name == "systemd_dns"
    assert traj.success is True
    assert traj.total_reward > 0
    assert len(traj.steps) >= 1
    assert buffer.size == 1
    assert mock_sb_instance.is_cleaned is True
