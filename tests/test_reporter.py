"""Unit tests for benchmark reporting and Markdown/JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from engine.reporter import BenchmarkReporter
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryStep


def test_benchmark_reporter_empty() -> None:
    """Test reporter statistics computation on empty trajectories."""
    reporter = BenchmarkReporter()
    summary = reporter.generate_summary_data([], model_name="qwen2.5-coder:7b", backend="ollama")
    assert summary["total_episodes"] == 0
    assert summary["pass_rate"] == 0.0

    md = reporter.generate_markdown(summary)
    assert "OS-AutoFix Benchmark Report" in md
    assert "0.0%" in md


def test_benchmark_reporter_with_trajectories(tmp_path: Path) -> None:
    """Test reporter aggregation, markdown formatting, and file writes."""
    reporter = BenchmarkReporter(output_dir=tmp_path / "reports")

    step_success = TrajectoryStep(
        step_index=1,
        state_observation="OK",
        thought="Fixing DNS",
        command="systemctl restart systemd-resolved",
        timeout_seconds=15,
        stdout="",
        stderr="",
        exit_code=0,
        reward=0.95,
        done=True,
    )
    step_fail = TrajectoryStep(
        step_index=1,
        state_observation="Fail",
        thought="Broken route",
        command="wrong command",
        timeout_seconds=15,
        stdout="",
        stderr="Error",
        exit_code=1,
        reward=-0.05,
        done=True,
    )

    t1 = EpisodeTrajectory(
        scenario_name="systemd_dns",
        instance_id="autofix-1",
        steps=[step_success],
        success=True,
        total_reward=0.95,
        duration_seconds=2.5,
        verification_message="Resolved",
    )
    t2 = EpisodeTrajectory(
        scenario_name="network_routing",
        instance_id="autofix-2",
        steps=[step_fail],
        success=False,
        total_reward=-0.05,
        duration_seconds=3.0,
        verification_message="Failed to reach gateway",
    )

    summary = reporter.generate_summary_data(
        [t1, t2], model_name="qwen2.5-coder:7b", backend="ollama"
    )
    assert summary["total_episodes"] == 2
    assert summary["successful_episodes"] == 1
    assert summary["pass_rate"] == 0.5
    assert "systemd_dns" in summary["scenarios"]
    assert "network_routing" in summary["scenarios"]

    md_file, json_file = reporter.write_reports(
        [t1, t2],
        model_name="qwen2.5-coder:7b",
        backend="ollama",
    )

    assert md_file.exists()
    assert json_file.exists()

    md_content = md_file.read_text(encoding="utf-8")
    assert "systemd_dns" in md_content
    assert "network_routing" in md_content
    assert "50.0%" in md_content
    assert "autofix-2" in md_content

    json_content = json.loads(json_file.read_text(encoding="utf-8"))
    assert json_content["pass_rate"] == 0.5
    assert len(json_content["episodes"]) == 2
