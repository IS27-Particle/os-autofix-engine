"""Benchmark analytics and automated report generator for Markdown and JSON summaries."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trainer.trajectory_buffer import EpisodeTrajectory


class BenchmarkReporter:
    """Consumes benchmark episode trajectories and exports structured Markdown and JSON summaries."""

    def __init__(self, output_dir: Path | str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary_data(
        self,
        trajectories: list[EpisodeTrajectory],
        model_name: str = "unknown",
        backend: str = "ollama",
    ) -> dict[str, Any]:
        """Compute aggregated statistics from evaluation trajectories."""
        total_episodes = len(trajectories)
        if total_episodes == 0:
            return {
                "timestamp": time.time(),
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "model_name": model_name,
                "backend": backend,
                "total_episodes": 0,
                "successful_episodes": 0,
                "pass_rate": 0.0,
                "avg_reward": 0.0,
                "avg_steps": 0.0,
                "avg_duration_seconds": 0.0,
                "scenarios": {},
            }

        successful = [t for t in trajectories if t.success]
        pass_rate = len(successful) / total_episodes
        avg_reward = sum(t.total_reward for t in trajectories) / total_episodes
        avg_steps = sum(len(t.steps) for t in trajectories) / total_episodes
        avg_duration = sum(t.duration_seconds for t in trajectories) / total_episodes

        # Group by scenario
        scenario_groups: dict[str, list[EpisodeTrajectory]] = {}
        for t in trajectories:
            scenario_groups.setdefault(t.scenario_name, []).append(t)

        scenario_breakdown: dict[str, Any] = {}
        for sc_name, sc_trajs in scenario_groups.items():
            sc_total = len(sc_trajs)
            sc_success = len([t for t in sc_trajs if t.success])
            sc_pass_rate = sc_success / sc_total if sc_total > 0 else 0.0
            sc_avg_steps = sum(len(t.steps) for t in sc_trajs) / sc_total if sc_total > 0 else 0.0
            sc_avg_reward = (
                sum(t.total_reward for t in sc_trajs) / sc_total if sc_total > 0 else 0.0
            )
            sc_avg_duration = (
                sum(t.duration_seconds for t in sc_trajs) / sc_total if sc_total > 0 else 0.0
            )

            failures = [
                {
                    "instance_id": t.instance_id,
                    "steps_taken": len(t.steps),
                    "message": t.verification_message,
                    "last_command": t.steps[-1].command if t.steps else "",
                }
                for t in sc_trajs
                if not t.success
            ]

            scenario_breakdown[sc_name] = {
                "total": sc_total,
                "successful": sc_success,
                "pass_rate": sc_pass_rate,
                "avg_steps": sc_avg_steps,
                "avg_reward": sc_avg_reward,
                "avg_duration_seconds": sc_avg_duration,
                "failures": failures,
            }

        return {
            "timestamp": time.time(),
            "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "model_name": model_name,
            "backend": backend,
            "total_episodes": total_episodes,
            "successful_episodes": len(successful),
            "pass_rate": pass_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_duration_seconds": avg_duration,
            "scenarios": scenario_breakdown,
            "episodes": [asdict(t) for t in trajectories],
        }

    def generate_markdown(self, summary_data: dict[str, Any]) -> str:
        """Generate GitHub Flavored Markdown evaluation report."""
        pass_rate_pct = summary_data["pass_rate"] * 100
        pass_rate_badge = "🟢" if pass_rate_pct >= 80 else ("🟡" if pass_rate_pct >= 50 else "🔴")

        md_lines = [
            f"# 📊 OS-AutoFix Benchmark Report: `{summary_data['model_name']}`",
            "",
            f"- **Execution Time:** {summary_data['formatted_time']}",
            f"- **Model Tag:** `{summary_data['model_name']}` ({summary_data['backend']})",
            f"- **Overall Pass Rate:** {pass_rate_badge} **{pass_rate_pct:.1f}%** ({summary_data['successful_episodes']}/{summary_data['total_episodes']} resolved)",
            f"- **Average Steps to Resolve:** `{summary_data['avg_steps']:.2f}`",
            f"- **Average Duration per Episode:** `{summary_data['avg_duration_seconds']:.2f}s`",
            f"- **Average Cumulative Reward:** `{summary_data['avg_reward']:.3f}`",
            "",
            "---",
            "",
            "## 📋 Scenario Breakdown",
            "",
            "| Scenario | Pass Rate | Solved | Avg Steps | Avg Reward | Avg Duration |",
            "|---|---|---|---|---|---|",
        ]

        for sc_name, sc_data in sorted(summary_data["scenarios"].items()):
            sc_pct = sc_data["pass_rate"] * 100
            sc_badge = "🟢" if sc_pct >= 80 else ("🟡" if sc_pct >= 50 else "🔴")
            md_lines.append(
                f"| `{sc_name}` | {sc_badge} {sc_pct:.1f}% | {sc_data['successful']}/{sc_data['total']} | "
                f"{sc_data['avg_steps']:.1f} | {sc_data['avg_reward']:.2f} | {sc_data['avg_duration_seconds']:.2f}s |"
            )

        md_lines.extend(["", "---", "", "## 🔍 Failure Diagnostics & Root Causes", ""])

        has_failures = False
        for sc_name, sc_data in sorted(summary_data["scenarios"].items()):
            failures = sc_data.get("failures", [])
            if failures:
                has_failures = True
                md_lines.append(f"### `{sc_name}` Failures ({len(failures)})")
                for idx, f in enumerate(failures, 1):
                    md_lines.extend(
                        [
                            f"**{idx}. Instance `{f['instance_id']}` (at step {f['steps_taken']}):**",
                            f"- Last Command: `{f['last_command'] or 'None'}`",
                            f"- Verification: `{f['message'] or 'Unresolved'}`",
                            "",
                        ]
                    )

        if not has_failures:
            md_lines.append("🎉 **No failures recorded across any diagnostic scenario!**")

        md_lines.extend(
            [
                "",
                "---",
                "*Report automatically generated by [OS-AutoFix Engine](https://github.com/Antigravity/os-autofix-engine)*",
            ]
        )

        return "\n".join(md_lines) + "\n"

    def write_reports(
        self,
        trajectories: list[EpisodeTrajectory],
        model_name: str = "unknown",
        backend: str = "ollama",
        md_filename: str = "benchmark_latest.md",
        json_filename: str = "benchmark_latest.json",
    ) -> tuple[Path, Path]:
        """Compute, format, and save both Markdown and JSON reports."""
        summary = self.generate_summary_data(trajectories, model_name=model_name, backend=backend)
        md_content = self.generate_markdown(summary)

        md_path = self.output_dir / md_filename
        json_path = self.output_dir / json_filename

        md_path.write_text(md_content, encoding="utf-8")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return md_path, json_path
