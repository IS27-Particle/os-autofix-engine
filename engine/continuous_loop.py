"""Autonomous continuous closed-loop policy self-improvement orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import EngineConfig
from engine.deployer import OllamaDeployer
from engine.orchestrator import Orchestrator
from scenarios.registry import get_all_scenarios
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryBuffer

logger = logging.getLogger("os_autofix.engine.loop")
console = Console()


@dataclass
class LoopIterationResult:
    """Outcome metrics for a single self-improvement loop iteration."""

    iteration: int
    model_tag: str
    baseline_pass_rate: float
    evaluated_pass_rate: float
    delta_pass_rate: float
    total_episodes: int
    successful_episodes: int
    avg_reward: float
    training_type: str
    status: Literal["promoted", "rolled_back", "failed"]
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContinuousFeedbackLoop:
    """Autonomous engine orchestrating iterative Benchmark -> Collect -> Train -> Deploy -> Verify cycles."""

    def __init__(
        self,
        config: EngineConfig,
        deployer: OllamaDeployer | None = None,
        alert_dispatcher: Any | None = None,
        base_model_tag: str = "qwen2.5-coder:7b",
        model_family_prefix: str = "os-fixer",
        min_pass_rate_threshold: float = 0.5,
    ) -> None:
        self.config = config
        self.deployer = deployer or OllamaDeployer(base_url=config.llm.ollama_base_url)
        if alert_dispatcher is not None:
            self.alert_dispatcher = alert_dispatcher
        else:
            from monitoring.alerts import WebhookAlertDispatcher

            self.alert_dispatcher = WebhookAlertDispatcher(webhook_url=config.webhook_alert_url)

        self.current_model = base_model_tag
        self.model_family_prefix = model_family_prefix
        self.min_pass_rate_threshold = min_pass_rate_threshold
        self.history: list[LoopIterationResult] = []

    async def run_loop(
        self,
        iterations: int = 3,
        samples_per_iter: int = 6,
        training_type: Literal["sft", "grpo"] = "sft",
        dry_run: bool = False,
    ) -> list[LoopIterationResult]:
        """Execute N closed-loop self-improvement cycles."""
        console.print(
            Panel.fit(
                f"[bold cyan]Starting Continuous Self-Improvement Loop[/bold cyan]\n"
                f"Iterations: {iterations} | Samples/Iter: {samples_per_iter} | "
                f"Training: {training_type.upper()} | Base Model: {self.current_model}"
            )
        )

        all_scenarios = get_all_scenarios()

        for iter_idx in range(1, iterations + 1):
            iter_start = time.monotonic()
            new_tag = f"{self.model_family_prefix}:v{iter_idx}"

            console.print(
                f"\n[bold yellow]══════════ Iteration {iter_idx}/{iterations} ({new_tag}) ══════════[/bold yellow]"
            )

            # ---------------------------------------------------------
            # Step 1: Benchmark baseline performance of current model
            # ---------------------------------------------------------
            console.print(f"[bold]Phase 1: Baseline Evaluation of '{self.current_model}'...[/bold]")
            self.config.llm.model_name = self.current_model
            bench_buffer = TrajectoryBuffer()
            bench_orch = Orchestrator(config=self.config, trajectory_buffer=bench_buffer)

            baseline_results = await bench_orch.run_benchmark(all_scenarios, iterations=1)
            baseline_pass_rate = self._calc_pass_rate(baseline_results)
            console.print(f"  • Baseline Pass Rate: [cyan]{baseline_pass_rate * 100:.1f}%[/cyan]")

            # ---------------------------------------------------------
            # Step 2: Collect exploration trajectories
            # ---------------------------------------------------------
            console.print(
                f"\n[bold]Phase 2: Collecting {samples_per_iter} exploration rollouts...[/bold]"
            )
            collect_buffer = TrajectoryBuffer()
            collect_orch = Orchestrator(config=self.config, trajectory_buffer=collect_buffer)

            await collect_orch.collect_dataset(all_scenarios, total_samples=samples_per_iter)
            successful_trajs = collect_buffer.get_successful()
            console.print(
                f"  • Collected {collect_buffer.size} episodes ({len(successful_trajs)} successful fixes)"
            )

            # ---------------------------------------------------------
            # Step 3: Filter & Export Datasets
            # ---------------------------------------------------------
            console.print("\n[bold]Phase 3: Exporting Trajectory Datasets...[/bold]")
            iter_data_dir = self.config.data_dir / f"iter_{iter_idx}"
            iter_data_dir.mkdir(parents=True, exist_ok=True)

            sft_file = iter_data_dir / "dataset_unsloth_sharegpt.jsonl"
            grpo_file = iter_data_dir / "dataset_trl_grpo.jsonl"

            collect_buffer.export_unsloth_sharegpt(sft_file, successful_only=True)
            collect_buffer.export_trl_grpo(grpo_file)

            # ---------------------------------------------------------
            # Step 4: Fine-Tuning Pipeline Execution
            # ---------------------------------------------------------
            console.print(
                f"\n[bold]Phase 4: Training Policy Adapter ({training_type.upper()})...[/bold]"
            )
            iter_out_dir = Path("outputs") / f"model_v{iter_idx}"
            iter_out_dir.mkdir(parents=True, exist_ok=True)

            train_meta: dict[str, Any] = {}
            if training_type == "sft":
                from trainer.train_sft import train_sft

                # If no successful samples in this round, use fallback all trajectories
                train_file = (
                    sft_file
                    if len(successful_trajs) > 0
                    else (iter_data_dir / "dataset_fallback.jsonl")
                )
                if not sft_file.exists() or len(successful_trajs) == 0:
                    collect_buffer.export_unsloth_sharegpt(train_file, successful_only=False)

                train_meta = train_sft(
                    dataset_path=train_file,
                    model_name=self.current_model,
                    output_dir=iter_out_dir,
                    epochs=2,
                    dry_run=dry_run,
                )
            else:
                from trainer.train_grpo import train_grpo

                train_meta = train_grpo(
                    dataset_path=grpo_file,
                    model_name=self.current_model,
                    output_dir=iter_out_dir,
                    epochs=1,
                    dry_run=dry_run,
                )

            # ---------------------------------------------------------
            # Step 5: Export GGUF & Deploy / Register with Ollama
            # ---------------------------------------------------------
            console.print(f"\n[bold]Phase 5: Deploying '{new_tag}' to Ollama...[/bold]")
            gguf_path = train_meta.get("gguf_path", self.current_model)

            # If deploying in dry-run or remote Ollama without shared storage, base on current model tag
            base_for_deploy = (
                self.current_model if (dry_run or not Path(gguf_path).is_file()) else str(gguf_path)
            )

            modelfile_path = iter_out_dir / "Modelfile"
            await self.deployer.deploy_model(
                model_name=new_tag,
                base_model_or_gguf=base_for_deploy,
                output_modelfile_path=modelfile_path,
            )

            # ---------------------------------------------------------
            # Step 6: Verify and Benchmark the new model generation
            # ---------------------------------------------------------
            console.print(f"\n[bold]Phase 6: Evaluating new generation '{new_tag}'...[/bold]")
            verify_cfg = self.config
            verify_cfg.llm.model_name = new_tag

            verify_buffer = TrajectoryBuffer()
            verify_orch = Orchestrator(config=verify_cfg, trajectory_buffer=verify_buffer)

            eval_results = await verify_orch.run_benchmark(all_scenarios, iterations=1)
            eval_pass_rate = self._calc_pass_rate(eval_results)
            delta = eval_pass_rate - baseline_pass_rate

            console.print(
                f"  • New Pass Rate: [bold]{eval_pass_rate * 100:.1f}%[/bold] "
                f"(Delta: {'+' if delta >= 0 else ''}{delta * 100:.1f}%)"
            )

            # ---------------------------------------------------------
            # Step 7: Auto-Rollback Decision Logic & Alerting
            # ---------------------------------------------------------
            status: Literal["promoted", "rolled_back", "failed"] = "promoted"
            if delta < -0.15 or eval_pass_rate < self.min_pass_rate_threshold:
                console.print(
                    f"[bold red]WARNING: Generation '{new_tag}' regressed below threshold! Rolling back to '{self.current_model}'...[/bold red]"
                )
                status = "rolled_back"
                try:
                    await self.alert_dispatcher.dispatch_model_regression(
                        model_tag=new_tag,
                        rolled_back_to=self.current_model,
                        eval_pass_rate=eval_pass_rate,
                        baseline_pass_rate=baseline_pass_rate,
                        delta=delta,
                        iteration=iter_idx,
                    )
                except Exception as e:
                    logger.warning("Failed dispatching model regression alert: %s", e)
            else:
                console.print(
                    f"[bold green]PROMOTING '{new_tag}' as the new active policy![/bold green]"
                )
                try:
                    await self.alert_dispatcher.dispatch_model_promoted(
                        model_tag=new_tag,
                        eval_pass_rate=eval_pass_rate,
                        baseline_pass_rate=baseline_pass_rate,
                        delta=delta,
                        iteration=iter_idx,
                    )
                except Exception as e:
                    logger.warning("Failed dispatching model promotion alert: %s", e)

                self.current_model = new_tag

            duration = time.monotonic() - iter_start
            iter_result = LoopIterationResult(
                iteration=iter_idx,
                model_tag=new_tag,
                baseline_pass_rate=baseline_pass_rate,
                evaluated_pass_rate=eval_pass_rate,
                delta_pass_rate=delta,
                total_episodes=len(eval_results),
                successful_episodes=len([r for r in eval_results if r.success]),
                avg_reward=sum([r.total_reward for r in eval_results]) / len(eval_results)
                if eval_results
                else 0.0,
                training_type=training_type,
                status=status,
                duration_seconds=duration,
                metadata=train_meta,
            )
            self.history.append(iter_result)

        self._display_loop_summary()
        return self.history

    def _calc_pass_rate(self, results: list[EpisodeTrajectory]) -> float:
        """Calculate percentage of successful episodes."""
        if not results:
            return 0.0
        success_count = sum(1 for r in results if r.success)
        return success_count / len(results)

    def _display_loop_summary(self) -> None:
        """Render final summary table across all self-improvement loop iterations."""
        table = Table(
            title="Continuous Self-Improvement Loop History",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Iter", justify="center")
        table.add_column("Model Tag", style="cyan")
        table.add_column("Base Rate", justify="right")
        table.add_column("Eval Rate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Avg Reward", justify="right")
        table.add_column("Decision", justify="center")
        table.add_column("Duration (s)", justify="right")

        for r in self.history:
            delta_str = (
                f"[green]+{r.delta_pass_rate * 100:.1f}%[/green]"
                if r.delta_pass_rate >= 0
                else f"[red]{r.delta_pass_rate * 100:.1f}%[/red]"
            )
            decision_str = (
                "[bold green]PROMOTED[/bold green]"
                if r.status == "promoted"
                else "[bold red]ROLLED BACK[/bold red]"
            )

            table.add_row(
                str(r.iteration),
                r.model_tag,
                f"{r.baseline_pass_rate * 100:.1f}%",
                f"{r.evaluated_pass_rate * 100:.1f}%",
                delta_str,
                f"{r.avg_reward:.2f}",
                decision_str,
                f"{r.duration_seconds:.1f}",
            )

        console.print(table)
        console.print(f"[bold]Active Stable Model:[/bold] [green]{self.current_model}[/green]")
