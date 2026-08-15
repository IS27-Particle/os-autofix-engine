"""Asynchronous worker pool orchestrator dispatching parallel Incus sandbox evaluation and training runs."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from rich.console import Console
from rich.table import Table

from config.settings import EngineConfig
from engine.client import LLMClientError, PolicyClient
from sandbox.base import BaseSandbox
from sandbox.incus_sandbox import IncusSandbox
from scenarios.base_scenario import BaseScenario
from trainer.trajectory_buffer import EpisodeTrajectory, TrajectoryBuffer, TrajectoryStep

logger = logging.getLogger("os_autofix.engine.orchestrator")
console = Console()


class Orchestrator:
    """Async worker pool manager executing OS-level policy benchmarking and dataset collection."""

    def __init__(
        self,
        config: EngineConfig,
        trajectory_buffer: TrajectoryBuffer | None = None,
        custom_sandbox_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.client = PolicyClient(config.llm)
        self.buffer = trajectory_buffer or TrajectoryBuffer()
        self.sandbox_factory = custom_sandbox_factory or self._default_sandbox_factory
        self._semaphore = asyncio.Semaphore(config.workers)

    def _default_sandbox_factory(self, instance_name: str) -> BaseSandbox:
        """Create a default IncusSandbox instance."""
        return IncusSandbox(
            instance_name=instance_name,
            config=self.config.incus,
        )

    async def run_single_episode(
        self,
        scenario: BaseScenario,
        episode_idx: int = 1,
    ) -> EpisodeTrajectory:
        """Execute a complete autonomous repair episode for a given scenario."""
        safe_scenario_name = scenario.name.replace("_", "-")
        instance_id = (
            f"{self.config.incus.instance_prefix}-{safe_scenario_name}-{uuid.uuid4().hex[:6]}"
        )
        logger.info(
            "Starting Episode #%d for scenario '%s' on instance '%s'",
            episode_idx,
            scenario.name,
            instance_id,
        )

        sandbox = self.sandbox_factory(instance_id)
        start_time = time.monotonic()
        steps: list[TrajectoryStep] = []
        total_reward = 0.0
        success = False
        verification_msg = ""

        try:
            from monitoring.dashboard import GLOBAL_DASHBOARD
            from monitoring.metrics import EPISODE_STEPS, SANDBOXES_ACTIVE, TASKS_TOTAL

            SANDBOXES_ACTIVE.inc()
            GLOBAL_DASHBOARD.update_worker(
                worker_id=episode_idx,
                scenario=scenario.name,
                instance_id=instance_id,
                step=0,
                max_steps=scenario.max_steps,
                thought="Initializing sandbox...",
                status="SETUP",
            )
        except Exception:
            pass

        try:
            # 1. Setup Sandbox VM/Container
            logger.info("[%s] Initializing sandbox environment...", instance_id)
            await sandbox.setup()

            # 2. Setup baseline scenario packages/services
            logger.info("[%s] Preparing scenario '%s'...", instance_id, scenario.name)
            await scenario.setup(sandbox)

            # 3. Create baseline snapshot for zero-copy isolation
            await sandbox.create_snapshot("snap-baseline")

            # 4. Inject fault
            logger.info("[%s] Injecting fault for scenario '%s'...", instance_id, scenario.name)
            await scenario.inject_fault(sandbox)

            # 5. Verify fault is active
            initial_verified, fault_msg = await scenario.verify(sandbox)
            if initial_verified:
                logger.warning(
                    "[%s] Scenario '%s' passed verification immediately after fault injection! Check fault injector.",
                    instance_id,
                    scenario.name,
                )

            # 6. Take snapshot of faulty state
            await sandbox.create_snapshot("snap-fault-injected")

            # 7. Initialize agent conversation
            initial_state_obs = f"System fault injected: {scenario.description}"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": scenario.get_prompt()},
                {"role": "user", "content": f"OBSERVATION:\n{initial_state_obs}"},
            ]

            # 8. Step Execution Loop
            max_steps = scenario.max_steps
            for step_idx in range(1, max_steps + 1):
                logger.info(
                    "[%s] Step %d/%d requesting agent action...", instance_id, step_idx, max_steps
                )

                try:
                    action, raw_completion = await self.client.get_next_action(messages)
                except LLMClientError as e:
                    logger.error("[%s] Policy client error: %s", instance_id, e)
                    step_record = TrajectoryStep(
                        step_index=step_idx,
                        state_observation=f"LLM Error: {e}",
                        thought="",
                        command="",
                        timeout_seconds=0,
                        stdout="",
                        stderr=str(e),
                        exit_code=1,
                        reward=-0.1,
                        done=True,
                    )
                    steps.append(step_record)
                    total_reward -= 0.1
                    break

                try:
                    from monitoring.dashboard import GLOBAL_DASHBOARD

                    GLOBAL_DASHBOARD.update_worker(
                        worker_id=episode_idx,
                        scenario=scenario.name,
                        instance_id=instance_id,
                        step=step_idx,
                        max_steps=max_steps,
                        thought=action.thought,
                        command=action.command,
                        status="RUNNING",
                    )
                except Exception:
                    pass

                exec_res = None
                if action.command.strip():
                    exec_res = await sandbox.execute(
                        action.command,
                        timeout_seconds=action.timeout_seconds
                        or self.config.incus.command_timeout_seconds,
                    )

                # Penalize each step slightly to encourage shortest path fixes
                step_reward = -self.config.step_penalty

                # Check if system is fixed
                verified, v_msg = await scenario.verify(sandbox)
                verification_msg = v_msg

                if verified:
                    success = True
                    step_reward += self.config.success_reward

                is_episode_done = verified or action.is_done or (step_idx == max_steps)

                step_record = TrajectoryStep(
                    step_index=step_idx,
                    state_observation=exec_res.combined_output
                    if exec_res
                    else "No command executed",
                    thought=action.thought,
                    command=action.command,
                    timeout_seconds=action.timeout_seconds,
                    stdout=exec_res.stdout if exec_res else "",
                    stderr=exec_res.stderr if exec_res else "",
                    exit_code=exec_res.exit_code if exec_res else 0,
                    reward=step_reward,
                    done=is_episode_done,
                    raw_model_completion=raw_completion,
                )
                steps.append(step_record)
                total_reward += step_reward

                if is_episode_done:
                    if verified:
                        logger.info(
                            "[%s] Scenario '%s' resolved successfully at step %d!",
                            instance_id,
                            scenario.name,
                            step_idx,
                        )
                    else:
                        logger.info(
                            "[%s] Episode ended without resolving scenario '%s' (step %d)",
                            instance_id,
                            scenario.name,
                            step_idx,
                        )
                    break

                # Update conversation messages for next step
                messages.append({"role": "assistant", "content": raw_completion})
                if exec_res:
                    obs_feedback = (
                        f"[EXIT CODE: {exec_res.exit_code}]\n"
                        f"STDOUT:\n{exec_res.stdout or '[No output]'}\n"
                    )
                    if exec_res.stderr:
                        obs_feedback += f"STDERR:\n{exec_res.stderr}\n"
                    if exec_res.timed_out:
                        obs_feedback += "[WARNING: Command timed out]\n"
                else:
                    obs_feedback = "[No command executed]"

                messages.append({"role": "user", "content": obs_feedback})

            # Final verification pass if not already marked success
            if not success:
                success, verification_msg = await scenario.verify(sandbox)

        finally:
            # 9. Clean up sandbox instance
            logger.info("[%s] Terminating and destroying sandbox...", instance_id)
            await sandbox.cleanup()
            try:
                from monitoring.dashboard import GLOBAL_DASHBOARD
                from monitoring.metrics import SANDBOXES_ACTIVE

                SANDBOXES_ACTIVE.dec()
                GLOBAL_DASHBOARD.update_worker(
                    worker_id=episode_idx,
                    scenario=scenario.name,
                    instance_id=instance_id,
                    status="SUCCESS" if success else "FAILED",
                )
            except Exception:
                pass

        duration = time.monotonic() - start_time
        trajectory = EpisodeTrajectory(
            scenario_name=scenario.name,
            instance_id=instance_id,
            steps=steps,
            success=success,
            total_reward=total_reward,
            duration_seconds=duration,
            verification_message=verification_msg,
        )

        try:
            from monitoring.dashboard import GLOBAL_DASHBOARD
            from monitoring.metrics import EPISODE_STEPS, TASKS_TOTAL

            status_str = "success" if success else "failure"
            TASKS_TOTAL.inc(
                scenario=scenario.name,
                model_tag=self.config.llm.model_name,
                status=status_str,
            )
            EPISODE_STEPS.observe(
                len(steps),
                scenario=scenario.name,
                model_tag=self.config.llm.model_name,
            )
            GLOBAL_DASHBOARD.record_episode_result(
                scenario=scenario.name,
                success=success,
                steps=len(steps),
                reward=total_reward,
                duration=duration,
            )
        except Exception:
            pass

        await self.buffer.add_trajectory(trajectory)
        return trajectory

    async def _worker_task(
        self,
        queue: asyncio.Queue[tuple[BaseScenario, int]],
        results: list[EpisodeTrajectory],
    ) -> None:
        """Worker loop processing episodes from the task queue."""
        while not queue.empty():
            try:
                scenario, ep_idx = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            async with self._semaphore:
                try:
                    traj = await self.run_single_episode(scenario, episode_idx=ep_idx)
                    results.append(traj)
                except Exception as e:
                    logger.error(
                        "Fatal error during episode execution for '%s': %s", scenario.name, e
                    )
                finally:
                    queue.task_done()

    async def run_benchmark(
        self,
        scenarios: list[BaseScenario],
        iterations: int = 1,
    ) -> list[EpisodeTrajectory]:
        """Run benchmark evaluation across specified scenarios with parallel workers."""
        queue: asyncio.Queue[tuple[BaseScenario, int]] = asyncio.Queue()
        episode_count = 1

        for _ in range(iterations):
            for sc in scenarios:
                queue.put_nowait((sc, episode_count))
                episode_count += 1

        total_tasks = queue.qsize()
        console.print(
            f"[bold green]Starting evaluation benchmark:[/bold green] {total_tasks} episodes across {len(scenarios)} scenarios with {self.config.workers} workers."
        )

        results: list[EpisodeTrajectory] = []
        worker_tasks = [
            asyncio.create_task(self._worker_task(queue, results))
            for _ in range(self.config.workers)
        ]

        await asyncio.gather(*worker_tasks)
        self._display_summary_table(results)

        if results:
            pass_rate = sum(1 for r in results if r.success) / len(results)
            try:
                from monitoring.metrics import MODEL_PASS_RATE

                MODEL_PASS_RATE.set(pass_rate, model_tag=self.config.llm.model_name)
            except Exception:
                pass

        return results

    async def collect_dataset(
        self,
        scenarios: list[BaseScenario],
        total_samples: int = 10,
    ) -> TrajectoryBuffer:
        """Run parallel data collection loops to populate trajectory buffer."""
        queue: asyncio.Queue[tuple[BaseScenario, int]] = asyncio.Queue()

        for i in range(total_samples):
            sc = scenarios[i % len(scenarios)]
            queue.put_nowait((sc, i + 1))

        console.print(
            f"[bold blue]Starting trajectory dataset collection:[/bold blue] {total_samples} samples across {len(scenarios)} scenarios with {self.config.workers} workers."
        )

        results: list[EpisodeTrajectory] = []
        worker_tasks = [
            asyncio.create_task(self._worker_task(queue, results))
            for _ in range(self.config.workers)
        ]

        await asyncio.gather(*worker_tasks)
        self._display_summary_table(results)
        return self.buffer

    def _display_summary_table(self, trajectories: list[EpisodeTrajectory]) -> None:
        """Render a formatted Rich table of benchmark / collection outcomes."""
        table = Table(
            title="OS-AutoFix Episode Evaluation Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Scenario", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Steps", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Verification Details", style="dim")

        success_count = 0
        total_duration = 0.0

        for t in trajectories:
            status_str = "[green]SUCCESS[/green]" if t.success else "[red]FAILED[/red]"
            if t.success:
                success_count += 1
            total_duration += t.duration_seconds

            table.add_row(
                t.scenario_name,
                status_str,
                str(len(t.steps)),
                f"{t.total_reward:.2f}",
                f"{t.duration_seconds:.1f}",
                t.verification_message[:40] if t.verification_message else "-",
            )

        console.print(table)
        total = len(trajectories)
        rate = (success_count / total * 100) if total > 0 else 0.0
        console.print(
            f"[bold]Total Episodes:[/bold] {total} | "
            f"[bold green]Success Rate:[/bold green] {rate:.1f}% ({success_count}/{total}) | "
            f"[bold]Total Time:[/bold] {total_duration:.1f}s"
        )
