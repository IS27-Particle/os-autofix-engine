"""Live Terminal UI (TUI) Dashboard for real-time monitoring of sandboxes, agent thoughts, commands, and telemetry."""

from __future__ import annotations

import collections
import dataclasses
import threading
import time

from rich.align import Align
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


@dataclasses.dataclass
class WorkerState:
    """Live state of an individual evaluation worker."""

    worker_id: int
    scenario: str = "idle"
    instance_id: str = "-"
    step: int = 0
    max_steps: int = 0
    thought: str = "Waiting for assignment..."
    command: str = "-"
    status: str = "IDLE"
    updated_at: float = dataclasses.field(default_factory=time.time)


class DashboardManager:
    """Thread-safe dashboard manager coordinating live Rich terminal UI rendering."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        endpoint: str = "http://10.0.0.25:11434",
        worker_count: int = 4,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.worker_count = worker_count
        self.start_time = time.time()
        self._lock = threading.Lock()

        # Worker state map
        self.workers: dict[int, WorkerState] = {
            i + 1: WorkerState(worker_id=i + 1) for i in range(worker_count)
        }

        # Aggregated telemetry
        self.total_episodes = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        self.total_rewards: list[float] = []
        self.events: collections.deque[str] = collections.deque(maxlen=8)

    def update_worker(
        self,
        worker_id: int,
        scenario: str | None = None,
        instance_id: str | None = None,
        step: int | None = None,
        max_steps: int | None = None,
        thought: str | None = None,
        command: str | None = None,
        status: str | None = None,
    ) -> None:
        """Update live worker state in thread-safe manner."""
        with self._lock:
            if worker_id not in self.workers:
                self.workers[worker_id] = WorkerState(worker_id=worker_id)
            w = self.workers[worker_id]
            if scenario is not None:
                w.scenario = scenario
            if instance_id is not None:
                w.instance_id = instance_id
            if step is not None:
                w.step = step
            if max_steps is not None:
                w.max_steps = max_steps
            if thought is not None:
                w.thought = thought
            if command is not None:
                w.command = command
            if status is not None:
                w.status = status
            w.updated_at = time.time()

    def record_episode_result(
        self,
        scenario: str,
        success: bool,
        steps: int,
        reward: float,
        duration: float,
    ) -> None:
        """Record episode outcome into telemetry counters."""
        with self._lock:
            self.total_episodes += 1
            if success:
                self.successful_episodes += 1
            else:
                self.failed_episodes += 1
            self.total_rewards.append(reward)

            status_tag = "[green]SUCCESS[/green]" if success else "[red]FAIL[/red]"
            msg = f"[{time.strftime('%X')}] {scenario} -> {status_tag} in {steps} steps ({duration:.1f}s, R={reward:.2f})"
            self.events.append(msg)

    def add_event(self, event_text: str) -> None:
        """Add arbitrary informational event log to live feed."""
        with self._lock:
            self.events.append(f"[{time.strftime('%X')}] {event_text}")

    def render(self) -> RenderableType:
        """Construct full screen Rich Layout renderable."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="workers", size=9),
            Layout(name="lower"),
        )
        layout["lower"].split_row(
            Layout(name="telemetry", ratio=1),
            Layout(name="events", ratio=1),
        )

        with self._lock:
            # 1. Header
            uptime = int(time.time() - self.start_time)
            m, s = divmod(uptime, 60)
            h, m = divmod(m, 60)
            header_text = Text.assemble(
                ("OS-AutoFix Telemetry Dashboard", "bold cyan"),
                ("  |  Model: ", "dim"),
                (self.model_name, "bold green"),
                ("  |  Endpoint: ", "dim"),
                (self.endpoint, "yellow"),
                ("  |  Uptime: ", "dim"),
                (f"{h:02d}:{m:02d}:{s:02d}", "bold white"),
            )
            layout["header"].update(Panel(Align.center(header_text), style="bold blue"))

            # 2. Worker Status Table
            w_table = Table(
                expand=True,
                title="Active Sandbox Workers",
                header_style="bold magenta",
                show_lines=True,
            )
            w_table.add_column("Worker", justify="center", width=8)
            w_table.add_column("Scenario", style="cyan", width=18)
            w_table.add_column("Instance ID", style="dim", width=22)
            w_table.add_column("Progress", justify="center", width=10)
            w_table.add_column("Status", justify="center", width=12)
            w_table.add_column("Active Command", style="yellow")

            for wid in sorted(self.workers):
                w = self.workers[wid]
                step_str = f"{w.step}/{w.max_steps}" if w.max_steps > 0 else "-"
                status_color = (
                    "green"
                    if w.status in ("SUCCESS", "READY")
                    else "yellow"
                    if w.status in ("RUNNING", "INJECTING")
                    else "red"
                    if w.status == "FAILED"
                    else "dim"
                )
                w_table.add_row(
                    f"#{w.worker_id}",
                    w.scenario,
                    w.instance_id,
                    step_str,
                    f"[{status_color}]{w.status}[/{status_color}]",
                    w.command[:60] if w.command else "-",
                )
            layout["workers"].update(w_table)

            # 3. Telemetry Panel
            pass_rate = (
                (self.successful_episodes / self.total_episodes * 100)
                if self.total_episodes > 0
                else 0.0
            )
            avg_reward = (
                sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0.0
            )
            active_count = sum(
                1 for w in self.workers.values() if w.status in ("RUNNING", "SETUP", "INJECTING")
            )

            t_table = Table(show_header=False, expand=True, box=None)
            t_table.add_column("Metric", style="bold")
            t_table.add_column("Value", justify="right")

            t_table.add_row(
                "Active Sandboxes:", f"[cyan]{active_count} / {self.worker_count}[/cyan]"
            )
            t_table.add_row("Total Episodes:", str(self.total_episodes))
            t_table.add_row("Successful:", f"[green]{self.successful_episodes}[/green]")
            t_table.add_row("Failed:", f"[red]{self.failed_episodes}[/red]")
            t_table.add_row("Pass Rate:", f"[bold cyan]{pass_rate:.1f}%[/bold cyan]")
            t_table.add_row("Average Reward:", f"{avg_reward:.2f}")

            layout["telemetry"].update(
                Panel(t_table, title="Rolling Performance Metrics", border_style="green")
            )

            # 4. Live Events Feed
            event_lines = (
                list(self.events) if self.events else ["[dim]No events recorded yet.[/dim]"]
            )
            layout["events"].update(
                Panel(
                    "\n".join(event_lines),
                    title="Live Diagnostic & Action Feed",
                    border_style="yellow",
                )
            )

        return layout

    def run_live(self, duration_seconds: float | None = None, refresh_rate: float = 2.0) -> None:
        """Run interactive Live terminal dashboard loop."""
        end_time = time.time() + duration_seconds if duration_seconds else None
        with Live(self.render(), console=console, refresh_per_second=int(1 / refresh_rate)) as live:
            try:
                while True:
                    if end_time and time.time() >= end_time:
                        break
                    live.update(self.render())
                    time.sleep(refresh_rate)
            except KeyboardInterrupt:
                pass


# Global singleton dashboard manager
GLOBAL_DASHBOARD = DashboardManager()
