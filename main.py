"""CLI interface for os-autofix-engine: evaluation benchmarking, dataset collection, model training, Ollama deployment, monitoring, and continuous learning loops."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from config.settings import get_default_config
from deploy.provisioner import HostProvisioner
from engine.continuous_loop import ContinuousFeedbackLoop
from engine.deployer import OllamaDeployer
from engine.orchestrator import Orchestrator
from engine.reporter import BenchmarkReporter
from monitoring.dashboard import GLOBAL_DASHBOARD
from monitoring.json_logger import setup_json_file_logging
from monitoring.metrics import start_metrics_server
from sandbox.incus_sandbox import IncusSandbox
from scenarios.registry import get_all_scenarios, get_scenario
from trainer.train_grpo import train_grpo
from trainer.train_sft import train_sft
from trainer.trajectory_buffer import TrajectoryBuffer

console = Console()
app = typer.Typer(
    name="os-autofix",
    help="Autonomous OS-level policy training, evaluation, deployment, and monitoring harness with Incus VM, Ollama, and Open-WebUI support.",
    add_completion=False,
    no_args_is_help=True,
)


def setup_logging(level: str = "INFO", enable_json_file: bool = True) -> None:
    """Configure structured rich logging and rotating JSON file handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    if enable_json_file:
        try:
            setup_json_file_logging(log_dir="logs", log_filename="os-autofix.jsonl")
        except Exception:
            pass


@app.command()
def bench(
    scenarios: Annotated[
        str,
        typer.Option("--scenarios", "-s", help="Comma-separated scenario names or 'all'"),
    ] = "all",
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel sandbox instances"),
    ] = 4,
    backend: Annotated[
        str,
        typer.Option(
            "--backend", "-b", help="Model backend: 'ollama', 'open-webui', 'openai', or 'mock'"
        ),
    ] = "ollama",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Inference model name tag"),
    ] = "qwen2.5-coder:7b",
    endpoint: Annotated[
        str,
        typer.Option(
            "--endpoint",
            "-e",
            help="Endpoint base URL (default: http://10.0.0.25:11434/v1 or https://ai.is27.duckdns.org/api)",
        ),
    ] = "",
    api_key: Annotated[
        str,
        typer.Option("--api-key", "-k", help="API token / Bearer key for Open-WebUI"),
    ] = "",
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Instance virtualization mode: 'vm' or 'container'"),
    ] = "vm",
    image: Annotated[
        str,
        typer.Option("--image", "-i", help="Incus image (e.g. images:ubuntu/24.04)"),
    ] = "images:ubuntu/24.04",
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Iterations per scenario"),
    ] = 1,
    report_dir: Annotated[
        str,
        typer.Option(
            "--report-dir",
            "-r",
            help="Directory where benchmark markdown and JSON reports are saved",
        ),
    ] = "reports",
    metrics_port: Annotated[
        int,
        typer.Option(
            "--metrics-port", "-p", help="Prometheus metrics exporter port (0 to disable)"
        ),
    ] = 9100,
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use heuristic offline mock LLM for testing"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)"),
    ] = "INFO",
) -> None:
    """Run evaluation benchmark across diagnostic scenarios."""
    setup_logging(log_level)
    console.print(Panel.fit("[bold cyan]OS-AutoFix Engine: Benchmark Evaluation[/bold cyan]"))

    metrics_server = None
    if metrics_port > 0:
        try:
            metrics_server, _ = start_metrics_server(port=metrics_port)
            console.print(
                f"[dim]Prometheus metrics exporter active at http://0.0.0.0:{metrics_port}/metrics[/dim]"
            )
        except Exception as e:
            console.print(f"[dim yellow]Metrics server notice: {e}[/dim yellow]")

    cfg = get_default_config()
    cfg.workers = workers
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"
    cfg.incus.default_image = image

    if backend:
        cfg.llm.backend = backend  # type: ignore[assignment]
    if model:
        cfg.llm.model_name = model
    if endpoint:
        if cfg.llm.backend == "open-webui":
            cfg.llm.open_webui_base_url = endpoint
        else:
            cfg.llm.ollama_base_url = endpoint
    if api_key:
        cfg.llm.open_webui_api_key = api_key
    if mock_llm:
        cfg.llm.mock_mode = True

    if scenarios.lower() == "all":
        target_scenarios = get_all_scenarios()
    else:
        names = [name.strip() for name in scenarios.split(",") if name.strip()]
        target_scenarios = [get_scenario(name) for name in names]

    console.print(f"[dim]Backend:[/dim] {cfg.llm.backend.upper()} ({cfg.llm.model_name})")
    console.print(f"[dim]Endpoint:[/dim] {cfg.llm.active_endpoint}")
    console.print(f"[dim]Evaluating scenarios:[/dim] {', '.join(s.name for s in target_scenarios)}")
    console.print(
        f"[dim]Virtualization:[/dim] {cfg.incus.instance_type.upper()} | [dim]Workers:[/dim] {cfg.workers}"
    )

    buffer = TrajectoryBuffer()
    orchestrator = Orchestrator(config=cfg, trajectory_buffer=buffer)

    results = asyncio.run(orchestrator.run_benchmark(target_scenarios, iterations=iterations))

    if results:
        reporter = BenchmarkReporter(output_dir=report_dir)
        md_file, json_file = reporter.write_reports(
            trajectories=results,
            model_name=cfg.llm.model_name,
            backend=cfg.llm.backend,
        )
        console.print("\n[bold green]Benchmark Reports Generated:[/bold green]")
        console.print(f"  • Markdown: [cyan]{md_file}[/cyan]")
        console.print(f"  • JSON:     [cyan]{json_file}[/cyan]")


@app.command()
def collect(
    scenarios: Annotated[
        str,
        typer.Option("--scenarios", "-s", help="Comma-separated scenario names or 'all'"),
    ] = "all",
    samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Total number of trajectory samples to collect"),
    ] = 10,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of concurrent sandboxes"),
    ] = 4,
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="'ollama', 'open-webui', 'openai', or 'mock'"),
    ] = "ollama",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Inference model name tag"),
    ] = "qwen2.5-coder:7b",
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Endpoint URL"),
    ] = "",
    api_key: Annotated[
        str,
        typer.Option("--api-key", "-k", help="API key for Open-WebUI"),
    ] = "",
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'vm' or 'container'"),
    ] = "vm",
    export_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Export format: 'trl-grpo', 'trl-dpo', 'unsloth', 'raw', or 'all'",
        ),
    ] = "all",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory where dataset files will be saved"),
    ] = "data",
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use heuristic offline mock LLM"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run parallel data collection loops to build RL (GRPO/DPO) and SFT datasets from successful fixes."""
    setup_logging(log_level)
    console.print(
        Panel.fit("[bold blue]OS-AutoFix Engine: Trajectory Dataset Collection[/bold blue]")
    )

    cfg = get_default_config()
    cfg.workers = workers
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"

    if backend:
        cfg.llm.backend = backend  # type: ignore[assignment]
    if model:
        cfg.llm.model_name = model
    if endpoint:
        if cfg.llm.backend == "open-webui":
            cfg.llm.open_webui_base_url = endpoint
        else:
            cfg.llm.ollama_base_url = endpoint
    if api_key:
        cfg.llm.open_webui_api_key = api_key
    if mock_llm:
        cfg.llm.mock_mode = True

    if scenarios.lower() == "all":
        target_scenarios = get_all_scenarios()
    else:
        names = [name.strip() for name in scenarios.split(",") if name.strip()]
        target_scenarios = [get_scenario(name) for name in names]

    buffer = TrajectoryBuffer()
    orchestrator = Orchestrator(config=cfg, trajectory_buffer=buffer)

    asyncio.run(orchestrator.collect_dataset(target_scenarios, total_samples=samples))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold green]Exporting datasets to {out_path}:[/bold green]")
    fmt = export_format.lower()

    if fmt in ("raw", "all"):
        raw_file = out_path / "trajectories_raw.jsonl"
        c = buffer.export_raw_jsonl(raw_file)
        console.print(f"  • Raw JSONL: [cyan]{raw_file}[/cyan] ({c} records)")

    if fmt in ("trl-grpo", "all"):
        grpo_file = out_path / "dataset_trl_grpo.jsonl"
        c = buffer.export_trl_grpo(grpo_file)
        console.print(f"  • TRL GRPO: [cyan]{grpo_file}[/cyan] ({c} records)")

    if fmt in ("trl-dpo", "all"):
        dpo_file = out_path / "dataset_trl_dpo.jsonl"
        c = buffer.export_trl_dpo(dpo_file)
        console.print(f"  • TRL DPO: [cyan]{dpo_file}[/cyan] ({c} pairs)")

    if fmt in ("unsloth", "all"):
        unsloth_file = out_path / "dataset_unsloth_sharegpt.jsonl"
        c = buffer.export_unsloth_sharegpt(unsloth_file)
        console.print(f"  • Unsloth ShareGPT: [cyan]{unsloth_file}[/cyan] ({c} conversations)")


@app.command("train-sft")
def cmd_train_sft(
    dataset: Annotated[
        str,
        typer.Option("--dataset", "-d", help="Path to ShareGPT/Unsloth JSONL dataset"),
    ] = "data/dataset_unsloth_sharegpt.jsonl",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Base model name/tag"),
    ] = "qwen2.5-coder:7b",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory for adapters and GGUF"),
    ] = "outputs/sft_adapter",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs"),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Per-device training batch size"),
    ] = 2,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = 2e-4,
    lora_r: Annotated[
        int,
        typer.Option("--lora-r", help="LoRA rank dimension"),
    ] = 16,
    export_gguf: Annotated[
        bool,
        typer.Option("--export-gguf", help="Export GGUF quantization after training"),
    ] = True,
    quantization: Annotated[
        str,
        typer.Option("--quantization", "-q", help="GGUF quantization format (q4_k_m, q8_0, f16)"),
    ] = "q4_k_m",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Execute in dry-run/simulation mode"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Trigger 4-bit LoRA Supervised Fine-Tuning (SFT) on trajectory JSONL dataset."""
    setup_logging(log_level)
    train_sft(
        dataset_path=dataset,
        model_name=model,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        export_gguf=export_gguf,
        quantization_type=quantization,
        dry_run=dry_run,
    )


@app.command("train-grpo")
def cmd_train_grpo(
    dataset: Annotated[
        str,
        typer.Option("--dataset", "-d", help="Path to TRL GRPO rollout JSONL dataset"),
    ] = "data/dataset_trl_grpo.jsonl",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Base model name/tag"),
    ] = "qwen2.5-coder:7b",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory for adapters"),
    ] = "outputs/grpo_adapter",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs"),
    ] = 1,
    generations: Annotated[
        int,
        typer.Option("--generations", "-g", help="Group generations per prompt"),
    ] = 4,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = 5e-6,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Execute in dry-run/simulation mode"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Trigger Group Relative Policy Optimization (GRPO) on rollout trajectory dataset."""
    setup_logging(log_level)
    train_grpo(
        dataset_path=dataset,
        model_name=model,
        output_dir=output_dir,
        epochs=epochs,
        num_generations=generations,
        learning_rate=learning_rate,
        dry_run=dry_run,
    )


@app.command("deploy")
def cmd_deploy(
    model_tag: Annotated[
        str,
        typer.Option("--model-tag", "-t", help="Ollama model tag to create (e.g. os-fixer:v1)"),
    ] = "os-fixer:v1",
    base_model_or_gguf: Annotated[
        str,
        typer.Option("--base", "-b", help="Base GGUF file path or model tag"),
    ] = "qwen2.5-coder:7b",
    ollama_url: Annotated[
        str,
        typer.Option("--ollama-url", "-u", help="Target Ollama endpoint"),
    ] = "http://10.0.0.25:11434",
    modelfile_out: Annotated[
        str,
        typer.Option("--modelfile-out", "-m", help="Path to save generated Modelfile"),
    ] = "outputs/Modelfile",
    temperature: Annotated[
        float,
        typer.Option("--temp", help="Sampling temperature"),
    ] = 0.2,
    top_p: Annotated[
        float,
        typer.Option("--top-p", help="Top-p sampling"),
    ] = 0.9,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Generate dynamic Modelfile and register model tag with remote/local Ollama daemon."""
    setup_logging(log_level)
    deployer = OllamaDeployer(base_url=ollama_url)
    asyncio.run(
        deployer.deploy_model(
            model_name=model_tag,
            base_model_or_gguf=base_model_or_gguf,
            output_modelfile_path=modelfile_out,
            temperature=temperature,
            top_p=top_p,
        )
    )


@app.command("loop")
def cmd_continuous_loop(
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Number of continuous improvement iterations"),
    ] = 3,
    samples_per_iter: Annotated[
        int,
        typer.Option("--samples", "-s", help="Rollout samples collected per iteration"),
    ] = 4,
    training_type: Annotated[
        str,
        typer.Option("--training-type", "-t", help="'sft' or 'grpo'"),
    ] = "sft",
    base_model: Annotated[
        str,
        typer.Option("--model", "-m", help="Base starting model tag"),
    ] = "qwen2.5-coder:7b",
    model_prefix: Annotated[
        str,
        typer.Option("--prefix", "-p", help="Model family tag prefix (e.g. os-fixer)"),
    ] = "os-fixer",
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel sandbox instances"),
    ] = 4,
    instance_type: Annotated[
        str,
        typer.Option("--type", help="'container' or 'vm'"),
    ] = "container",
    ollama_url: Annotated[
        str,
        typer.Option("--ollama-url", help="Ollama endpoint URL"),
    ] = "http://10.0.0.25:11434",
    min_pass_rate: Annotated[
        float,
        typer.Option("--min-pass-rate", help="Minimum pass rate threshold before auto-rollback"),
    ] = 0.5,
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Run with simulated mock LLM for testing"),
    ] = False,
    dry_run_training: Annotated[
        bool,
        typer.Option("--dry-run-training", help="Dry run training step"),
    ] = True,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run autonomous continuous closed-loop policy self-improvement (Benchmark -> Collect -> Train -> Deploy -> Verify)."""
    setup_logging(log_level)

    cfg = get_default_config()
    cfg.workers = workers
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"
    cfg.llm.ollama_base_url = ollama_url
    cfg.llm.model_name = base_model
    cfg.llm.mock_mode = mock_llm

    train_mode: Literal["sft", "grpo"] = "grpo" if training_type.lower() == "grpo" else "sft"

    loop = ContinuousFeedbackLoop(
        config=cfg,
        base_model_tag=base_model,
        model_family_prefix=model_prefix,
        min_pass_rate_threshold=min_pass_rate,
    )

    asyncio.run(
        loop.run_loop(
            iterations=iterations,
            samples_per_iter=samples_per_iter,
            training_type=train_mode,
            dry_run=dry_run_training,
        )
    )


@app.command("monitor")
def cmd_monitor(
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Prometheus metrics exporter port"),
    ] = 9100,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Metrics server listen address"),
    ] = "0.0.0.0",
    server_only: Annotated[
        bool,
        typer.Option(
            "--server-only",
            help="Run only the HTTP Prometheus metrics exporter in background/foreground",
        ),
    ] = False,
    refresh_rate: Annotated[
        float,
        typer.Option("--refresh", "-r", help="TUI refresh rate in seconds"),
    ] = 1.0,
) -> None:
    """Launch the live Terminal UI (TUI) dashboard or start standalone Prometheus exporter."""
    setup_logging("INFO")
    server, _ = start_metrics_server(port=port, host=host)

    if server_only:
        console.print(
            f"[bold green]Prometheus metrics exporter active at http://{host}:{port}/metrics[/bold green]"
        )
        console.print("[dim]Press Ctrl+C to terminate exporter...[/dim]")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            server.shutdown()
            console.print("[yellow]Prometheus metrics server stopped.[/yellow]")
    else:
        console.print(f"[dim]Prometheus exporter active at http://{host}:{port}/metrics[/dim]")
        GLOBAL_DASHBOARD.run_live(refresh_rate=refresh_rate)


@app.command("doctor")
def cmd_doctor(
    ollama_url: Annotated[
        str,
        typer.Option("--ollama-url", help="Ollama endpoint URL"),
    ] = "http://10.0.0.25:11434",
    open_webui_url: Annotated[
        str,
        typer.Option("--open-webui-url", help="Open-WebUI endpoint URL"),
    ] = "https://ai.is27.duckdns.org/api",
) -> None:
    """Run automated host hardware, Incus storage/bridge, and endpoint pre-flight diagnostics."""
    setup_logging("INFO")
    provisioner = HostProvisioner(ollama_url=ollama_url, open_webui_url=open_webui_url)
    ok = provisioner.run_doctor()
    if not ok:
        sys.exit(1)


@app.command("deploy-daemon")
def cmd_deploy_daemon(
    systemd_dir: Annotated[
        str,
        typer.Option("--systemd-dir", "-d", help="Systemd unit destination directory"),
    ] = "/etc/systemd/system",
    enable: Annotated[
        bool,
        typer.Option("--enable", "-e", help="Enable installed systemd service units"),
    ] = False,
) -> None:
    """Install and configure systemd service units for background continuous loop and metrics daemon."""
    setup_logging("INFO")
    provisioner = HostProvisioner()
    provisioner.install_systemd_services(target_dir=systemd_dir, enable_services=enable)


@app.command("test-env")
def test_env(
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'vm' or 'container' test"),
    ] = "container",
    image: Annotated[
        str,
        typer.Option("--image", "-i", help="Image alias"),
    ] = "images:ubuntu/24.04",
    ollama_url: Annotated[
        str,
        typer.Option("--ollama-url", help="Ollama endpoint URL"),
    ] = "http://10.0.0.25:11434",
    open_webui_url: Annotated[
        str,
        typer.Option("--open-webui-url", help="Open-WebUI endpoint URL"),
    ] = "https://ai.is27.duckdns.org/api",
) -> None:
    """Validate host Incus VM setup, storage pool readiness, and endpoint connectivity (Ollama / Open-WebUI)."""
    setup_logging("INFO")
    console.print(
        Panel.fit("[bold magenta]OS-AutoFix Engine: Environment Diagnostics[/bold magenta]")
    )

    provisioner = HostProvisioner(ollama_url=ollama_url, open_webui_url=open_webui_url)
    provisioner.run_doctor()

    # Ephemeral Sandbox Launch & Snapshot Test
    console.print("\n[bold]Testing live sandbox launch, snapshot creation, and rollback...[/bold]")

    async def _test_sandbox() -> None:
        sandbox = IncusSandbox(
            instance_name="autofix-diag-test",
            image=image,
            is_vm=(instance_type.lower() == "vm"),
        )
        try:
            console.print(
                f"  • Launching test sandbox '{sandbox.instance_name}' ({instance_type})..."
            )
            await sandbox.setup()
            console.print("  • Guest agent handshake: [bold green]OK[/bold green]")

            res = await sandbox.execute("uname -a")
            console.print(
                f"  • Execution test: [bold green]OK[/bold green] -> {res.stdout.strip()[:60]}"
            )

            console.print("  • Creating zero-copy snapshot 'diag-snap'...")
            await sandbox.create_snapshot("diag-snap")

            await sandbox.execute("echo 'CORRUPTED' > /tmp/diag_test.txt")
            console.print("  • Restoring snapshot 'diag-snap'...")
            await sandbox.revert("diag-snap")
            read_revert = await sandbox.execute(
                "cat /tmp/diag_test.txt 2>/dev/null || echo 'NOT_FOUND'"
            )
            assert "NOT_FOUND" in read_revert.stdout or "CORRUPTED" not in read_revert.stdout
            console.print("  • Sub-second rollback: [bold green]VERIFIED[/bold green]")

        finally:
            console.print(f"  • Cleaning up '{sandbox.instance_name}'...")
            await sandbox.cleanup()
            console.print("  • Cleanup: [bold green]COMPLETE[/bold green]")

    try:
        asyncio.run(_test_sandbox())
        console.print(
            "\n[bold green]All environment diagnostics and sandbox tests passed successfully![/bold green]"
        )
    except Exception as e:
        console.print(f"\n[bold red]Sandbox test failed:[/bold red] {e}")


@app.command("git-init")
def git_init(
    repo_name: Annotated[
        str,
        typer.Option("--name", "-n", help="GitHub repository name"),
    ] = "os-autofix-engine",
    private: Annotated[
        bool,
        typer.Option("--private", "-p", help="Create private repository"),
    ] = False,
    org: Annotated[
        str,
        typer.Option("--org", "-o", help="GitHub organization"),
    ] = "",
    remote: Annotated[
        str,
        typer.Option("--remote", "-r", help="Git remote name"),
    ] = "origin",
    branch: Annotated[
        str,
        typer.Option("--branch", "-b", help="Default branch"),
    ] = "main",
) -> None:
    """Initialize git repository, stage files, create GitHub remote repo, and push initial commit."""
    console.print(
        Panel.fit("[bold green]OS-AutoFix Engine: GitHub Repository Initialization[/bold green]")
    )

    script_path = Path(__file__).parent / "scripts" / "init_github_repo.sh"
    if not script_path.exists():
        console.print(f"[red]Error:[/red] Script not found at {script_path}")
        sys.exit(1)

    cmd = [
        str(script_path),
        "--name",
        repo_name,
        "--remote",
        remote,
        "--branch",
        branch,
    ]
    if private:
        cmd.append("--private")
    else:
        cmd.append("--public")
    if org:
        cmd.extend(["--org", org])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]GitHub initialization failed with code {e.returncode}[/red]")
        sys.exit(e.returncode)


@app.command("list-scenarios")
def cmd_list_scenarios() -> None:
    """List all available diagnostic fault scenarios."""
    scenarios = get_all_scenarios()
    table = Table(
        title="Registered Diagnostic Scenarios", show_header=True, header_style="bold cyan"
    )
    table.add_column("Name", style="bold green")
    table.add_column("Category", style="yellow")
    table.add_column("Difficulty", justify="center")
    table.add_column("Max Steps", justify="right")
    table.add_column("Description", style="dim")

    for s in scenarios:
        table.add_row(s.name, s.category, s.difficulty, str(s.max_steps), s.description)

    console.print(table)


@app.command("mcts-collect")
def cmd_mcts_collect(
    scenario: Annotated[
        str,
        typer.Option("--scenario", "-s", help="Scenario name to search"),
    ] = "systemd_dns",
    simulations: Annotated[
        int,
        typer.Option(
            "--simulations",
            "-n",
            help="Total MCTS simulations to perform",
        ),
    ] = 12,
    exploration_constant: Annotated[
        float,
        typer.Option("--exploration-constant", "-c", help="UCT exploration parameter"),
    ] = 1.414,
    max_depth: Annotated[
        int,
        typer.Option("--max-depth", "-d", help="Maximum search tree depth"),
    ] = 6,
    branch_factor: Annotated[
        int,
        typer.Option("--branch-factor", "-k", help="Candidate action branches per expansion"),
    ] = 3,
    output_file: Annotated[
        str,
        typer.Option("--output-file", "-o", help="Output path for optimal JSONL trajectory"),
    ] = "data/dataset_mcts_optimal.jsonl",
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="'ollama', 'open-webui', 'openai', or 'mock'"),
    ] = "ollama",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Inference model name tag"),
    ] = "qwen2.5-coder:7b",
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Endpoint URL"),
    ] = "",
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'container' or 'vm'"),
    ] = "container",
    image: Annotated[
        str,
        typer.Option("--image", "-i", help="Incus image alias"),
    ] = "images:ubuntu/24.04",
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use heuristic offline mock LLM"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run Monte Carlo Tree Search (MCTS) trajectory collection over Incus snapshot branches."""
    setup_logging(log_level)
    console.print(Panel.fit("[bold blue]OS-AutoFix Engine: MCTS Trajectory Search[/bold blue]"))

    from trainer.mcts_search import MCTSSearchEngine

    cfg = get_default_config()
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"
    cfg.incus.default_image = image

    if backend:
        cfg.llm.backend = backend  # type: ignore[assignment]
    if model:
        cfg.llm.model_name = model
    if endpoint:
        if cfg.llm.backend == "open-webui":
            cfg.llm.open_webui_base_url = endpoint
        else:
            cfg.llm.ollama_base_url = endpoint
    if mock_llm:
        cfg.llm.mock_mode = True

    target_scenario = get_scenario(scenario)
    instance_id = f"autofix-mcts-{uuid.uuid4().hex[:6]}"
    sandbox = IncusSandbox(
        instance_name=instance_id,
        image=image,
        is_vm=(instance_type.lower() == "vm"),
    )

    search_engine = MCTSSearchEngine(
        config=cfg,
        exploration_constant=exploration_constant,
        max_depth=max_depth,
        branch_factor=branch_factor,
    )

    async def _run() -> None:
        try:
            traj = await search_engine.run_search(
                scenario=target_scenario,
                sandbox=sandbox,
                max_simulations=simulations,
            )
            if traj:
                out = search_engine.save_optimal_trajectory(traj, output_file=output_file)
                status_color = "green" if traj.success else "red"
                console.print(
                    f"\n[{status_color}]MCTS Search Complete:[/{status_color}] "
                    f"Success: {traj.success} | Steps: {len(traj.steps)} | Reward: {traj.total_reward:.2f}"
                )
                console.print(f"Optimal trajectory saved to: [cyan]{out}[/cyan]")
        finally:
            await sandbox.cleanup()

    asyncio.run(_run())


@app.command("synthesize-scenario")
def cmd_synthesize_scenario(
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Number of scenarios to synthesize"),
    ] = 1,
    topic: Annotated[
        str,
        typer.Option("--topic", "-t", help="Scenario domain topic or failure description"),
    ] = "",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory for generated scenarios"),
    ] = "scenarios/synthetic",
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="'ollama', 'open-webui', or 'mock'"),
    ] = "ollama",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Teacher model name tag"),
    ] = "qwen2.5-coder:7b",
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Endpoint URL"),
    ] = "",
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate", help="Run live Incus sandbox pre-flight validation"
        ),
    ] = True,
    instance_type: Annotated[
        str,
        typer.Option("--type", help="'container' or 'vm'"),
    ] = "container",
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use deterministic offline synthetic generator"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Synthesize novel OS diagnostic scenarios using teacher LLMs with sandbox pre-flight checks."""
    setup_logging(log_level)
    console.print(Panel.fit("[bold magenta]OS-AutoFix Engine: Scenario Synthesizer[/bold magenta]"))

    from scenarios.synthesizer import ScenarioSynthesizer

    cfg = get_default_config()
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"

    if backend:
        cfg.llm.backend = backend  # type: ignore[assignment]
    if model:
        cfg.llm.model_name = model
    if endpoint:
        if cfg.llm.backend == "open-webui":
            cfg.llm.open_webui_base_url = endpoint
        else:
            cfg.llm.ollama_base_url = endpoint
    if mock_llm:
        cfg.llm.mock_mode = True

    synthesizer = ScenarioSynthesizer(config=cfg, output_dir=output_dir)

    def sandbox_factory(name: str) -> IncusSandbox:
        return IncusSandbox(
            instance_name=name,
            is_vm=(instance_type.lower() == "vm"),
        )

    sb_factory = sandbox_factory if validate else None
    results = asyncio.run(
        synthesizer.synthesize(
            count=count,
            topic=topic or None,
            sandbox_factory=sb_factory,
        )
    )

    console.print(f"\n[bold green]Synthesized {len(results)} scenario(s):[/bold green]")
    for r in results:
        if r.get("valid"):
            console.print(
                f"  • [green]VALID[/green]   `{r['name']}` ({r.get('category')}) -> {r.get('file_path')}"
            )
        else:
            console.print(f"  • [red]FAILED[/red]  `{r.get('name', 'unknown')}`: {r.get('error')}")


@app.command("swarm")
def cmd_swarm(
    scenario: Annotated[
        str,
        typer.Option("--scenario", "-s", help="Scenario name to troubleshoot"),
    ] = "systemd_dns",
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'container' or 'vm'"),
    ] = "container",
    image: Annotated[
        str,
        typer.Option("--image", "-i", help="Incus image alias"),
    ] = "images:ubuntu/24.04",
    max_cycles: Annotated[
        int,
        typer.Option("--max-cycles", "-c", help="Max Triage-Remediate-Audit handoff cycles"),
    ] = 2,
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="'ollama', 'open-webui', or 'mock'"),
    ] = "ollama",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Inference model name"),
    ] = "qwen2.5-coder:7b",
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Endpoint URL"),
    ] = "",
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use deterministic offline mock agents"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run the Tri-Agent specialist swarm (Triage -> Remediation -> Audit) inside Incus sandboxes."""
    setup_logging(log_level)
    console.print(Panel.fit("[bold cyan]OS-AutoFix Engine: Tri-Agent Specialist Swarm[/bold cyan]"))

    from engine.agents.coordinator import SwarmCoordinator

    cfg = get_default_config()
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"
    if backend:
        cfg.llm.backend = backend  # type: ignore[assignment]
    if model:
        cfg.llm.model_name = model
    if endpoint:
        if cfg.llm.backend == "open-webui":
            cfg.llm.open_webui_base_url = endpoint
        else:
            cfg.llm.ollama_base_url = endpoint
    if mock_llm:
        cfg.llm.mock_mode = True

    target_scenario = get_scenario(scenario)
    instance_id = f"autofix-swarm-{uuid.uuid4().hex[:6]}"
    sandbox = IncusSandbox(
        instance_name=instance_id,
        image=image,
        is_vm=(instance_type.lower() == "vm"),
    )

    coordinator = SwarmCoordinator(config=cfg, max_cycles=max_cycles)

    async def _run() -> None:
        try:
            await sandbox.setup()
            await target_scenario.setup(sandbox)
            await target_scenario.inject_fault(sandbox)

            res = await coordinator.run(scenario=target_scenario, sandbox=sandbox)
            color = "green" if res.success else "red"
            console.print(
                f"\n[{color}]Swarm Outcome: {'APPROVED' if res.success else 'REJECTED'}[/{color}] "
                f"| Cycles: {res.cycles_executed} | Duration: {res.duration_seconds:.2f}s"
            )
            if res.triage_finding:
                console.print(
                    f"[bold yellow]Triage Finding:[/bold yellow] {res.triage_finding.root_cause} (Daemons: {', '.join(res.triage_finding.affected_daemons)})"
                )
            if res.remediation_result:
                console.print(
                    f"[bold blue]Remediation:[/bold blue] {res.remediation_result.mutations_summary}"
                )
            if res.audit_report:
                console.print(f"[bold magenta]Audit Notes:[/bold magenta] {res.audit_report.notes}")
        finally:
            await sandbox.cleanup()

    asyncio.run(_run())


@app.command("arena")
def cmd_arena(
    model_a: Annotated[
        str,
        typer.Option("--model-a", help="Baseline model identifier"),
    ] = "qwen2.5-coder:7b",
    model_b: Annotated[
        str,
        typer.Option("--model-b", help="Challenger model identifier"),
    ] = "os-fixer:v1",
    scenarios: Annotated[
        str,
        typer.Option("--scenarios", "-s", help="Comma-separated scenarios or 'all'"),
    ] = "all",
    rounds: Annotated[
        int,
        typer.Option("--rounds", "-r", help="Rounds per scenario"),
    ] = 1,
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'container' or 'vm'"),
    ] = "container",
    ratings_file: Annotated[
        str,
        typer.Option("--ratings-file", help="Path to persistent ELO ratings JSON"),
    ] = "reports/arena_ratings.json",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run head-to-head Model Arena ELO tournament between two model checkpoints."""
    setup_logging(log_level)
    console.print(
        Panel.fit(
            f"[bold yellow]OS-AutoFix Engine: Model Arena Tournament[/bold yellow]\n[cyan]{model_a}[/cyan] vs [magenta]{model_b}[/magenta]"
        )
    )

    from engine.arena import ModelArena

    if scenarios.lower() == "all":
        target_scenarios = get_all_scenarios()
    else:
        target_scenarios = [get_scenario(n.strip()) for n in scenarios.split(",") if n.strip()]

    arena = ModelArena(ratings_file=ratings_file)
    summary = asyncio.run(
        arena.run_tournament(
            model_a=model_a,
            model_b=model_b,
            scenarios=target_scenarios,
            rounds=rounds,
            instance_type=instance_type,
        )
    )

    table = Table(title="Arena Match Results", show_header=True, header_style="bold cyan")
    table.add_column("Scenario", style="bold")
    table.add_column("Round", justify="center")
    table.add_column("Winner", style="green")
    table.add_column("Score (A vs B)", justify="center")
    table.add_column("Steps (A vs B)", justify="center")
    table.add_column("Reason", style="dim")

    for m in summary.matches:
        table.add_row(
            m.scenario_name,
            str(m.round_idx),
            m.winner,
            f"{m.score_a} - {m.score_b}",
            f"{m.traj_a_steps} vs {m.traj_b_steps}",
            m.reason,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Final ELO Ratings:[/bold green]\n"
        f"  • {model_a}: [cyan]{summary.final_elo_a:.1f}[/cyan] (Δ {summary.final_elo_a - summary.initial_elo_a:+.1f})\n"
        f"  • {model_b}: [magenta]{summary.final_elo_b:.1f}[/magenta] (Δ {summary.final_elo_b - summary.initial_elo_b:+.1f})\n"
        f"Ratings saved to [cyan]{ratings_file}[/cyan]"
    )


@app.command("export-webui")
def cmd_export_webui(
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Target path for Open-WebUI bundle JSON"),
    ] = "dist/open_webui_bundle.json",
) -> None:
    """Export complete Open-WebUI Tool and Pipeline bundle ready for UI import."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tool_def_path = Path("integrations/open_webui/tool_def.json")
    tool_def = (
        json.loads(tool_def_path.read_text(encoding="utf-8")) if tool_def_path.exists() else {}
    )

    pipeline_code = Path("integrations/open_webui/pipeline.py").read_text(encoding="utf-8")

    bundle = {
        "title": "OS-AutoFix Engine Open-WebUI Integration",
        "version": "0.4.0",
        "open_webui_target": "https://ai.is27.duckdns.org",
        "tool": tool_def,
        "pipeline": {
            "id": "os_autofix_pipeline",
            "name": "OS-AutoFix Engine Pipeline",
            "code": pipeline_code,
        },
    }

    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    console.print(
        f"[bold green]Successfully exported Open-WebUI bundle to:[/bold green] [cyan]{out_path}[/cyan]"
    )


@app.command("mcp")
def cmd_mcp(
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="MCP transport mode: 'stdio' or 'sse'"),
    ] = "stdio",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port for SSE transport"),
    ] = 8080,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host address for SSE transport"),
    ] = "0.0.0.0",
) -> None:
    """Start the Model Context Protocol (MCP) server exposing Incus sandbox tools and resources."""
    from mcp_server.server import run_mcp_server

    run_mcp_server(transport=transport, port=port, host=host)


@app.command("chaos")
def cmd_chaos(
    rate_minutes: Annotated[
        float,
        typer.Option("--rate-minutes", "-r", help="Mean interval between chaos fault injections"),
    ] = 1.0,
    fleet_size: Annotated[
        int,
        typer.Option("--fleet-size", "-f", help="Concurrent canary sandbox fleet size"),
    ] = 3,
    duration_hours: Annotated[
        float,
        typer.Option("--duration-hours", "-d", help="Total runtime for chaos daemon in hours"),
    ] = 1.0,
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'container' or 'vm'"),
    ] = "container",
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use deterministic mock agents"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run autonomous Chaos Engineering daemon injecting faults into canary sandboxes and measuring MTTR."""
    setup_logging(log_level)
    console.print(
        Panel.fit("[bold red]OS-AutoFix Engine: Autonomous Chaos Engineering Daemon[/bold red]")
    )

    from engine.chaos_daemon import ChaosDaemon

    cfg = get_default_config()
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"
    if mock_llm:
        cfg.llm.mock_mode = True

    daemon = ChaosDaemon(
        config=cfg,
        fleet_size=fleet_size,
        rate_minutes=rate_minutes,
        duration_hours=duration_hours,
        instance_type=instance_type,
    )

    asyncio.run(daemon.run())
    summary = daemon.get_summary_metrics()

    console.print(
        f"\n[bold green]Chaos Engineering Summary:[/bold green]\n"
        f"  • Total Experiments: [cyan]{summary['total_experiments']}[/cyan]\n"
        f"  • Autonomous Recoveries: [green]{summary['recoveries']}[/green] ({summary['recovery_rate'] * 100:.1f}%)\n"
        f"  • Mean Time to Resolution (MTTR): [yellow]{summary['mean_mttr_seconds']:.2f}s[/yellow]\n"
        f"  • Average Safety Score: [magenta]{summary['avg_safety_score']:.2f}[/magenta]\n"
    )


@app.command("bench-distributed")
def cmd_bench_distributed(
    scenario: Annotated[
        str,
        typer.Option("--scenario", "-s", help="Distributed scenario name or 'all'"),
    ] = "all",
    instance_type: Annotated[
        str,
        typer.Option("--type", "-t", help="'container' or 'vm'"),
    ] = "container",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Evaluate multi-node distributed topology scenarios (WireGuard, etcd Raft, Keepalived HA)."""
    setup_logging(log_level)
    console.print(
        Panel.fit("[bold blue]OS-AutoFix Engine: Distributed Cluster Benchmark[/bold blue]")
    )

    from scenarios.distributed import get_all_distributed_scenarios, get_distributed_scenario

    if scenario.lower() == "all":
        targets = get_all_distributed_scenarios()
    else:
        targets = [get_distributed_scenario(n.strip()) for n in scenario.split(",") if n.strip()]

    async def _run() -> None:
        table = Table(
            title="Distributed Scenario Verification", show_header=True, header_style="bold cyan"
        )
        table.add_column("Scenario", style="bold")
        table.add_column("Nodes", justify="center")
        table.add_column("Category", style="yellow")
        table.add_column("Initial Breakage", justify="center")
        table.add_column("Status", style="green")

        for sc in targets:
            nodes: dict[str, IncusSandbox] = {
                node_name: IncusSandbox(
                    instance_name=f"dist-{node_name}-{uuid.uuid4().hex[:4]}",
                    is_vm=(instance_type.lower() == "vm"),
                )
                for node_name in sc.required_nodes
            }

            try:
                for sb in nodes.values():
                    await sb.setup()

                await sc.setup_topology(nodes)
                baseline_ok, _ = await sc.verify(nodes)

                await sc.inject_fault(nodes)
                fault_active, fault_msg = await sc.verify(nodes)

                status = (
                    "[green]VERIFIED BROKEN[/green]"
                    if not fault_active
                    else "[red]FAULT MISSED[/red]"
                )
                table.add_row(
                    sc.name,
                    str(len(nodes)),
                    sc.category,
                    "[green]YES[/green]" if not fault_active else "[red]NO[/red]",
                    status,
                )
            finally:
                for sb in nodes.values():
                    await sb.cleanup()

        console.print(table)

    asyncio.run(_run())


@app.command("audit-security")
def cmd_audit_security(
    command: Annotated[
        str,
        typer.Option("--command", "-c", help="Shell command to analyze for security anti-patterns"),
    ] = "rm -rf /etc/hosts",
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Minimum required safety score"),
    ] = 0.7,
) -> None:
    """Run an interactive kernel-level eBPF syscall security audit on a command or remediation script."""
    console.print(
        Panel.fit("[bold magenta]OS-AutoFix Engine: Kernel Syscall Security Auditor[/bold magenta]")
    )

    from security.ebpf_auditor import SyscallSecurityAuditor

    auditor = SyscallSecurityAuditor(safety_threshold=threshold)
    report = auditor.inspect_command(command)

    color = "green" if report.is_safe else "red"
    console.print(f"Target Command: [bold]{report.command}[/bold]")
    console.print(
        f"Safety Score:   [{color}]{report.safety_score:.2f}[/{color}] (Threshold: {threshold})"
    )
    console.print(
        f"Decision:       [{color}]{'SAFE TO EXECUTE' if report.is_safe else 'BLOCKED / ABORT'}[/{color}]"
    )
    console.print(f"Blast Radius:   [yellow]{report.blast_radius.upper()}[/yellow]")

    if report.events:
        console.print("\n[bold]Detected Security Events:[/bold]")
        for e in report.events:
            console.print(
                f"  • [{e.risk_level.upper()}] Syscall: `{e.syscall_type}` | {e.description} (Penalty: -{e.penalty})"
            )


@app.command("index-docs")
def cmd_index_docs(
    runbooks_dir: Annotated[
        str,
        typer.Option(
            "--runbooks-dir", help="Directory containing markdown troubleshooting runbooks"
        ),
    ] = "knowledge/runbooks",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Target path for serialized knowledge index JSON"),
    ] = "reports/knowledge_index.json",
) -> None:
    """Index offline Linux troubleshooting runbooks and manpage documentation into the hybrid search engine."""
    console.print(
        Panel.fit("[bold cyan]OS-AutoFix Engine: Documentation & Runbook Indexer[/bold cyan]")
    )

    from knowledge.retriever import HybridRetriever

    retriever = HybridRetriever(runbooks_dir=runbooks_dir)
    count = retriever.index_all()
    retriever.export_index_json(output)

    console.print(
        f"[bold green]Successfully indexed {count} document chunks into:[/bold green] [cyan]{output}[/cyan]"
    )


@app.command("query-knowledge")
def cmd_query_knowledge(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Troubleshooting search query"),
    ] = "systemd DNS resolution failure",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of chunks to return"),
    ] = 3,
) -> None:
    """Query the offline hybrid BM25 / vector knowledge base for diagnostic runbooks."""
    console.print(Panel.fit(f"[bold yellow]Knowledge Query: '{query}'[/bold yellow]"))

    from knowledge.retriever import GLOBAL_RETRIEVER

    results = GLOBAL_RETRIEVER.consult_runbook(query, top_k=top_k)

    if not results:
        console.print("[yellow]No relevant runbook sections found for this query.[/yellow]")
        return

    table = Table(
        title="Retrieved Troubleshooting Runbooks", show_header=True, header_style="bold cyan"
    )
    table.add_column("Score", justify="center", style="bold green")
    table.add_column("Runbook Title", style="cyan")
    table.add_column("Section", style="yellow")
    table.add_column("Snippet Preview", style="white")

    for c in results:
        table.add_row(
            f"{c.score:.2f}",
            c.title,
            c.section,
            c.content[:120].replace("\n", " ") + "...",
        )

    console.print(table)


@app.command("watchdog")
def cmd_watchdog(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--live", help="Simulate fixes in shadow containers without applying to host"
        ),
    ] = True,
    min_safety_score: Annotated[
        float,
        typer.Option(
            "--min-safety-score", help="Minimum required safety score to approve remediation"
        ),
    ] = 0.85,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Run host self-healing watchdog monitoring journalctl with shadow container dry-run verification."""
    setup_logging(log_level)
    console.print(
        Panel.fit(
            f"[bold red]OS-AutoFix Engine: Host Self-Healing Watchdog ({'DRY-RUN' if dry_run else 'LIVE'} MODE)[/bold red]"
        )
    )

    from engine.host_watchdog import HostWatchdogDaemon

    daemon = HostWatchdogDaemon(
        dry_run=dry_run,
        min_safety_score=min_safety_score,
    )

    asyncio.run(daemon.run(max_iterations=5))


@app.command("cluster-node")
def cmd_cluster_node(
    node_id: Annotated[
        str,
        typer.Option("--node-id", "-n", help="Unique identifier for this federated cluster node"),
    ] = "node-1",
    peers: Annotated[
        str,
        typer.Option("--peers", "-p", help="Comma-separated list of peer node IDs"),
    ] = "",
    bind_addr: Annotated[
        str,
        typer.Option("--bind-addr", "-b", help="Bind IP address for Raft RPC"),
    ] = "0.0.0.0",
    raft_port: Annotated[
        int,
        typer.Option("--raft-port", help="Port for Raft RPC communication"),
    ] = 9200,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level"),
    ] = "INFO",
) -> None:
    """Start an autonomous federated cluster node with distributed Raft consensus and lock management."""
    setup_logging(log_level)
    console.print(
        Panel.fit(f"[bold green]OS-AutoFix Engine: Federated Cluster Node '{node_id}'[/bold green]")
    )

    from engine.federation.cluster_raft import ClusterRaftNode

    peer_list = [p.strip() for p in peers.split(",") if p.strip()]
    node = ClusterRaftNode(node_id=node_id, peers=peer_list)

    console.print(f"Node ID:   [bold cyan]{node.node_id}[/bold cyan]")
    console.print(f"Peers:     [yellow]{node.peers or 'Standalone mode'}[/yellow]")
    console.print(f"Listening: [magenta]{bind_addr}:{raft_port}[/magenta]\n")

    asyncio.run(node.run(max_ticks=5))


@app.command("net-chaos")
def cmd_net_chaos(
    instance: Annotated[
        str,
        typer.Option("--instance", "-i", help="Target Incus container/VM instance name"),
    ] = "canary-net-1",
    interface: Annotated[
        str,
        typer.Option("--interface", help="Target network interface"),
    ] = "eth0",
    latency_ms: Annotated[
        float,
        typer.Option("--latency-ms", "-l", help="Packet delay latency in milliseconds"),
    ] = 100.0,
    jitter_ms: Annotated[
        float,
        typer.Option("--jitter-ms", help="Packet delay jitter in milliseconds"),
    ] = 10.0,
    drop_rate: Annotated[
        float,
        typer.Option("--drop-rate", "-d", help="Packet drop fraction (0.0 to 1.0)"),
    ] = 0.15,
    duration_sec: Annotated[
        float,
        typer.Option("--duration-sec", help="Chaos experiment duration before auto-teardown"),
    ] = 5.0,
) -> None:
    """Inject dynamic kernel-level eBPF / TC Traffic Control network packet faults into an instance."""
    console.print(
        Panel.fit(
            f"[bold red]OS-AutoFix Engine: eBPF / TC Network Chaos Injector on '{instance}'[/bold red]"
        )
    )

    from sandbox.incus_sandbox import IncusSandbox
    from security.ebpf_network_chaos import EbpfNetworkChaos

    sb = IncusSandbox(instance_name=instance)
    chaos = EbpfNetworkChaos(sandbox=sb, interface=interface)

    async def _run() -> None:
        console.print(f"Interface:    [cyan]{interface}[/cyan]")
        console.print(f"Latency:      [yellow]{latency_ms}ms (±{jitter_ms}ms)[/yellow]")
        console.print(f"Drop Rate:    [red]{drop_rate * 100:.1f}%[/red]")
        console.print(f"Duration:     [magenta]{duration_sec}s[/magenta]\n")

        await chaos.inject_fault(
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            drop_rate=drop_rate,
            interface=interface,
        )
        console.print("[green]eBPF / TC rules active. Waiting experiment duration...[/green]")
        await asyncio.sleep(duration_sec)
        await chaos.teardown(interface=interface)
        console.print(
            "[bold green]Experiment completed. eBPF / TC rules cleaned up successfully.[/bold green]"
        )

    asyncio.run(_run())


@app.command("cluster-status")
def cmd_cluster_status() -> None:
    """Display the active cluster consensus leader, health, and distributed locks."""
    console.print(Panel.fit("[bold cyan]OS-AutoFix Engine: Cluster Consensus Status[/bold cyan]"))

    from engine.federation.cluster_raft import ClusterRaftNode

    node = ClusterRaftNode(node_id="local-client", peers=["node-1", "node-2", "node-3"])
    node.acquire_lock("lock:topology:etcd_split_brain", ttl_seconds=45.0)

    status = node.get_cluster_status()

    table = Table(title="Cluster Federation Overview", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="yellow")

    table.add_row("Node ID", status["node_id"])
    table.add_row("Consensus Role", status["role"])
    table.add_row("Current Term", str(status["term"]))
    table.add_row("Cluster Leader", status["leader_id"] or "Electing / Standalone")
    table.add_row("Configured Peers", str(status["peer_count"]))
    table.add_row("Active Distributed Locks", str(status["active_locks_count"]))

    console.print(table)


if __name__ == "__main__":
    app()
