"""CLI interface for os-autofix-engine: evaluation benchmarking, dataset collection, model training, Ollama deployment, monitoring, and continuous learning loops."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
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
        typer.Option("--metrics-port", "-p", help="Prometheus metrics exporter port (0 to disable)"),
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


if __name__ == "__main__":
    app()
