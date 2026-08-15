"""Model Context Protocol (MCP) server exposing Incus sandbox lifecycle, fault injection, and diagnostics to external AI agents."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from config.settings import get_default_config
from deploy.provisioner import HostProvisioner
from engine.orchestrator import Orchestrator
from engine.reporter import BenchmarkReporter
from sandbox.base import BaseSandbox
from sandbox.incus_sandbox import IncusSandbox
from scenarios.registry import get_all_scenarios, get_scenario
from trainer.trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger("os_autofix.mcp_server")

# Global tracking of active sandboxes created via MCP
ACTIVE_SANDBOXES: dict[str, BaseSandbox] = {}

mcp = MCPServer(
    name="os-autofix-engine",
    description="Autonomous OS-level policy evaluation and Incus sandbox control server.",
    version="0.2.0",
)


@mcp.tool()
async def tool_list_scenarios() -> list[dict[str, Any]]:
    """Return all available operating system diagnostic scenarios with categories and difficulties."""
    scenarios = get_all_scenarios()
    return [
        {
            "name": s.name,
            "category": s.category,
            "difficulty": s.difficulty,
            "max_steps": s.max_steps,
            "description": s.description,
        }
        for s in scenarios
    ]


@mcp.tool()
async def tool_create_sandbox(
    instance_type: str = "container",
    image: str = "images:ubuntu/24.04",
) -> dict[str, Any]:
    """Spawn an isolated ephemeral Incus container or VM and return the allocated instance ID."""
    instance_id = f"autofix-mcp-{uuid.uuid4().hex[:6]}"
    is_vm = instance_type.lower() == "vm"
    sandbox = IncusSandbox(
        instance_name=instance_id,
        image=image,
        is_vm=is_vm,
    )

    logger.info("MCP: Spawning sandbox '%s' (%s)...", instance_id, instance_type)
    await sandbox.setup()
    await sandbox.create_snapshot("snap-baseline")

    ACTIVE_SANDBOXES[instance_id] = sandbox
    return {
        "instance_id": instance_id,
        "instance_type": "vm" if is_vm else "container",
        "image": image,
        "status": "READY",
        "baseline_snapshot": "snap-baseline",
    }


@mcp.tool()
async def tool_inject_fault(
    instance_id: str,
    scenario_name: str,
) -> dict[str, Any]:
    """Inject a diagnostic fault scenario into an existing sandbox instance."""
    if instance_id not in ACTIVE_SANDBOXES:
        return {
            "success": False,
            "error": f"Sandbox instance '{instance_id}' not found in active registry.",
        }

    sandbox = ACTIVE_SANDBOXES[instance_id]
    try:
        scenario = get_scenario(scenario_name)
    except KeyError as e:
        return {"success": False, "error": str(e)}

    logger.info("MCP: Setting up scenario '%s' in '%s'...", scenario_name, instance_id)
    await scenario.setup(sandbox)
    await sandbox.create_snapshot("snap-clean")

    logger.info("MCP: Injecting fault '%s' in '%s'...", scenario_name, instance_id)
    await scenario.inject_fault(sandbox)
    await sandbox.create_snapshot("snap-fault-injected")

    is_resolved, msg = await scenario.verify(sandbox)
    return {
        "success": True,
        "scenario": scenario_name,
        "instance_id": instance_id,
        "initial_fault_active": not is_resolved,
        "verification_message": msg,
        "prompt": scenario.get_prompt(),
    }


@mcp.tool()
async def tool_exec_command(
    instance_id: str,
    command: str,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Execute a non-interactive shell command inside a sandbox instance with output truncation."""
    if instance_id not in ACTIVE_SANDBOXES:
        return {
            "success": False,
            "error": f"Sandbox instance '{instance_id}' not found in active registry.",
        }

    sandbox = ACTIVE_SANDBOXES[instance_id]
    logger.info("MCP: Executing command in '%s': %s", instance_id, command[:80])
    res = await sandbox.execute(command, timeout_seconds=timeout_seconds)

    return {
        "success": res.exit_code == 0,
        "instance_id": instance_id,
        "command": res.command,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "exit_code": res.exit_code,
        "timed_out": res.timed_out,
    }


@mcp.tool()
async def tool_verify_fix(
    instance_id: str,
    scenario_name: str,
) -> dict[str, Any]:
    """Run the scenario verification check to assert whether the OS fault is resolved."""
    if instance_id not in ACTIVE_SANDBOXES:
        return {
            "success": False,
            "error": f"Sandbox instance '{instance_id}' not found in active registry.",
        }

    sandbox = ACTIVE_SANDBOXES[instance_id]
    try:
        scenario = get_scenario(scenario_name)
    except KeyError as e:
        return {"success": False, "error": str(e)}

    is_resolved, msg = await scenario.verify(sandbox)
    return {
        "success": is_resolved,
        "scenario": scenario_name,
        "instance_id": instance_id,
        "is_resolved": is_resolved,
        "verification_message": msg,
    }


@mcp.tool()
async def tool_revert_sandbox(
    instance_id: str,
    snapshot_name: str = "snap-baseline",
) -> dict[str, Any]:
    """Revert a sandbox instance to a clean or previous snapshot."""
    if instance_id not in ACTIVE_SANDBOXES:
        return {
            "success": False,
            "error": f"Sandbox instance '{instance_id}' not found in active registry.",
        }

    sandbox = ACTIVE_SANDBOXES[instance_id]
    logger.info("MCP: Reverting '%s' to snapshot '%s'...", instance_id, snapshot_name)
    try:
        await sandbox.revert(snapshot_name)
        return {
            "success": True,
            "instance_id": instance_id,
            "snapshot_restored": snapshot_name,
        }
    except Exception as e:
        return {"success": False, "error": f"Snapshot restore failed: {e}"}


@mcp.tool()
async def tool_destroy_sandbox(
    instance_id: str,
) -> dict[str, Any]:
    """Clean up and delete an ephemeral sandbox instance."""
    if instance_id not in ACTIVE_SANDBOXES:
        return {
            "success": False,
            "error": f"Sandbox instance '{instance_id}' not found in active registry.",
        }

    sandbox = ACTIVE_SANDBOXES.pop(instance_id)
    logger.info("MCP: Destroying sandbox '%s'...", instance_id)
    try:
        await sandbox.cleanup()
        return {"success": True, "instance_id": instance_id, "status": "DESTROYED"}
    except Exception as e:
        return {"success": False, "error": f"Cleanup failed: {e}"}


@mcp.tool()
async def tool_run_benchmark(
    scenarios: str = "all",
    workers: int = 4,
    iterations: int = 1,
    instance_type: str = "container",
    model: str = "qwen2.5-coder:7b",
    backend: str = "ollama",
) -> dict[str, Any]:
    """Execute a parallel benchmark run across scenarios and return summary analytics."""
    cfg = get_default_config()
    cfg.workers = workers
    cfg.incus.instance_type = "container" if instance_type.lower() == "container" else "vm"
    cfg.llm.model_name = model
    cfg.llm.backend = backend  # type: ignore[assignment]

    if scenarios.lower() == "all":
        target_scenarios = get_all_scenarios()
    else:
        names = [n.strip() for n in scenarios.split(",") if n.strip()]
        target_scenarios = [get_scenario(n) for n in names]

    buffer = TrajectoryBuffer()
    orchestrator = Orchestrator(config=cfg, trajectory_buffer=buffer)

    results = await orchestrator.run_benchmark(target_scenarios, iterations=iterations)
    reporter = BenchmarkReporter(output_dir="reports")
    summary = reporter.generate_summary_data(results, model_name=model, backend=backend)
    reporter.write_reports(results, model_name=model, backend=backend)

    return {
        "total_episodes": summary["total_episodes"],
        "successful_episodes": summary["successful_episodes"],
        "pass_rate": summary["pass_rate"],
        "avg_duration_seconds": summary["avg_duration_seconds"],
        "avg_reward": summary["avg_reward"],
        "scenarios": summary["scenarios"],
    }


@mcp.resource("report://benchmark/latest")
async def resource_benchmark_report() -> str:
    """Return latest benchmark summary report in Markdown format."""
    report_file = Path("reports/benchmark_latest.md")
    if report_file.exists():
        return report_file.read_text(encoding="utf-8")
    return "# No Benchmark Reports Available\nRun `bench` to generate evaluation reports."


@mcp.resource("status://cluster")
async def resource_cluster_status() -> str:
    """Return health status of Incus hypervisor, KVM, and remote LLM endpoints."""
    provisioner = HostProvisioner()
    kvm = provisioner.check_kvm()
    incus = provisioner.check_incus()
    storage = provisioner.check_storage_pools()
    bridges = provisioner.check_network_bridge()
    ollama = provisioner.check_ollama()

    status_data = {
        "active_sandboxes_count": len(ACTIVE_SANDBOXES),
        "active_instances": list(ACTIVE_SANDBOXES.keys()),
        "kvm": kvm,
        "incus": incus,
        "storage_pools": storage,
        "network_bridges": bridges,
        "ollama": ollama,
    }
    return json.dumps(status_data, indent=2)


def run_mcp_server(transport: str = "stdio", port: int = 8080, host: str = "0.0.0.0") -> None:
    """Entrypoint to run the MCP server with chosen transport (stdio or sse)."""
    if transport.lower() == "sse":
        logger.info("Starting MCP Server over SSE on http://%s:%d...", host, port)
        mcp.run(transport="sse")
    else:
        logger.info("Starting MCP Server over stdio...")
        mcp.run(transport="stdio")
