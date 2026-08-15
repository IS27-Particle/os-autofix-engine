"""Unit tests for Model Context Protocol (MCP) server tools, resources, and configuration generator."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mcp_server.server import (
    ACTIVE_SANDBOXES,
    resource_benchmark_report,
    resource_cluster_status,
    tool_create_sandbox,
    tool_destroy_sandbox,
    tool_exec_command,
    tool_inject_fault,
    tool_list_scenarios,
    tool_revert_sandbox,
    tool_verify_fix,
)
from scripts.register_mcp import generate_mcp_configs
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_mcp_tool_list_scenarios() -> None:
    """Test listing diagnostic scenarios through MCP tool."""
    scenarios = await tool_list_scenarios()
    assert len(scenarios) == 8
    names = [s["name"] for s in scenarios]
    assert "systemd_dns" in names
    assert "zfs_mount" in names
    assert "docker_socket" in names
    assert "iptables_lockout" in names
    assert "mac_enforcement" in names


@pytest.mark.asyncio
async def test_mcp_sandbox_lifecycle_and_execution() -> None:
    """Test complete MCP tool lifecycle with mock sandbox."""
    mock_sb = MockSandbox("autofix-mcp-test")

    with patch("mcp_server.server.IncusSandbox", return_value=mock_sb):
        # 1. Create sandbox
        create_res = await tool_create_sandbox(instance_type="container")
        instance_id = create_res["instance_id"]
        assert create_res["status"] == "READY"
        assert instance_id in ACTIVE_SANDBOXES

        # 2. Inject fault
        inject_res = await tool_inject_fault(instance_id, scenario_name="systemd_dns")
        assert inject_res["success"] is True
        assert inject_res["scenario"] == "systemd_dns"
        assert inject_res["initial_fault_active"] is True

        # 3. Execute command
        exec_res = await tool_exec_command(
            instance_id, command="systemctl restart systemd-resolved"
        )
        assert exec_res["success"] is True
        assert "restarted" in exec_res["stdout"]

        # 4. Verify fix
        verify_res = await tool_verify_fix(instance_id, scenario_name="systemd_dns")
        assert verify_res["success"] is True
        assert verify_res["is_resolved"] is True

        # 5. Revert sandbox
        revert_res = await tool_revert_sandbox(instance_id, snapshot_name="snap-clean")
        assert revert_res["success"] is True
        assert revert_res["snapshot_restored"] == "snap-clean"

        # 6. Destroy sandbox
        destroy_res = await tool_destroy_sandbox(instance_id)
        assert destroy_res["success"] is True
        assert instance_id not in ACTIVE_SANDBOXES


@pytest.mark.asyncio
async def test_mcp_invalid_instance_handling() -> None:
    """Test error handling when passing non-existent instance IDs."""
    fake_id = "non-existent-box"
    res_exec = await tool_exec_command(fake_id, command="ls")
    assert res_exec["success"] is False
    assert "not found" in res_exec["error"]

    res_fault = await tool_inject_fault(fake_id, scenario_name="systemd_dns")
    assert res_fault["success"] is False

    res_destroy = await tool_destroy_sandbox(fake_id)
    assert res_destroy["success"] is False


@pytest.mark.asyncio
async def test_mcp_resources() -> None:
    """Test MCP resource providers for benchmark reports and cluster status."""
    report_text = await resource_benchmark_report()
    assert report_text is not None

    cluster_json = await resource_cluster_status()
    data = json.loads(cluster_json)
    assert "active_sandboxes_count" in data
    assert "kvm" in data
    assert "incus" in data
    assert "storage_pools" in data


def test_register_mcp_configs() -> None:
    """Test generation of client configuration snippets."""
    configs = generate_mcp_configs()
    assert "claude_desktop" in configs
    assert "open_webui" in configs
    assert "os-autofix" in configs["claude_desktop"]["mcpServers"]
    args = configs["claude_desktop"]["mcpServers"]["os-autofix"]["args"]
    assert "mcp" in args
