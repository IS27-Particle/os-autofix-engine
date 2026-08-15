"""Unit tests for Open-WebUI pipeline streaming and tool definitions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from integrations.open_webui.pipeline import Pipeline
from tests.conftest import MockSandbox


def test_pipeline_scenario_matching() -> None:
    """Test matching user prompts to registered fault scenarios."""
    pipe = Pipeline()
    assert pipe._match_scenario("Fix DNS resolution failure in systemd") == "systemd_dns"
    assert pipe._match_scenario("Default route missing on eth0") == "network_routing"
    assert pipe._match_scenario("ZFS pool mountpoint not accessible") == "zfs_mount"
    assert pipe._match_scenario("Docker socket permission denied") == "docker_socket"
    assert pipe._match_scenario("IPTables blocking outbound traffic") == "iptables_lockout"


def test_tool_def_json_validity() -> None:
    """Test structure and validity of tool_def.json schema."""
    tool_file = Path("integrations/open_webui/tool_def.json")
    assert tool_file.exists()
    data = json.loads(tool_file.read_text(encoding="utf-8"))

    assert data["id"] == "os_autofix_tools"
    assert "specs" in data
    spec_names = [s["name"] for s in data["specs"]]
    assert "list_scenarios" in spec_names
    assert "run_command" in spec_names


@pytest.mark.asyncio
async def test_pipeline_streaming_execution() -> None:
    """Test streaming chunks from pipeline with mock sandbox."""
    from config.settings import EngineConfig

    pipe = Pipeline()
    mock_sb = MockSandbox("webui-mock")
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    with patch("integrations.open_webui.pipeline.IncusSandbox", return_value=mock_sb):
        with patch("integrations.open_webui.pipeline.get_default_config", return_value=cfg):
            chunks: list[str] = []
            async for chunk in pipe.pipe(
                user_message="Please fix broken DNS",
                model_id="qwen2.5-coder:7b",
                messages=[],
                body={},
            ):
                chunks.append(chunk)

            full_stream = "".join(chunks)
            assert "OS-AutoFix Autonomous Diagnostic Engine" in full_stream
            assert "systemd_dns" in full_stream
            assert "RESOLVED" in full_stream or "Ephemeral sandbox destroyed" in full_stream
