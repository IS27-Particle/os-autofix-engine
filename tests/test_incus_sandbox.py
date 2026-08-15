"""Unit and integration tests for IncusSandbox driver."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import IncusConfig
from sandbox.incus_sandbox import IncusAgentTimeoutError, IncusSandbox


@pytest.mark.asyncio
async def test_incus_sandbox_initialization() -> None:
    """Test default properties and configuration binding."""
    cfg = IncusConfig(
        instance_prefix="testbox",
        default_image="images:ubuntu/24.04",
        instance_type="vm",
    )
    sb = IncusSandbox(config=cfg)
    assert sb.instance_name.startswith("testbox-")
    assert sb.image == "images:ubuntu/24.04"
    assert sb.is_vm is True


@pytest.mark.asyncio
async def test_incus_sandbox_truncation() -> None:
    """Test strict 2000 character output truncation enforcement."""
    cfg = IncusConfig(max_output_chars=2000)
    sb = IncusSandbox(instance_name="mock-box", config=cfg)

    long_output = "A" * 3500

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(long_output.encode("utf-8"), b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        res = await sb.execute("cat /huge_log.txt")

        assert res.exit_code == 0
        assert res.truncated is True
        assert len(res.stdout) < 2500
        assert "STDOUT TRUNCATED" in res.stdout
        assert res.stdout.startswith("A" * 2000)


@pytest.mark.asyncio
async def test_incus_sandbox_timeout_handling() -> None:
    """Test 15-second execution timeout kills hung guest processes."""
    cfg = IncusConfig(command_timeout_seconds=2)
    sb = IncusSandbox(instance_name="mock-box", config=cfg)

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=None)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        res = await sb.execute("sleep 100", timeout_seconds=1)

        assert res.timed_out is True
        assert res.exit_code == 124
        assert "timed out after 1 seconds" in res.stderr
        mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_incus_agent_polling_timeout() -> None:
    """Test that agent polling raises IncusAgentTimeoutError if guest agent never responds."""
    cfg = IncusConfig(agent_wait_timeout_seconds=2, agent_poll_interval_seconds=0.2)
    sb = IncusSandbox(instance_name="dead-box", config=cfg)

    with patch.object(sb, "_run_incus_cli", return_value=(1, "", "agent not ready")):
        with pytest.raises(IncusAgentTimeoutError):
            await sb.wait_until_ready(timeout_seconds=1)


@pytest.mark.asyncio
async def test_incus_snapshot_and_revert_workflow() -> None:
    """Test snapshot creation and revert calls generate appropriate CLI args."""
    sb = IncusSandbox(instance_name="snap-box")

    cli_calls: list[list[str]] = []

    async def fake_cli(args: list[str], **kwargs: object) -> tuple[int, str, str]:
        cli_calls.append(args)
        return 0, "__READY__", ""

    with patch.object(sb, "_run_incus_cli", side_effect=fake_cli):
        await sb.create_snapshot("snap-1")
        assert "snap-1" in sb._snapshots
        assert cli_calls[-1] == ["snapshot", "create", "snap-box", "snap-1"]

        await sb.revert("snap-1")
        assert any(call == ["snapshot", "restore", "snap-box", "snap-1"] for call in cli_calls)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_incus_live_ephemeral_container_lifecycle() -> None:
    """End-to-end integration test creating a live sandbox, snapshotting, modifying file, and reverting."""
    cfg = IncusConfig(instance_prefix="test-ci", instance_type="container")
    sb = IncusSandbox(config=cfg, is_vm=False)

    try:
        # 1. Setup & Launch
        await sb.setup()
        assert sb._is_ready is True

        # 2. Command execution test
        whoami_res = await sb.execute("whoami")
        assert whoami_res.exit_code == 0
        assert "root" in whoami_res.stdout.strip()

        # Non-zero exit code capture
        fail_res = await sb.execute("ls /non_existent_path_12345")
        assert fail_res.exit_code != 0
        assert fail_res.stderr != ""

        # 3. Snapshot Creation
        await sb.create_snapshot("clean-state")

        # 4. Modify Guest State
        await sb.execute("echo 'CORRUPT_VALUE' > /etc/ci_test_file.txt")
        verify_mod = await sb.execute("cat /etc/ci_test_file.txt")
        assert "CORRUPT_VALUE" in verify_mod.stdout

        # 5. Snapshot Revert & State Assertion
        await sb.revert("clean-state")
        verify_revert = await sb.execute("cat /etc/ci_test_file.txt")
        assert verify_revert.exit_code != 0 or "CORRUPT_VALUE" not in verify_revert.stdout

    finally:
        # 6. Clean Deletion
        await sb.cleanup()
