"""Unit tests for CRIU Process State Preserver and Live Hotpatcher."""

from __future__ import annotations

import pytest

from engine.criu_state_preserver import CRIUStatePreserver
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_criu_dump_and_restore_workflow() -> None:
    """Test process checkpointing and live restoration via CRIU."""
    sandbox = MockSandbox("criu-sandbox-1")
    preserver = CRIUStatePreserver()

    pid = await preserver.find_daemon_pid(sandbox, "systemd-resolved")
    assert pid == 4242

    # Checkpoint
    chk_res = await preserver.checkpoint_process(sandbox, pid=pid, process_name="systemd-resolved")
    assert chk_res.dump_success is True
    assert chk_res.pid == 4242
    assert chk_res.checkpoint_id.startswith("chk-4242-")

    # Restore
    restore_ok = await preserver.restore_process(sandbox, chk_res)
    assert restore_ok is True
    assert chk_res.restore_success is True


@pytest.mark.asyncio
async def test_criu_hotpatch_success() -> None:
    """Test executing a live hotpatch with state preservation."""
    sandbox = MockSandbox("criu-hotpatch-sandbox")
    preserver = CRIUStatePreserver()

    res = await preserver.hotpatch_with_preservation(
        sandbox=sandbox,
        daemon_name="systemd-resolved",
        patch_command="echo 'nameserver 1.1.1.1' > /etc/resolv.conf",
        rollback_command="echo 'nameserver 127.0.0.99' > /etc/resolv.conf",
    )

    assert res.dump_success is True
    assert res.restore_success is True
    assert res.rolled_back is False
    assert res.duration_seconds >= 0.0


@pytest.mark.asyncio
async def test_criu_hotpatch_rollback_on_failure() -> None:
    """Test triggering automated rollback when patch mutation fails."""
    sandbox = MockSandbox("criu-rollback-sandbox")
    preserver = CRIUStatePreserver()

    # Pass an invalid command that exits non-zero (e.g., trying to cat non-existent file)
    res = await preserver.hotpatch_with_preservation(
        sandbox=sandbox,
        daemon_name="systemd-resolved",
        patch_command="cat /nonexistent/invalid/path/forcing/failure",
        rollback_command="echo 'ROLLBACK_TRIGGERED' > /tmp/rollback.log",
    )

    assert res.dump_success is True
    assert res.restore_success is False
    assert res.rolled_back is True
    assert "/tmp/rollback.log" in sandbox.files
