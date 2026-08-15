"""Unit tests for Firecracker MicroVM driver and factory integration."""

from __future__ import annotations

import pytest

from sandbox.drivers.firecracker_sandbox import FirecrackerSandbox
from sandbox.drivers.proxmox_sandbox import ProxmoxSandbox
from sandbox.factory import create_sandbox


@pytest.mark.asyncio
async def test_firecracker_sandbox_lifecycle() -> None:
    """Test setup, file operations, snapshots, and cleanup in Firecracker microVM."""
    sb = FirecrackerSandbox(instance_name="test-fc-1", mock_mode=True)

    await sb.setup()
    assert sb._running is True

    # Test file write and read
    await sb.write_file("/etc/test.conf", "hello=world\n")
    read_back = await sb.read_file("/etc/test.conf")
    assert "hello=world" in read_back

    # Test snapshot & rollback
    await sb.create_snapshot("snap1")
    await sb.write_file("/etc/test.conf", "mutated=true\n")
    assert "mutated=true" in await sb.read_file("/etc/test.conf")

    await sb.revert("snap1")
    assert "hello=world" in await sb.read_file("/etc/test.conf")

    state = await sb.get_state()
    assert state["type"] == "firecracker_microvm"
    assert state["running"] is True

    await sb.cleanup()
    assert sb._running is False


@pytest.mark.asyncio
async def test_proxmox_sandbox_lifecycle() -> None:
    """Test Proxmox VE sandbox mock lifecycle."""
    sb = ProxmoxSandbox(instance_name="test-pve-1", mock_mode=True)

    await sb.setup()
    assert sb._running is True

    await sb.write_file("/etc/pve_test.conf", "cluster=true\n")
    assert "cluster=true" in await sb.read_file("/etc/pve_test.conf")

    await sb.create_snapshot("pve_snap")
    await sb.write_file("/etc/pve_test.conf", "corrupted\n")
    await sb.revert("pve_snap")
    assert "cluster=true" in await sb.read_file("/etc/pve_test.conf")

    await sb.cleanup()


def test_sandbox_factory() -> None:
    """Test creating different sandbox drivers via unified factory."""
    fc = create_sandbox("firecracker", "fc-inst", mock_mode=True)
    assert isinstance(fc, FirecrackerSandbox)

    pve = create_sandbox("proxmox", "pve-inst", mock_mode=True)
    assert isinstance(pve, ProxmoxSandbox)

    from tests.conftest import MockSandbox

    mock_sb = create_sandbox("mock", "mock-inst")
    assert isinstance(mock_sb, MockSandbox)
    assert mock_sb.name == "mock-inst"
