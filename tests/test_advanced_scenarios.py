"""Unit tests for advanced diagnostic scenarios: ZFS mount, Docker socket, and IPTables lockout."""

from __future__ import annotations

import pytest

from scenarios.docker_socket import DockerSocketScenario
from scenarios.iptables_lockout import IPTablesLockoutScenario
from scenarios.zfs_mount import ZFSMountScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_zfs_mount_scenario_lifecycle() -> None:
    """Test full setup, fault injection, and resolution verification for zfs_mount."""
    sandbox = MockSandbox("zfs-testbox")
    scenario = ZFSMountScenario()

    # 1. Setup baseline
    await scenario.setup(sandbox)
    resolved, msg = await scenario.verify(sandbox)
    assert resolved is True
    assert "verified intact" in msg.lower()

    # 2. Inject fault (unmount dataset)
    await scenario.inject_fault(sandbox)
    resolved_fault, msg_fault = await scenario.verify(sandbox)
    assert resolved_fault is False
    assert "not mounted" in msg_fault.lower() or "unreadable" in msg_fault.lower()

    # 3. Apply fix (re-mount filesystem)
    await sandbox.execute("mount -o loop /var/lib/storage_backing/disk.img /mnt/data")
    resolved_fixed, msg_fixed = await scenario.verify(sandbox)
    assert resolved_fixed is True


@pytest.mark.asyncio
async def test_docker_socket_scenario_lifecycle() -> None:
    """Test full setup, fault injection, and resolution verification for docker_socket."""
    sandbox = MockSandbox("docker-testbox")
    scenario = DockerSocketScenario()

    # 1. Setup baseline
    await scenario.setup(sandbox)
    resolved, msg = await scenario.verify(sandbox)
    assert resolved is True

    # 2. Inject 0000 permission lockout
    await scenario.inject_fault(sandbox)
    resolved_fault, msg_fault = await scenario.verify(sandbox)
    assert resolved_fault is False
    assert "not accessible" in msg_fault.lower()

    # 3. Apply fix (restore 0660 permissions)
    await sandbox.execute("chmod 0660 /var/run/docker.sock")
    resolved_fixed, msg_fixed = await scenario.verify(sandbox)
    assert resolved_fixed is True


@pytest.mark.asyncio
async def test_iptables_lockout_scenario_lifecycle() -> None:
    """Test full setup, fault injection, and resolution verification for iptables_lockout."""
    sandbox = MockSandbox("iptables-testbox")
    scenario = IPTablesLockoutScenario()

    # 1. Setup baseline
    await scenario.setup(sandbox)
    resolved, msg = await scenario.verify(sandbox)
    assert resolved is True

    # 2. Inject outbound DROP firewall rules
    await scenario.inject_fault(sandbox)
    resolved_fault, msg_fault = await scenario.verify(sandbox)
    assert resolved_fault is False
    assert "drop" in msg_fault.lower()

    # 3. Apply fix (flush output rules)
    await sandbox.execute("iptables -F OUTPUT")
    resolved_fixed, msg_fixed = await scenario.verify(sandbox)
    assert resolved_fixed is True
