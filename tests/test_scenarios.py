"""Unit tests validating scenario fault injection and verification mechanics."""

from __future__ import annotations

import pytest

from scenarios.file_permissions import FilePermissionsScenario
from scenarios.network_routing import NetworkRoutingScenario
from scenarios.package_corruption import PackageCorruptionScenario
from scenarios.registry import get_all_scenarios, get_scenario, list_scenarios
from scenarios.systemd_dns import SystemdDNSScenario
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_systemd_dns_scenario_lifecycle(mock_sandbox: MockSandbox) -> None:
    """Test DNS scenario setup, fault injection detection, and resolution verification."""
    scenario = SystemdDNSScenario()
    await mock_sandbox.setup()

    # 1. Setup
    setup_ok = await scenario.setup(mock_sandbox)
    assert setup_ok is True

    # 2. Baseline verification (should be healthy)
    init_pass, _ = await scenario.verify(mock_sandbox)
    assert init_pass is True

    # 3. Inject fault
    inject_ok = await scenario.inject_fault(mock_sandbox)
    assert inject_ok is True

    # 4. Verify fault is detected
    fault_pass, fault_msg = await scenario.verify(mock_sandbox)
    assert fault_pass is False
    assert "DNS" in fault_msg or "failed" in fault_msg

    # 5. Apply fix
    await mock_sandbox.execute("systemctl restart systemd-resolved")

    # 6. Verify resolution
    resolved_pass, _ = await scenario.verify(mock_sandbox)
    assert resolved_pass is True


@pytest.mark.asyncio
async def test_network_routing_scenario_lifecycle(mock_sandbox: MockSandbox) -> None:
    """Test Network routing scenario fault injection and recovery."""
    scenario = NetworkRoutingScenario()
    await mock_sandbox.setup()

    await scenario.setup(mock_sandbox)
    init_pass, _ = await scenario.verify(mock_sandbox)
    assert init_pass is True

    # Inject fault
    await scenario.inject_fault(mock_sandbox)
    fault_pass, _ = await scenario.verify(mock_sandbox)
    assert fault_pass is False

    # Apply fix
    await mock_sandbox.execute("ip route replace default via 10.0.0.1 dev eth0")
    resolved_pass, _ = await scenario.verify(mock_sandbox)
    assert resolved_pass is True


@pytest.mark.asyncio
async def test_package_corruption_scenario_lifecycle(mock_sandbox: MockSandbox) -> None:
    """Test Package corruption scenario lock detection and cleanup verification."""
    scenario = PackageCorruptionScenario()
    await mock_sandbox.setup()

    await scenario.setup(mock_sandbox)
    init_pass, _ = await scenario.verify(mock_sandbox)
    assert init_pass is True

    # Inject fault
    await scenario.inject_fault(mock_sandbox)
    fault_pass, _ = await scenario.verify(mock_sandbox)
    assert fault_pass is False

    # Apply fix
    await mock_sandbox.execute("rm -f /var/lib/dpkg/lock-frontend && dpkg --configure -a")
    resolved_pass, _ = await scenario.verify(mock_sandbox)
    assert resolved_pass is True


@pytest.mark.asyncio
async def test_file_permissions_scenario_lifecycle(mock_sandbox: MockSandbox) -> None:
    """Test Sudoers/SSH permission lockout and restoration."""
    scenario = FilePermissionsScenario()
    await mock_sandbox.setup()

    await scenario.setup(mock_sandbox)
    init_pass, _ = await scenario.verify(mock_sandbox)
    assert init_pass is True

    # Inject fault (0777 sudoers)
    await scenario.inject_fault(mock_sandbox)
    fault_pass, _ = await scenario.verify(mock_sandbox)
    assert fault_pass is False

    # Apply fix (0440 sudoers)
    await mock_sandbox.execute("chmod 0440 /etc/sudoers")
    resolved_pass, _ = await scenario.verify(mock_sandbox)
    assert resolved_pass is True


def test_scenario_registry() -> None:
    """Test scenario registry discovery and retrieval."""
    scenarios_list = list_scenarios()
    assert "systemd_dns" in scenarios_list
    assert "network_routing" in scenarios_list
    assert "package_corruption" in scenarios_list
    assert "file_permissions" in scenarios_list

    dns_sc = get_scenario("systemd_dns")
    assert isinstance(dns_sc, SystemdDNSScenario)

    all_sc = get_all_scenarios()
    assert len(all_sc) >= 4

    with pytest.raises(KeyError):
        get_scenario("non_existent_scenario_name")
