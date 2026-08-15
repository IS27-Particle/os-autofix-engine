"""Scenario registry and discovery mechanisms."""

from __future__ import annotations

from scenarios.base_scenario import BaseScenario
from scenarios.docker_socket import DockerSocketScenario
from scenarios.file_permissions import FilePermissionsScenario
from scenarios.iptables_lockout import IPTablesLockoutScenario
from scenarios.mac_enforcement import MacEnforcementScenario
from scenarios.network_routing import NetworkRoutingScenario
from scenarios.package_corruption import PackageCorruptionScenario
from scenarios.systemd_dns import SystemdDNSScenario
from scenarios.threat_hunt_persistence import ThreatHuntPersistenceScenario
from scenarios.zfs_mount import ZFSMountScenario

_REGISTRY: dict[str, type[BaseScenario]] = {
    SystemdDNSScenario.name: SystemdDNSScenario,
    NetworkRoutingScenario.name: NetworkRoutingScenario,
    PackageCorruptionScenario.name: PackageCorruptionScenario,
    FilePermissionsScenario.name: FilePermissionsScenario,
    ZFSMountScenario.name: ZFSMountScenario,
    DockerSocketScenario.name: DockerSocketScenario,
    IPTablesLockoutScenario.name: IPTablesLockoutScenario,
    MacEnforcementScenario.name: MacEnforcementScenario,
    ThreatHuntPersistenceScenario.name: ThreatHuntPersistenceScenario,
}


def register_scenario(scenario_cls: type[BaseScenario]) -> None:
    """Register a new troubleshooting scenario in the global registry."""
    _REGISTRY[scenario_cls.name] = scenario_cls


def get_scenario(name: str) -> BaseScenario:
    """Instantiate and return a scenario by its unique identifier."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Scenario '{name}' not found. Available scenarios: {available}")
    return _REGISTRY[name]()


def list_scenarios() -> list[str]:
    """Return a list of all registered scenario names."""
    return sorted(_REGISTRY.keys())


def get_all_scenarios() -> list[BaseScenario]:
    """Return instantiated instances of all registered scenarios."""
    return [cls() for cls in _REGISTRY.values()]
