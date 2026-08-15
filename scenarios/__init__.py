"""Scenarios package for OS diagnostic fault injection, resolution verification, and synthetic generation."""

from scenarios.base_scenario import BaseScenario
from scenarios.docker_socket import DockerSocketScenario
from scenarios.file_permissions import FilePermissionsScenario
from scenarios.iptables_lockout import IPTablesLockoutScenario
from scenarios.mac_enforcement import MacEnforcementScenario
from scenarios.network_routing import NetworkRoutingScenario
from scenarios.package_corruption import PackageCorruptionScenario
from scenarios.registry import (
    get_all_scenarios,
    get_scenario,
    list_scenarios,
    register_scenario,
)
from scenarios.synthesizer import ScenarioSynthesizer
from scenarios.systemd_dns import SystemdDNSScenario
from scenarios.threat_hunt_persistence import ThreatHuntPersistenceScenario
from scenarios.zfs_mount import ZFSMountScenario

__all__ = [
    "BaseScenario",
    "SystemdDNSScenario",
    "NetworkRoutingScenario",
    "PackageCorruptionScenario",
    "FilePermissionsScenario",
    "ZFSMountScenario",
    "DockerSocketScenario",
    "IPTablesLockoutScenario",
    "MacEnforcementScenario",
    "ThreatHuntPersistenceScenario",
    "ScenarioSynthesizer",
    "get_scenario",
    "list_scenarios",
    "get_all_scenarios",
    "register_scenario",
]
