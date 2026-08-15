"""Distributed multi-node scenario package."""

from scenarios.distributed.base_distributed import BaseDistributedScenario
from scenarios.distributed.etcd_split_brain import EtcdSplitBrainScenario
from scenarios.distributed.reverse_proxy_ha import ReverseProxyHAScenario
from scenarios.distributed.wireguard_mesh import WireGuardMeshScenario

DISTRIBUTED_SCENARIOS: dict[str, type[BaseDistributedScenario]] = {
    "wireguard_mesh": WireGuardMeshScenario,
    "etcd_split_brain": EtcdSplitBrainScenario,
    "reverse_proxy_ha": ReverseProxyHAScenario,
}


def get_distributed_scenario(name: str) -> BaseDistributedScenario:
    """Retrieve an instantiated distributed scenario by name."""
    if name not in DISTRIBUTED_SCENARIOS:
        raise KeyError(
            f"Unknown distributed scenario '{name}'. Available: {list(DISTRIBUTED_SCENARIOS.keys())}"
        )
    return DISTRIBUTED_SCENARIOS[name]()


def get_all_distributed_scenarios() -> list[BaseDistributedScenario]:
    """Retrieve instances of all registered distributed scenarios."""
    return [cls() for cls in DISTRIBUTED_SCENARIOS.values()]


__all__ = [
    "BaseDistributedScenario",
    "WireGuardMeshScenario",
    "EtcdSplitBrainScenario",
    "ReverseProxyHAScenario",
    "DISTRIBUTED_SCENARIOS",
    "get_distributed_scenario",
    "get_all_distributed_scenarios",
]
