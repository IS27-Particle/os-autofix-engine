"""Unified Sandbox Factory for Incus, Firecracker, Proxmox VE, and Mock runtimes."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import EngineConfig
from sandbox.base import BaseSandbox
from sandbox.drivers.firecracker_sandbox import FirecrackerSandbox
from sandbox.drivers.proxmox_sandbox import ProxmoxSandbox
from sandbox.incus_sandbox import IncusSandbox

logger = logging.getLogger("os_autofix.sandbox.factory")


def create_sandbox(
    driver_type: str = "incus",
    instance_name: str = "canary-instance",
    config: EngineConfig | None = None,
    **kwargs: Any,
) -> BaseSandbox:
    """Instantiate a sandbox instance for the specified hypervisor driver.

    Args:
        driver_type: One of 'incus', 'firecracker', 'proxmox', 'mock'.
        instance_name: Unique identifier for the instance.
        config: Optional global engine configuration.
        **kwargs: Driver-specific overrides.

    Returns:
        BaseSandbox: Concrete sandbox implementation.
    """
    driver_norm = driver_type.lower().strip()

    if driver_norm == "firecracker":
        return FirecrackerSandbox(
            instance_name=instance_name,
            kernel_image_path=kwargs.get("kernel_image_path", "/var/lib/firecracker/vmlinux-6.1"),
            rootfs_base_path=kwargs.get(
                "rootfs_base_path", "/var/lib/firecracker/ubuntu-24.04.ext4"
            ),
            vcpus=kwargs.get("vcpus", 2),
            mem_size_mib=kwargs.get("mem_size_mib", 1024),
            mock_mode=kwargs.get("mock_mode", False),
        )

    if driver_norm == "proxmox":
        return ProxmoxSandbox(
            instance_name=instance_name,
            host=kwargs.get("host", "https://proxmox.local:8006"),
            node=kwargs.get("node", "pve"),
            vmid=kwargs.get("vmid", 100),
            api_token=kwargs.get("api_token"),
            mock_mode=kwargs.get("mock_mode", False),
        )

    if driver_norm == "mock":
        from tests.conftest import MockSandbox

        return MockSandbox(name=instance_name)

    # Default: Incus
    incus_cfg = config.incus if config else None
    return IncusSandbox(
        instance_name=instance_name,
        config=incus_cfg,
        is_vm=kwargs.get("is_vm", False),
    )
