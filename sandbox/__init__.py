"""Sandbox abstractions and Incus virtualization driver."""

from sandbox.base import BaseSandbox, ExecutionResult
from sandbox.drivers.firecracker_sandbox import FirecrackerSandbox
from sandbox.drivers.proxmox_sandbox import ProxmoxSandbox
from sandbox.factory import create_sandbox
from sandbox.incus_sandbox import (
    IncusAgentTimeoutError,
    IncusSandbox,
    IncusSandboxError,
)

__all__ = [
    "BaseSandbox",
    "ExecutionResult",
    "IncusSandbox",
    "IncusSandboxError",
    "IncusAgentTimeoutError",
    "FirecrackerSandbox",
    "ProxmoxSandbox",
    "create_sandbox",
]
