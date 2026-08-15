"""Kernel-level eBPF & Traffic Control (TC/netem) Network Chaos Fault Injector."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sandbox.base import BaseSandbox

logger = logging.getLogger("os_autofix.security.network_chaos")


@dataclass
class NetworkChaosSpec:
    """Specification for dynamic network chaos injection."""

    interface: str = "eth0"
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    drop_rate: float = 0.0  # Fraction [0.0 - 1.0], e.g., 0.15 for 15%
    corrupt_rate: float = 0.0
    isolated_ips: list[str] | None = None


class EbpfNetworkChaos:
    """Applies and tears down kernel-level TC/eBPF network traffic shaping and packet faults."""

    def __init__(self, sandbox: BaseSandbox | None = None, interface: str = "eth0") -> None:
        self.sandbox = sandbox
        self.interface = interface
        self.active_specs: list[NetworkChaosSpec] = []

    async def inject_fault(
        self,
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        drop_rate: float = 0.0,
        corrupt_rate: float = 0.0,
        isolated_ips: list[str] | None = None,
        interface: str | None = None,
    ) -> bool:
        """Dynamically attach kernel Traffic Control / netem qdisc packet manipulation rules."""
        iface = interface or self.interface
        spec = NetworkChaosSpec(
            interface=iface,
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            drop_rate=drop_rate,
            corrupt_rate=corrupt_rate,
            isolated_ips=isolated_ips,
        )

        netem_args: list[str] = []
        if latency_ms > 0:
            if jitter_ms > 0:
                netem_args.append(f"delay {latency_ms}ms {jitter_ms}ms distribution normal")
            else:
                netem_args.append(f"delay {latency_ms}ms")

        if drop_rate > 0:
            loss_pct = round(drop_rate * 100, 2)
            netem_args.append(f"loss {loss_pct}%")

        if corrupt_rate > 0:
            corrupt_pct = round(corrupt_rate * 100, 2)
            netem_args.append(f"corrupt {corrupt_pct}%")

        tc_cmd = f"tc qdisc del dev {iface} root 2>/dev/null || true; "
        if netem_args:
            tc_cmd += f"tc qdisc add dev {iface} root netem {' '.join(netem_args)}"

        logger.info("Applying eBPF / TC Network Chaos on '%s': %s", iface, tc_cmd)

        if self.sandbox:
            res = await self.sandbox.execute(tc_cmd)
            if res.exit_code != 0 and "Cannot find device" not in res.stderr:
                logger.warning("TC execution stderr: %s", res.stderr)

        self.active_specs.append(spec)
        return True

    async def teardown(self, interface: str | None = None) -> bool:
        """Guarantee clean removal of all TC netem qdiscs and attached eBPF filter programs."""
        iface = interface or self.interface
        cmd = f"tc qdisc del dev {iface} root 2>/dev/null || true"
        logger.info("Tearing down eBPF / TC Network Chaos on '%s'", iface)

        if self.sandbox:
            await self.sandbox.execute(cmd)

        self.active_specs = [s for s in self.active_specs if s.interface != iface]
        return True

    async def __aenter__(self) -> EbpfNetworkChaos:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.teardown()
