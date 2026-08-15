"""Unit tests for Kernel-level eBPF & TC Network Chaos Injector."""

from __future__ import annotations

import pytest

from security.ebpf_network_chaos import EbpfNetworkChaos
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_network_chaos_fault_injection_and_teardown() -> None:
    """Test applying and tearing down TC netem latency and packet drop rules."""
    sandbox = MockSandbox("chaos-net-test")
    chaos = EbpfNetworkChaos(sandbox=sandbox, interface="eth0")

    # 1. Inject latency & loss
    ok = await chaos.inject_fault(latency_ms=75.0, jitter_ms=5.0, drop_rate=0.10)
    assert ok is True
    assert len(chaos.active_specs) == 1
    assert chaos.active_specs[0].latency_ms == 75.0
    assert chaos.active_specs[0].drop_rate == 0.10

    # 2. Teardown
    ok_td = await chaos.teardown()
    assert ok_td is True
    assert len(chaos.active_specs) == 0


@pytest.mark.asyncio
async def test_network_chaos_context_manager() -> None:
    """Test asynchronous context manager lifecycle."""
    sandbox = MockSandbox("chaos-ctx-test")

    async with EbpfNetworkChaos(sandbox=sandbox, interface="wg0") as chaos:
        await chaos.inject_fault(drop_rate=0.25)
        assert len(chaos.active_specs) == 1

    # Verified cleaned up on exit
    assert len(chaos.active_specs) == 0
