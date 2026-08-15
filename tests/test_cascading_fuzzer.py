"""Unit tests for the Combinatorial Cascading Fault Fuzzer."""

from __future__ import annotations

import pytest

from config.settings import EngineConfig
from engine.cascading_fuzzer import CascadingFaultFuzzer
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_cascading_fault_injection_and_verification() -> None:
    """Test compound multi-domain fault injection and state verification."""
    sandbox = MockSandbox("fuzz-test-sb")
    fuzzer = CascadingFaultFuzzer()

    domains = ["network", "storage", "permissions"]

    # 1. Injected faults
    injected = await fuzzer.inject_compound_faults(sandbox, domains)
    assert len(injected) == 3
    assert "network" in injected
    assert "storage" in injected
    assert "permissions" in injected

    # 2. State verification before fix -> broken
    is_ok_pre, statuses_pre = await fuzzer.verify_compound_state(sandbox, domains)
    assert is_ok_pre is False
    assert statuses_pre["permissions"] is False

    # 3. Simulate repair
    await sandbox.execute("echo 'nameserver 8.8.8.8' > /etc/resolv.conf")
    await sandbox.execute("chmod 755 /mnt/data_pool /etc/hosts /tmp")

    # 4. State verification after fix -> healthy
    is_ok_post, statuses_post = await fuzzer.verify_compound_state(sandbox, domains)
    assert is_ok_post is True
    assert all(statuses_post.values())


@pytest.mark.asyncio
async def test_cascading_fuzzer_experiment_lifecycle() -> None:
    """Test running full fuzzing experiment with mock swarm."""
    cfg = EngineConfig()
    cfg.llm.mock_mode = True

    def mock_factory(name: str) -> MockSandbox:
        return MockSandbox(name)

    fuzzer = CascadingFaultFuzzer(config=cfg, sandbox_factory=mock_factory)
    res = await fuzzer.run_fuzzing_experiment(
        domains=["network", "permissions"],
        sandbox_name="test-fuzz-canary",
    )

    assert res.fuzz_id.startswith("fuzz-")
    assert len(res.domains_injected) == 2
    assert res.mttr_seconds >= 0.0
