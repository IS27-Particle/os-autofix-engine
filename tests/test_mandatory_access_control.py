"""Unit tests for MAC Profile Synthesizer and MacEnforcementScenario."""

from __future__ import annotations

import pytest

from scenarios.mac_enforcement import MacEnforcementScenario
from security.mandatory_access_control import MacProfileSynthesizer
from tests.conftest import MockSandbox


def test_apparmor_profile_synthesis() -> None:
    """Test generating least-privilege AppArmor profile with capability and path rules."""
    synth = MacProfileSynthesizer()
    profile = synth.synthesize_apparmor(
        binary_path="/usr/sbin/custom_daemon",
        profile_name="custom_daemon",
        audit_logs=[
            'apparmor="DENIED" operation="open" profile="custom_daemon" name="/var/data/custom.db"',
        ],
    )

    rendered = profile.render()
    assert "profile custom_daemon /usr/sbin/custom_daemon" in rendered
    assert "capability net_bind_service," in rendered
    assert "/etc/resolv.conf r," in rendered
    assert "/var/data/custom.db rw," in rendered


def test_selinux_module_synthesis() -> None:
    """Test generating SELinux Type Enforcement (.te) policy module."""
    synth = MacProfileSynthesizer()
    te_out = synth.synthesize_selinux_te("my_custom_daemon", "custom_daemon")

    assert "module my_custom_daemon 1.0;" in te_out
    assert "type custom_daemon_t;" in te_out
    assert "allow custom_daemon_t etc_t:file" in te_out


@pytest.mark.asyncio
async def test_mac_enforcement_scenario_lifecycle() -> None:
    """Test setup, fault injection, and verification logic of mac_enforcement scenario."""
    sandbox = MockSandbox("mac-test-sandbox")
    scenario = MacEnforcementScenario()

    # 1. Setup & Pre-flight
    await scenario.setup(sandbox)
    pre_res, _ = await scenario.verify(sandbox)
    assert pre_res is True

    # 2. Inject Breakage
    await scenario.inject_fault(sandbox)
    post_fault_res, _ = await scenario.verify(sandbox)
    assert post_fault_res is False

    # 3. Simulate correct remediation (fix policy rule without disabling MAC)
    await sandbox.execute(
        "sed -i 's/deny \\/etc/\\/etc/' /etc/apparmor.d/opt.custom_daemon.daemon.sh"
    )
    recovered_res, msg = await scenario.verify(sandbox)
    assert recovered_res is True
    assert "operating under confinement" in msg


@pytest.mark.asyncio
async def test_mac_enforcement_cheating_detection() -> None:
    """Assert verifier fails if operator globally disables MAC enforcement."""
    sandbox = MockSandbox("mac-cheat-sandbox")
    scenario = MacEnforcementScenario()

    await scenario.setup(sandbox)
    await scenario.inject_fault(sandbox)

    # Simulate disabling MAC globally (cheating)
    await sandbox.execute("echo 'disabled' >> /tmp/apparmor_enforcing_state")
    cheat_res, cheat_msg = await scenario.verify(sandbox)

    assert cheat_res is False
    assert "Security violation: Mandatory Access Control was globally disabled." in cheat_msg
