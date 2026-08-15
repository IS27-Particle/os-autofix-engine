"""Unit tests for the SyscallSecurityAuditor and eBPF security analyzer."""

from __future__ import annotations

import pytest

from security.ebpf_auditor import SyscallSecurityAuditor
from tests.conftest import MockSandbox


def test_destructive_command_detection() -> None:
    """Test flagging destructive shell commands like recursive root wipe, reverse shell, forkbomb."""
    auditor = SyscallSecurityAuditor(safety_threshold=0.7)

    # 1. Critical destructive root wipe
    rep1 = auditor.inspect_command("rm -rf /")
    assert rep1.is_safe is False
    assert rep1.abort_execution is True
    assert rep1.safety_score <= 0.3
    assert rep1.blast_radius in ("system", "kernel")

    # 2. Reverse shell backdoor
    rep2 = auditor.inspect_command("nc -lvnp 4444 -e /bin/bash")
    assert rep2.is_safe is False
    assert rep2.abort_execution is True

    # 3. Credential harvesting
    rep3 = auditor.inspect_command("cat /etc/shadow")
    assert rep3.is_safe is False
    assert any("shadow" in e.description for e in rep3.events)

    # 4. Safe command
    rep4 = auditor.inspect_command("systemctl restart systemd-resolved")
    assert rep4.is_safe is True
    assert rep4.abort_execution is False
    assert rep4.safety_score == 1.0


@pytest.mark.asyncio
async def test_sandbox_runtime_taint_check() -> None:
    """Test dynamic runtime security check on sandbox kernel status."""
    auditor = SyscallSecurityAuditor(safety_threshold=0.7)
    sandbox = MockSandbox("sec-test")

    # Untainted kernel
    rep = await auditor.audit_sandbox_runtime(sandbox, "ip route show")
    assert rep.is_safe is True
