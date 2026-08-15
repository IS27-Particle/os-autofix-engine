"""Unit tests for the Human-in-the-Loop Interactive Approval Gate."""

from __future__ import annotations

import asyncio

import pytest

from security.approval_gate import ApprovalStatus, InteractiveApprovalGate


def test_approval_threshold_window() -> None:
    """Test approval requirement condition: [0.70, 0.85]."""
    gate = InteractiveApprovalGate()

    assert gate.is_approval_required(0.75) is True
    assert gate.is_approval_required(0.80) is True
    assert gate.is_approval_required(0.69) is False  # Below 0.70 -> Immediate Abort
    assert gate.is_approval_required(0.95) is False  # Above 0.85 -> Auto-Approved


def test_discord_payload_formatting() -> None:
    """Test generating interactive Discord webhook payload with approval URLs."""
    gate = InteractiveApprovalGate(callback_base_url="https://api.autofix.internal/v1")
    req = gate.create_request(
        "systemctl restart networking", safety_score=0.78, blast_radius="local"
    )

    payload = gate.format_discord_payload(req)
    assert "embeds" in payload
    embed = payload["embeds"][0]
    assert "Human-in-the-Loop Approval Required" in embed["title"]
    assert req.action_id in embed["description"]
    assert "https://api.autofix.internal/v1/approve/" in embed["description"]


@pytest.mark.asyncio
async def test_interactive_approval_flow() -> None:
    """Test asynchronous approval decision flow."""
    gate = InteractiveApprovalGate(default_timeout_seconds=5.0)

    async def _approver_coroutine(action_id: str) -> None:
        await asyncio.sleep(0.05)
        gate.approve(action_id, reviewer="sre_oncall")

    req = gate.create_request("iptables -F", safety_score=0.75, blast_radius="system")

    # Launch background approval
    asyncio.create_task(_approver_coroutine(req.action_id))

    decision = await gate.submit_and_wait(
        "iptables -F", 0.75, "system", timeout_seconds=2.0, request=req
    )
    assert decision.approved is True
    assert decision.status == ApprovalStatus.APPROVED
    assert "sre_oncall" in decision.reason


@pytest.mark.asyncio
async def test_interactive_rejection_flow() -> None:
    """Test explicit operator rejection."""
    gate = InteractiveApprovalGate(default_timeout_seconds=5.0)

    async def _rejector_coroutine(action_id: str) -> None:
        await asyncio.sleep(0.05)
        gate.reject(action_id, reviewer="security_lead", reason="Violates change freeze")

    req = gate.create_request("reboot", safety_score=0.72, blast_radius="kernel")
    asyncio.create_task(_rejector_coroutine(req.action_id))

    decision = await gate.submit_and_wait(
        "reboot", 0.72, "kernel", timeout_seconds=2.0, request=req
    )
    assert decision.approved is False
    assert decision.status == ApprovalStatus.REJECTED
    assert "Violates change freeze" in decision.reason


@pytest.mark.asyncio
async def test_approval_timeout_flow() -> None:
    """Test timeout fallback when human operator does not respond."""
    gate = InteractiveApprovalGate(default_timeout_seconds=0.1)
    decision = await gate.submit_and_wait("chmod 777 /var/run", 0.72, "system", timeout_seconds=0.1)

    assert decision.approved is False
    assert decision.status == ApprovalStatus.TIMED_OUT
