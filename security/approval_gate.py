"""Human-in-the-Loop interactive webhook approval gate for high-blast-radius actions."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from monitoring.alerts import WebhookAlertDispatcher

logger = logging.getLogger("os_autofix.security.approval_gate")


class ApprovalStatus(str, Enum):
    """Lifecycle status of a human approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class ApprovalRequest:
    """An interactive action approval request pending human review."""

    action_id: str
    command: str
    safety_score: float
    blast_radius: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    decision_reason: str = ""
    reviewer: str = ""


@dataclass
class ApprovalDecision:
    """Decision returned by the approval gate."""

    action_id: str
    status: ApprovalStatus
    approved: bool
    reason: str


class InteractiveApprovalGate:
    """Gatekeeper intercepting high-blast-radius actions for human approval via Webhooks."""

    def __init__(
        self,
        webhook_dispatcher: WebhookAlertDispatcher | None = None,
        callback_base_url: str = "http://localhost:8000/api/v1",
        default_timeout_seconds: float = 60.0,
    ) -> None:
        self.dispatcher = webhook_dispatcher or WebhookAlertDispatcher()
        self.callback_base_url = callback_base_url.rstrip("/")
        self.default_timeout_seconds = default_timeout_seconds
        self.pending_requests: dict[str, ApprovalRequest] = {}
        self._events: dict[str, asyncio.Event] = {}

    def is_approval_required(self, safety_score: float) -> bool:
        """Check if action falls in the intermediate review window [0.70, 0.85]."""
        return 0.70 <= safety_score < 0.85

    def create_request(
        self,
        command: str,
        safety_score: float,
        blast_radius: str,
    ) -> ApprovalRequest:
        """Register a new action approval request."""
        action_id = f"act-{uuid.uuid4().hex[:8]}"
        req = ApprovalRequest(
            action_id=action_id,
            command=command,
            safety_score=safety_score,
            blast_radius=blast_radius,
        )
        self.pending_requests[action_id] = req
        self._events[action_id] = asyncio.Event()
        return req

    def format_discord_payload(self, req: ApprovalRequest) -> dict[str, Any]:
        """Format an interactive Discord webhook notification with review links."""
        approve_url = f"{self.callback_base_url}/approve/{req.action_id}"
        reject_url = f"{self.callback_base_url}/reject/{req.action_id}"

        return {
            "embeds": [
                {
                    "title": "⚠️ [OS-AutoFix] Human-in-the-Loop Approval Required",
                    "description": (
                        f"An agent has proposed a high-blast-radius action requiring human authorization.\n\n"
                        f"**Proposed Command:**\n```bash\n{req.command}\n```\n"
                        f"**Safety Score:** `{req.safety_score:.2f}` (Review range: 0.70 - 0.85)\n"
                        f"**Blast Radius:** `{req.blast_radius.upper()}`\n\n"
                        f"**Action ID:** `{req.action_id}`\n"
                        f"👉 [Approve Action]({approve_url}) | 🛑 [Reject Action]({reject_url})"
                    ),
                    "color": 15105570,  # Orange/Amber
                    "footer": {"text": "OS-AutoFix Autonomous Security Gate"},
                }
            ]
        }

    async def submit_and_wait(
        self,
        command: str,
        safety_score: float,
        blast_radius: str,
        timeout_seconds: float | None = None,
        request: ApprovalRequest | None = None,
    ) -> ApprovalDecision:
        """Submit approval request, dispatch webhook notification, and await human decision."""
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        req = request or self.create_request(command, safety_score, blast_radius)

        logger.info(
            "Approval Gate: Action '%s' requires approval (Score=%.2f, Blast=%s). Timeout: %.1fs",
            req.action_id,
            safety_score,
            blast_radius,
            timeout,
        )

        # Dispatch webhook alert
        if self.dispatcher.webhook_url:
            import httpx

            payload = self.format_discord_payload(req)
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(self.dispatcher.webhook_url, json=payload)
            except Exception as e:
                logger.warning("Approval Gate: Webhook dispatch error: %s", e)

        event = self._events[req.action_id]

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            req.status = ApprovalStatus.TIMED_OUT
            req.decision_reason = (
                f"Approval request timed out after {timeout} seconds. Auto-rejected."
            )
            logger.warning("Approval Gate: Action '%s' timed out. Rolling back.", req.action_id)

        approved = req.status == ApprovalStatus.APPROVED
        return ApprovalDecision(
            action_id=req.action_id,
            status=req.status,
            approved=approved,
            reason=req.decision_reason
            or f"Action {req.status.value.lower()} by {req.reviewer or 'system'}.",
        )

    def approve(self, action_id: str, reviewer: str = "human_operator") -> bool:
        """Record human approval for a pending action ID."""
        if action_id not in self.pending_requests:
            return False
        req = self.pending_requests[action_id]
        if req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.APPROVED
        req.reviewer = reviewer
        req.decision_reason = f"Explicitly approved by {reviewer}."
        if action_id in self._events:
            self._events[action_id].set()
        logger.info("Approval Gate: Action '%s' APPROVED by '%s'", action_id, reviewer)
        return True

    def reject(self, action_id: str, reviewer: str = "human_operator", reason: str = "") -> bool:
        """Record human rejection for a pending action ID."""
        if action_id not in self.pending_requests:
            return False
        req = self.pending_requests[action_id]
        if req.status != ApprovalStatus.PENDING:
            return False

        req.status = ApprovalStatus.REJECTED
        req.reviewer = reviewer
        req.decision_reason = reason or f"Explicitly rejected by {reviewer}."
        if action_id in self._events:
            self._events[action_id].set()
        logger.info("Approval Gate: Action '%s' REJECTED by '%s': %s", action_id, reviewer, reason)
        return True


GLOBAL_APPROVAL_GATE = InteractiveApprovalGate()
