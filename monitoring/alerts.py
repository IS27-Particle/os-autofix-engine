"""Asynchronous Webhook & Alert Dispatcher for model promotions, regressions, and worker failures."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Literal

import httpx

logger = logging.getLogger("os_autofix.monitoring.alerts")

AlertSeverity = Literal["info", "warning", "critical"]


@dataclasses.dataclass
class AlertPayload:
    """Standardized alert payload model."""

    title: str
    message: str
    severity: AlertSeverity = "info"
    event_type: str = "generic"
    model_tag: str = "unknown"
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    timestamp: float = dataclasses.field(default_factory=time.time)


def format_discord_payload(alert: AlertPayload) -> dict[str, Any]:
    """Format alert as Discord webhook embed."""
    color_map = {
        "info": 0x3498DB,  # Blue
        "warning": 0xF39C12,  # Orange
        "critical": 0xE74C3C,  # Red
    }
    fields = [
        {"name": "Event Type", "value": alert.event_type, "inline": True},
        {"name": "Model Tag", "value": alert.model_tag, "inline": True},
    ]
    for k, v in alert.metadata.items():
        fields.append({"name": str(k), "value": str(v), "inline": True})

    return {
        "content": f"**[OS-AutoFix Engine Alert]** - `{alert.severity.upper()}`",
        "embeds": [
            {
                "title": alert.title,
                "description": alert.message,
                "color": color_map.get(alert.severity, 0x95A5A6),
                "fields": fields,
                "footer": {
                    "text": f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}"
                },
            }
        ],
    }


def format_slack_payload(alert: AlertPayload) -> dict[str, Any]:
    """Format alert as Slack incoming webhook blocks."""
    color_map = {
        "info": "#3498db",
        "warning": "#f39c12",
        "critical": "#e74c3c",
    }
    return {
        "text": f"*{alert.title}* ({alert.severity.upper()}): {alert.message}",
        "attachments": [
            {
                "color": color_map.get(alert.severity, "#95a5a6"),
                "fields": [
                    {"title": "Event Type", "value": alert.event_type, "short": True},
                    {"title": "Model Tag", "value": alert.model_tag, "short": True},
                    *[
                        {"title": str(k), "value": str(v), "short": True}
                        for k, v in alert.metadata.items()
                    ],
                ],
            }
        ],
    }


def format_generic_payload(alert: AlertPayload) -> dict[str, Any]:
    """Format alert as standard structured JSON."""
    return {
        "title": alert.title,
        "message": alert.message,
        "severity": alert.severity,
        "event_type": alert.event_type,
        "model_tag": alert.model_tag,
        "metadata": alert.metadata,
        "timestamp": alert.timestamp,
    }


class WebhookAlertDispatcher:
    """Asynchronous dispatcher sending webhook notifications to Discord, Slack, or generic endpoints."""

    def __init__(self, webhook_url: str | None = None, timeout_seconds: float = 10.0) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout_seconds

    async def dispatch(self, alert: AlertPayload) -> bool:
        """Send formatted alert payload to configured webhook destination."""
        if not self.webhook_url:
            logger.debug("No webhook URL configured; skipping alert dispatch for: %s", alert.title)
            return False

        # Detect platform based on URL
        url_lower = self.webhook_url.lower()
        if "discord.com" in url_lower:
            payload = format_discord_payload(alert)
        elif "slack.com" in url_lower:
            payload = format_slack_payload(alert)
        else:
            payload = format_generic_payload(alert)

        logger.info(
            "Dispatching %s alert '%s' to webhook endpoint...", alert.severity.upper(), alert.title
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.webhook_url, json=payload)
                if resp.status_code in (200, 204):
                    logger.info(
                        "Alert '%s' successfully delivered (status %d)",
                        alert.title,
                        resp.status_code,
                    )
                    return True
                logger.warning(
                    "Webhook endpoint returned non-success status %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        except Exception as e:
            logger.error("Failed to dispatch alert to %s: %s", self.webhook_url, e)
            return False

    async def dispatch_model_promoted(
        self,
        model_tag: str,
        eval_pass_rate: float,
        baseline_pass_rate: float,
        delta: float,
        iteration: int,
    ) -> bool:
        """Notify on successful training and promotion of a new policy generation."""
        alert = AlertPayload(
            title=f"Policy Model Promoted: {model_tag}",
            message=f"Generation {model_tag} achieved {eval_pass_rate * 100:.1f}% pass rate (Delta: {delta * 100:+.1f}%) in Iteration #{iteration} and has been promoted as active.",
            severity="info",
            event_type="model_promoted",
            model_tag=model_tag,
            metadata={
                "Iteration": iteration,
                "New Pass Rate": f"{eval_pass_rate * 100:.1f}%",
                "Baseline Rate": f"{baseline_pass_rate * 100:.1f}%",
                "Delta": f"{delta * 100:+.1f}%",
            },
        )
        return await self.dispatch(alert)

    async def dispatch_model_regression(
        self,
        model_tag: str,
        rolled_back_to: str,
        eval_pass_rate: float,
        baseline_pass_rate: float,
        delta: float,
        iteration: int,
    ) -> bool:
        """Notify on model pass rate regression and automatic fallback trigger."""
        alert = AlertPayload(
            title=f"ALERT: Model Regression Detected: {model_tag}",
            message=f"Generation {model_tag} regressed to {eval_pass_rate * 100:.1f}% pass rate (Delta: {delta * 100:+.1f}%). Automatically rolling back active policy to '{rolled_back_to}'.",
            severity="critical",
            event_type="model_regression",
            model_tag=model_tag,
            metadata={
                "Iteration": iteration,
                "Failed Tag": model_tag,
                "Active Rollback Tag": rolled_back_to,
                "Evaluated Pass Rate": f"{eval_pass_rate * 100:.1f}%",
                "Baseline Rate": f"{baseline_pass_rate * 100:.1f}%",
                "Delta": f"{delta * 100:+.1f}%",
            },
        )
        return await self.dispatch(alert)

    async def dispatch_worker_failure(
        self,
        instance_id: str,
        scenario: str,
        error_message: str,
    ) -> bool:
        """Notify on Incus agent handshake timeout or persistent sandbox crash."""
        alert = AlertPayload(
            title=f"Worker Sandbox Failure: {scenario}",
            message=f"Instance '{instance_id}' encountered a fatal error during scenario '{scenario}': {error_message}",
            severity="warning",
            event_type="worker_failure",
            metadata={
                "Instance ID": instance_id,
                "Scenario": scenario,
                "Error": error_message[:200],
            },
        )
        return await self.dispatch(alert)
