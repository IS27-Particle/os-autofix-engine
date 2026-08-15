"""Unit tests for asynchronous alert dispatcher and Discord/Slack webhook payload formatters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from monitoring.alerts import (
    AlertPayload,
    WebhookAlertDispatcher,
    format_discord_payload,
    format_generic_payload,
    format_slack_payload,
)


def test_format_discord_payload() -> None:
    """Test Discord embed construction and color mapping."""
    alert = AlertPayload(
        title="Model Promoted",
        message="Model os-fixer:v2 reached 85% pass rate",
        severity="info",
        event_type="model_promoted",
        model_tag="os-fixer:v2",
        metadata={"Accuracy": "85.0%"},
    )
    payload = format_discord_payload(alert)

    assert "embeds" in payload
    embed = payload["embeds"][0]
    assert embed["title"] == "Model Promoted"
    assert embed["description"] == "Model os-fixer:v2 reached 85% pass rate"
    assert embed["color"] == 0x3498DB  # Info blue
    assert any(f["name"] == "Model Tag" for f in embed["fields"])


def test_format_slack_payload() -> None:
    """Test Slack webhook block formatting."""
    alert = AlertPayload(
        title="Model Regression",
        message="Model regressed below baseline",
        severity="critical",
        event_type="model_regression",
        model_tag="os-fixer:v3",
    )
    payload = format_slack_payload(alert)

    assert "attachments" in payload
    assert payload["attachments"][0]["color"] == "#e74c3c"  # Red
    assert "*Model Regression*" in payload["text"]


def test_format_generic_payload() -> None:
    """Test standard generic JSON alert formatting."""
    alert = AlertPayload(
        title="Worker Failure",
        message="Agent timeout",
        severity="warning",
    )
    payload = format_generic_payload(alert)
    assert payload["title"] == "Worker Failure"
    assert payload["severity"] == "warning"


@pytest.mark.asyncio
async def test_webhook_dispatcher_empty_url() -> None:
    """Test graceful skip when no webhook URL is configured."""
    dispatcher = WebhookAlertDispatcher(webhook_url="")
    alert = AlertPayload(title="Test", message="Test")
    ok = await dispatcher.dispatch(alert)
    assert ok is False


@pytest.mark.asyncio
async def test_webhook_dispatcher_discord_dispatch() -> None:
    """Test successful dispatch to Discord webhook URL."""
    dispatcher = WebhookAlertDispatcher(webhook_url="https://discord.com/api/webhooks/123/abc")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 204

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        ok = await dispatcher.dispatch_model_promoted(
            model_tag="os-fixer:v1",
            eval_pass_rate=0.8,
            baseline_pass_rate=0.5,
            delta=0.3,
            iteration=1,
        )
        assert ok is True


@pytest.mark.asyncio
async def test_webhook_dispatcher_regression_and_failure() -> None:
    """Test model regression and worker failure alert dispatches."""
    dispatcher = WebhookAlertDispatcher(webhook_url="https://hooks.slack.com/services/123/456")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        ok_reg = await dispatcher.dispatch_model_regression(
            model_tag="os-fixer:v2",
            rolled_back_to="os-fixer:v1",
            eval_pass_rate=0.3,
            baseline_pass_rate=0.8,
            delta=-0.5,
            iteration=2,
        )
        assert ok_reg is True

        ok_fail = await dispatcher.dispatch_worker_failure(
            instance_id="autofix-box-1",
            scenario="systemd_dns",
            error_message="Guest agent handshake timed out",
        )
        assert ok_fail is True
