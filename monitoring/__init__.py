"""Monitoring package providing real-time Prometheus telemetry, metrics exporter, live TUI dashboard, structured JSON logging, and webhook alerts."""

from monitoring.alerts import (
    AlertPayload,
    WebhookAlertDispatcher,
    format_discord_payload,
    format_generic_payload,
    format_slack_payload,
)
from monitoring.dashboard import GLOBAL_DASHBOARD, DashboardManager, WorkerState
from monitoring.json_logger import JSONFormatter, setup_json_file_logging
from monitoring.metrics import (
    EPISODE_STEPS,
    LLM_LATENCY_SECONDS,
    MODEL_PASS_RATE,
    REGISTRY,
    SANDBOX_REVERT_SECONDS,
    SANDBOXES_ACTIVE,
    TASKS_TOTAL,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    start_metrics_server,
)

__all__ = [
    "REGISTRY",
    "SANDBOXES_ACTIVE",
    "TASKS_TOTAL",
    "EPISODE_STEPS",
    "LLM_LATENCY_SECONDS",
    "SANDBOX_REVERT_SECONDS",
    "MODEL_PASS_RATE",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "start_metrics_server",
    "DashboardManager",
    "GLOBAL_DASHBOARD",
    "WorkerState",
    "JSONFormatter",
    "setup_json_file_logging",
    "AlertPayload",
    "WebhookAlertDispatcher",
    "format_discord_payload",
    "format_slack_payload",
    "format_generic_payload",
]
