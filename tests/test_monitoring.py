"""Unit tests for Prometheus metrics, scrapable exposition, JSON logging, and TUI dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from monitoring.dashboard import DashboardManager
from monitoring.json_logger import JSONFormatter
from monitoring.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
)


def test_counter_metric_logic() -> None:
    """Test Counter incrementing and label formatting."""
    counter = Counter(
        "test_tasks_total",
        "Test counter description",
        label_names=["scenario", "status"],
    )
    counter.inc(1.0, scenario="systemd_dns", status="success")
    counter.inc(2.0, scenario="systemd_dns", status="success")
    counter.inc(1.0, scenario="systemd_dns", status="failure")

    lines = counter.collect()
    assert any("test_tasks_total" in line for line in lines)
    assert any('scenario="systemd_dns",status="success"} 3.0' in line for line in lines)
    assert any('scenario="systemd_dns",status="failure"} 1.0' in line for line in lines)


def test_gauge_metric_logic() -> None:
    """Test Gauge set, inc, dec operations."""
    gauge = Gauge("test_sandboxes_active", "Test active instances")
    gauge.set(5.0)
    gauge.inc(2.0)
    gauge.dec(1.0)

    lines = gauge.collect()
    assert any("test_sandboxes_active 6.0" in line for line in lines)


def test_histogram_metric_logic() -> None:
    """Test Histogram bucket counts and observations."""
    hist = Histogram(
        "test_latency_seconds",
        "Test latency",
        label_names=["model"],
        buckets=[0.1, 0.5, 1.0],
    )
    hist.observe(0.05, model="qwen")
    hist.observe(0.4, model="qwen")
    hist.observe(0.8, model="qwen")

    lines = hist.collect()
    assert any('le="0.1",model="qwen"} 1' in line for line in lines)
    assert any('le="0.5",model="qwen"} 2' in line for line in lines)
    assert any('le="1.0",model="qwen"} 3' in line for line in lines)
    assert any('le="+Inf",model="qwen"} 3' in line for line in lines)
    assert any("test_latency_seconds_count" in line for line in lines)
    assert any("test_latency_seconds_sum" in line for line in lines)


def test_metrics_registry_exposition() -> None:
    """Test full Prometheus exposition text generation."""
    reg = MetricsRegistry()
    c = Counter("my_counter", "Desc")
    c.inc(5.0)
    reg.register(c)

    text = reg.generate_exposition()
    assert "# HELP my_counter Desc" in text
    assert "# TYPE my_counter counter" in text
    assert "my_counter 5.0" in text


def test_tui_dashboard_render() -> None:
    """Test DashboardManager layout construction and worker tracking."""
    dash = DashboardManager(model_name="qwen2.5-coder:7b", worker_count=2)
    dash.update_worker(
        worker_id=1,
        scenario="systemd_dns",
        instance_id="autofix-test-1",
        step=2,
        max_steps=8,
        thought="Restarting resolver",
        command="systemctl restart systemd-resolved",
        status="RUNNING",
    )
    dash.record_episode_result(
        scenario="systemd_dns",
        success=True,
        steps=2,
        reward=0.95,
        duration=3.2,
    )

    layout = dash.render()
    assert layout is not None
    assert dash.total_episodes == 1
    assert dash.successful_episodes == 1


def test_json_log_formatter(tmp_path: Path) -> None:
    """Test JSONFormatter output structure."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test event message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Test event message"
    assert "timestamp" in data
