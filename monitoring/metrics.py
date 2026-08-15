"""Prometheus metrics collector, exposition registry, and HTTP metrics server."""

from __future__ import annotations

import http.server
import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger("os_autofix.monitoring.metrics")


def _format_labels(labels: dict[str, str]) -> str:
    """Format dictionary of labels into Prometheus key-value string."""
    if not labels:
        return ""
    sorted_items = sorted(labels.items())
    formatted = ",".join(f'{k}="{v}"' for k, v in sorted_items)
    return f"{{{formatted}}}"


class Metric:
    """Base metric class."""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
        """Ensure all required labels are provided."""
        for required in self.label_names:
            if required not in labels:
                labels[required] = "unknown"
        # Filter only expected labels
        return tuple(sorted((k, str(labels[k])) for k in self.label_names if k in labels))

    def collect(self) -> list[str]:
        """Generate Prometheus exposition text lines."""
        return []


class Counter(Metric):
    """Monotonically increasing cumulative counter."""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None) -> None:
        super().__init__(name, description, label_names)
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        """Increment counter by specified amount (must be positive)."""
        if amount < 0:
            raise ValueError("Counter increments must be non-negative")
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] += amount

    def collect(self) -> list[str]:
        """Generate Prometheus exposition text for counter."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values and not self.label_names:
                lines.append(f"{self.name} 0.0")
            for key, val in self._values.items():
                label_str = _format_labels(dict(key))
                lines.append(f"{self.name}{label_str} {val}")
        return lines


class Gauge(Metric):
    """Value that can arbitrarily increase and decrease."""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None) -> None:
        super().__init__(name, description, label_names)
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def set(self, value: float, **labels: Any) -> None:
        """Set gauge to exact numeric value."""
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        """Increase gauge by amount."""
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, **labels: Any) -> None:
        """Decrease gauge by amount."""
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] -= amount

    def collect(self) -> list[str]:
        """Generate Prometheus exposition text for gauge."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values and not self.label_names:
                lines.append(f"{self.name} 0.0")
            for key, val in self._values.items():
                label_str = _format_labels(dict(key))
                lines.append(f"{self.name}{label_str} {val}")
        return lines


class Histogram(Metric):
    """Cumulative histogram tracking value distribution into buckets."""

    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> None:
        super().__init__(name, description, label_names)
        self.buckets = sorted(buckets or [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0])
        self._counts: dict[tuple[tuple[str, str], ...], dict[float, int]] = defaultdict(
            lambda: dict.fromkeys(self.buckets, 0)
        )
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._total_counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)

    def observe(self, value: float, **labels: Any) -> None:
        """Observe a numeric value."""
        key = self._validate_labels(labels)
        with self._lock:
            self._sums[key] += value
            self._total_counts[key] += 1
            for b in self.buckets:
                if value <= b:
                    self._counts[key][b] += 1

    def collect(self) -> list[str]:
        """Generate Prometheus exposition text for histogram."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key in self._total_counts:
                labels_dict = dict(key)
                # Buckets
                for b in self.buckets:
                    b_labels = dict(labels_dict)
                    b_labels["le"] = str(b)
                    label_str = _format_labels(b_labels)
                    lines.append(f"{self.name}_bucket{label_str} {self._counts[key][b]}")

                # +Inf bucket
                inf_labels = dict(labels_dict)
                inf_labels["le"] = "+Inf"
                lines.append(
                    f"{self.name}_bucket{_format_labels(inf_labels)} {self._total_counts[key]}"
                )

                # Sum and count
                base_label_str = _format_labels(labels_dict)
                lines.append(f"{self.name}_sum{base_label_str} {round(self._sums[key], 4)}")
                lines.append(f"{self.name}_count{base_label_str} {self._total_counts[key]}")
        return lines


class MetricsRegistry:
    """Central registry aggregating metrics for exposition."""

    def __init__(self) -> None:
        self._metrics: list[Metric] = []
        self._lock = threading.Lock()

    def register(self, metric: Metric) -> Metric:
        """Register a metric instance."""
        with self._lock:
            self._metrics.append(metric)
        return metric

    def generate_exposition(self) -> str:
        """Generate full Prometheus scrapable text."""
        output_lines: list[str] = []
        with self._lock:
            for metric in self._metrics:
                output_lines.extend(metric.collect())
        return "\n".join(output_lines) + "\n"


# Global Engine Metrics Registry
REGISTRY = MetricsRegistry()

# 1. Concurrent active sandboxes
SANDBOXES_ACTIVE = Gauge(
    "os_autofix_sandboxes_active",
    "Number of concurrent running Incus sandbox instances",
)
REGISTRY.register(SANDBOXES_ACTIVE)

# 2. Total completed tasks
TASKS_TOTAL = Counter(
    "os_autofix_tasks_total",
    "Total completed diagnostic tasks partitioned by scenario, model_tag, and status",
    label_names=["scenario", "model_tag", "status"],
)
REGISTRY.register(TASKS_TOTAL)

# 3. Episode resolution steps
EPISODE_STEPS = Histogram(
    "os_autofix_episode_steps",
    "Step counts per scenario resolution",
    label_names=["scenario", "model_tag"],
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 10, 15],
)
REGISTRY.register(EPISODE_STEPS)

# 4. LLM inference latency
LLM_LATENCY_SECONDS = Histogram(
    "os_autofix_llm_latency_seconds",
    "Inference latency per model request",
    label_names=["model_tag", "backend"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)
REGISTRY.register(LLM_LATENCY_SECONDS)

# 5. Sandbox snapshot revert latency
SANDBOX_REVERT_SECONDS = Histogram(
    "os_autofix_sandbox_revert_seconds",
    "ZFS/Btrfs CoW snapshot restore latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)
REGISTRY.register(SANDBOX_REVERT_SECONDS)

# 6. Model pass rate
MODEL_PASS_RATE = Gauge(
    "os_autofix_model_pass_rate",
    "Current benchmark generation pass rate accuracy",
    label_names=["model_tag"],
)
REGISTRY.register(MODEL_PASS_RATE)

# 7. Chaos engineering metrics
CHAOS_INJECTIONS_TOTAL = Counter(
    "os_autofix_chaos_injections_total",
    "Total autonomous chaos engineering fault injections",
    label_names=["scenario"],
)
REGISTRY.register(CHAOS_INJECTIONS_TOTAL)

CHAOS_MTTR_SECONDS = Histogram(
    "os_autofix_mttr_seconds",
    "Mean time to resolution (MTTR) latency in seconds",
    label_names=["scenario"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)
REGISTRY.register(CHAOS_MTTR_SECONDS)

CHAOS_SAFETY_VIOLATIONS = Counter(
    "os_autofix_safety_violations_total",
    "Total security and safety violations detected during remediation",
    label_names=["violation_type"],
)
REGISTRY.register(CHAOS_SAFETY_VIOLATIONS)


class MetricsHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler exposing /metrics endpoint for Prometheus scrapers."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/metrics", "/metrics/"):
            content = REGISTRY.generate_exposition().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path in ("/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found\n")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default stdout logging for clean output."""
        return


def start_metrics_server(
    port: int = 9100, host: str = "0.0.0.0"
) -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
    """Start standalone Prometheus metrics server in background daemon thread."""
    server = http.server.ThreadingHTTPServer((host, port), MetricsHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="PrometheusExporter")
    thread.start()
    logger.info("Prometheus metrics exporter started at http://%s:%d/metrics", host, port)
    return server, thread
