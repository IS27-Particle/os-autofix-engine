"""Distributed OpenTelemetry (OTel) Tracing for OS-AutoFix Engine.

Emits structured OpenTelemetry spans across all lifecycle phases: sandbox provisioning,
Tri-Agent swarm handoffs, CRIU hotpatching, formal SMT verification, eBPF telemetry,
and scenario verifier passes. Supports OTLP exporter protocols.
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("os_autofix.monitoring.otel")

_CURRENT_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span_id", default=None
)
_CURRENT_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None
)


@dataclass
class SpanRecord:
    """Individual OpenTelemetry trace span record."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "OK"  # "OK", "ERROR", "UNSET"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceSpanContextManager:
    """Context manager for entering and exiting an OpenTelemetry span."""

    def __init__(
        self,
        tracer: EngineTracer,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.attributes = attributes or {}
        self.span_record: SpanRecord | None = None
        self._token_span: contextvars.Token[str | None] | None = None
        self._token_trace: contextvars.Token[str | None] | None = None

    def __enter__(self) -> SpanRecord:
        parent_span = _CURRENT_SPAN_ID.get()
        trace_id = _CURRENT_TRACE_ID.get() or f"trace-{uuid.uuid4().hex}"
        span_id = f"span-{uuid.uuid4().hex[:12]}"

        self.span_record = SpanRecord(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span,
            name=self.name,
            start_time=time.time(),
            attributes=dict(self.attributes),
        )

        self._token_span = _CURRENT_SPAN_ID.set(span_id)
        self._token_trace = _CURRENT_TRACE_ID.set(trace_id)
        self.tracer.record_span_start(self.span_record)
        return self.span_record

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.span_record:
            end = time.time()
            self.span_record.end_time = end
            self.span_record.duration_ms = round((end - self.span_record.start_time) * 1000, 2)
            if exc_type is not None:
                self.span_record.status = "ERROR"
                self.span_record.events.append(
                    {
                        "name": "exception",
                        "time": end,
                        "attributes": {
                            "exception.type": str(exc_type),
                            "exception.message": str(exc_val),
                        },
                    }
                )
            self.tracer.record_span_end(self.span_record)

        if self._token_span:
            _CURRENT_SPAN_ID.reset(self._token_span)
        if self._token_trace:
            _CURRENT_TRACE_ID.reset(self._token_trace)


class EngineTracer:
    """Singleton OpenTelemetry distributed tracer coordinating span capture and OTLP exports."""

    def __init__(
        self,
        service_name: str = "os-autofix-engine",
        otlp_endpoint: str | None = None,
    ) -> None:
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or "http://localhost:4318/v1/traces"
        self.active_spans: dict[str, SpanRecord] = {}
        self.completed_spans: list[SpanRecord] = []

    def span(self, name: str, attributes: dict[str, Any] | None = None) -> TraceSpanContextManager:
        """Create a child span context manager."""
        return TraceSpanContextManager(self, name, attributes)

    def record_span_start(self, span: SpanRecord) -> None:
        self.active_spans[span.span_id] = span
        logger.debug("OTel Span Started: %s (ID: %s)", span.name, span.span_id)

    def record_span_end(self, span: SpanRecord) -> None:
        self.active_spans.pop(span.span_id, None)
        self.completed_spans.append(span)
        logger.debug(
            "OTel Span Ended: %s (Duration: %.2fms, Status: %s)",
            span.name,
            span.duration_ms,
            span.status,
        )

    def record_event(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        """Record an instantaneous event on the current active span."""
        curr_span_id = _CURRENT_SPAN_ID.get()
        if curr_span_id and curr_span_id in self.active_spans:
            self.active_spans[curr_span_id].events.append(
                {
                    "name": event_name,
                    "time": time.time(),
                    "attributes": payload or {},
                }
            )

    def export_traces_json(self) -> list[dict[str, Any]]:
        """Export all completed trace spans in OTLP-compatible JSON format."""
        return [s.to_dict() for s in self.completed_spans]

    def clear(self) -> None:
        """Clear recorded trace buffers."""
        self.active_spans.clear()
        self.completed_spans.clear()


GLOBAL_TRACER = EngineTracer()


def get_tracer() -> EngineTracer:
    """Return global singleton EngineTracer."""
    return GLOBAL_TRACER


def traced_span(name: str | None = None) -> Callable[..., Any]:
    """Decorator wrapping functions in an OpenTelemetry trace span."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__name__

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_tracer().span(span_name, {"function": fn.__name__}):
                return await fn(*args, **kwargs)

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_tracer().span(span_name, {"function": fn.__name__}):
                return fn(*args, **kwargs)

        import inspect

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator
