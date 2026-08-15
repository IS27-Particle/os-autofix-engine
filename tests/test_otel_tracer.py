"""Unit tests for OpenTelemetry distributed tracing."""

from __future__ import annotations

import pytest

from monitoring.otel_tracer import EngineTracer, SpanRecord, get_tracer, traced_span


def test_otel_span_context_and_duration() -> None:
    """Test basic span execution context and timing calculation."""
    tracer = EngineTracer(service_name="test-service")

    with tracer.span("test_operation", {"custom_attr": "val123"}) as span:
        assert span.name == "test_operation"
        assert span.trace_id.startswith("trace-")
        assert span.attributes["custom_attr"] == "val123"

    assert len(tracer.completed_spans) == 1
    completed: SpanRecord = tracer.completed_spans[0]
    assert completed.duration_ms >= 0.0
    assert completed.status == "OK"


def test_otel_parent_child_hierarchy() -> None:
    """Test nested parent and child span relationship tracking."""
    tracer = EngineTracer()

    with tracer.span("parent_span") as parent:
        with tracer.span("child_span_1") as child1:
            assert child1.parent_span_id == parent.span_id
            tracer.record_event("child_sub_event", {"metric": 42})

        with tracer.span("child_span_2") as child2:
            assert child2.parent_span_id == parent.span_id

    assert len(tracer.completed_spans) == 3
    names = [s.name for s in tracer.completed_spans]
    assert "child_span_1" in names
    assert "child_span_2" in names
    assert "parent_span" in names

    # Verify event capture
    child1_record = next(s for s in tracer.completed_spans if s.name == "child_span_1")
    assert len(child1_record.events) == 1
    assert child1_record.events[0]["name"] == "child_sub_event"


def test_otel_decorator_and_exception_handling() -> None:
    """Test @traced_span decorator and error status tagging on exceptions."""
    tracer = get_tracer()
    tracer.clear()

    @traced_span(name="failing_operation")
    def problematic_func() -> None:
        raise ValueError("Simulated fault")

    with pytest.raises(ValueError, match="Simulated fault"):
        problematic_func()

    assert len(tracer.completed_spans) >= 1
    err_span = next(s for s in tracer.completed_spans if s.name == "failing_operation")
    assert err_span.status == "ERROR"
    assert len(err_span.events) >= 1
    assert err_span.events[0]["name"] == "exception"


def test_otel_export_traces_json() -> None:
    """Test exporting spans in JSON/OTLP payload format."""
    tracer = EngineTracer()
    with tracer.span("exportable_span"):
        pass

    raw_json = tracer.export_traces_json()
    assert isinstance(raw_json, list)
    assert len(raw_json) == 1
    assert raw_json[0]["name"] == "exportable_span"
