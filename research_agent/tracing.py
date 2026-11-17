"""OpenTelemetry tracing using the GenAI semantic conventions.

Every research run is one `invoke_agent` span; every tool call is a child
`execute_tool` span. Any OTel backend (Arize Phoenix, Langfuse, Jaeger,
Azure Monitor) can display them.

Exporters, chosen by environment:
  OTEL_EXPORTER_OTLP_ENDPOINT set -> OTLP/HTTP (e.g. Phoenix at http://localhost:6006)
  RESEARCH_TRACE_CONSOLE=1        -> print spans to stdout
  otherwise                       -> tracing is on but nothing is exported
Tests attach an in-memory exporter with `use_exporter()`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

AGENT_NAME = "secure-research-agent"
_provider: TracerProvider | None = None


def _setup() -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    _provider = TracerProvider(resource=Resource.create({"service.name": AGENT_NAME}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
    if os.environ.get("RESEARCH_TRACE_CONSOLE") == "1":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return _provider


def use_exporter(exporter: SpanExporter) -> None:
    _setup().add_span_processor(SimpleSpanProcessor(exporter))


def tracer() -> trace.Tracer:
    return _setup().get_tracer(AGENT_NAME)


@contextmanager
def agent_span(question: str, user_id: str, mode: str) -> Iterator[trace.Span]:
    with tracer().start_as_current_span(f"invoke_agent {AGENT_NAME}") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", AGENT_NAME)
        span.set_attribute("research.user_id", user_id)
        span.set_attribute("research.mode", mode)
        span.set_attribute("research.question_length", len(question))  # not the text itself
        yield span


@contextmanager
def tool_span(tool_name: str) -> Iterator[trace.Span]:
    with tracer().start_as_current_span(f"execute_tool {tool_name}") as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", tool_name)
        yield span


@contextmanager
def llm_span(model: str) -> Iterator[trace.Span]:
    with tracer().start_as_current_span(f"chat {model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        yield span
