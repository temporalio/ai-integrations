"""Load shared fixtures and configure OpenAI Agents test isolation."""

from __future__ import annotations

import opentelemetry.trace
import pytest
from agents.tracing import set_trace_processors
from opentelemetry.util._once import Once

pytest_plugins = ["temporalio_ai_integrations_pytest"]

# The Agents SDK installs a process-global exporter by default. Tests that
# exercise tracing install their own processors, so begin with an empty set to
# prevent unrelated test spans from being exported from background threads.
set_trace_processors([])

CONTINUE_AS_NEW_SUGGEST_HISTORY_COUNT = 50


# OpenTelemetry's global providers are set-once per process with no public
# way to unset them, so tests needing their own provider must reset the
# globals directly -- the same isolation pattern OpenTelemetry's own test
# suite uses (opentelemetry.test.globals_test). Isolation cannot be delegated
# to scheduling: CI runs pytest-xdist with --dist=worksteal, which ignores
# xdist_group pinning, so any test in the suite can share a worker process
# with any other. Every test that installs a global provider must therefore
# use these fixtures and leave the globals reset behind it.


@pytest.fixture
def reset_otel_tracer_provider():
    """Isolate global OpenTelemetry tracer provider state around a test.

    Proxy tracers bound to a real provider stay bound forever; OTel has no
    rebind mechanism for tracers. Tests must install their provider before
    any span is created through a proxy tracer they care about.
    """
    opentelemetry.trace._TRACER_PROVIDER_SET_ONCE = Once()
    opentelemetry.trace._TRACER_PROVIDER = None
    yield
    opentelemetry.trace._TRACER_PROVIDER_SET_ONCE = Once()
    opentelemetry.trace._TRACER_PROVIDER = None
