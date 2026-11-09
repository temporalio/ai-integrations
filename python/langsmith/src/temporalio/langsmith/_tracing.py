"""Scope plugin destination defaults below explicit LangSmith addressing."""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langsmith import run_helpers

from temporalio.langsmith._signature import bind

_destination: ContextVar[dict[str, Any] | None] = ContextVar(
    "temporal_langsmith_destination", default=None
)
_installed = False


@contextmanager
def destination_context(destination: dict[str, Any]) -> Generator[None, None, None]:
    """Bind the receiving plugin's defaults without setting tracing_context."""
    token = _destination.set(destination)
    try:
        yield
    finally:
        _destination.reset(token)


def _with_defaults(container: Any) -> Any:
    """Supply defaults only when the decorator names neither destination mode."""
    destination = _destination.get()
    if (
        destination
        and not container.get("project_name")
        and container.get("address") is None
    ):
        return {**container, **destination}
    return container


def _with_container_defaults(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a LangSmith function to apply plugin defaults to its container input."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _destination.get() and (
            call := bind(original, args, kwargs, "container_input")
        ):
            container = call.arguments["container_input"]
            call.arguments["container_input"] = _with_defaults(container)
            args, kwargs = call.args, call.kwargs
        return original(*args, **kwargs)

    return wrapper


def install_addressing_override() -> None:
    """Insert plugin defaults after explicit addressing, before configure/env."""
    global _installed  # noqa: PLW0603
    if _installed:
        return

    original_resolve = getattr(run_helpers, "_resolve_traceable_addressing", None)
    if original_resolve is not None:
        run_helpers._resolve_traceable_addressing = _with_container_defaults(
            original_resolve
        )
    else:
        # Older LangSmith releases resolve project precedence inline. Only
        # supply the decorator fallback; leave run selection and setup native.
        run_helpers._setup_run = _with_container_defaults(run_helpers._setup_run)

    original_tiers = getattr(run_helpers, "_addressing_tiers", None)
    if original_tiers is not None:

        @functools.wraps(original_tiers)
        def tiers(*args: Any, **kwargs: Any) -> Any:
            levels = original_tiers(*args, **kwargs)
            destination = _destination.get()
            if not destination:
                return levels
            # trace() has a separate resolver: its last explicit tier is
            # configure(), followed by the environment inside native resolve().
            return (
                *levels[:-1],
                (destination.get("project_name"), destination.get("address")),
                levels[-1],
            )

        run_helpers._addressing_tiers = tiers
    else:
        original_project = run_helpers._get_project_name

        @functools.wraps(original_project)
        def project(*args: Any, **kwargs: Any) -> str | None:
            destination = _destination.get()
            if destination and (
                call := bind(original_project, args, kwargs, "project_name")
            ):
                context = run_helpers.get_tracing_context()
                if (
                    not call.arguments["project_name"]
                    and not context.get("project_name")
                    and context.get("parent") is None
                ):
                    call.arguments["project_name"] = destination.get("project_name")
                    args, kwargs = call.args, call.kwargs
            return original_project(*args, **kwargs)

        run_helpers._get_project_name = project

    _installed = True
