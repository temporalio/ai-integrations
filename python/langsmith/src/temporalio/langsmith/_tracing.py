"""Scope plugin destination defaults below explicit LangSmith addressing."""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langsmith import run_helpers

from temporalio.langsmith._signature import bind, signature

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
    original_signature = signature(original, "container_input")

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _destination.get() and (call := bind(original_signature, args, kwargs)):
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

    overrides: dict[str, Callable[..., Any]] = {}
    overrides["_resolve_traceable_addressing"] = _with_container_defaults(
        run_helpers._resolve_traceable_addressing
    )

    original_tiers = run_helpers._addressing_tiers

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

    overrides["_addressing_tiers"] = tiers

    # Patch nothing until every signature check has passed.
    for name, override in overrides.items():
        setattr(run_helpers, name, override)
    _installed = True
