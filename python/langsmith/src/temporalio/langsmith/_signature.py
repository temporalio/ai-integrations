"""Check the LangSmith parameters the plugin relies on, and read them by name."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection
from typing import Any

import langsmith


def require(owner: Any, kind: str, present: Collection[str], *names: str) -> None:
    """Raise ``RuntimeError``, naming ``owner``, unless ``present`` has all ``names``."""
    missing = " or ".join(repr(name) for name in names if name not in present)
    if missing:
        raise RuntimeError(
            f"temporalio-langsmith does not support LangSmith {langsmith.__version__}: "
            f"{owner.__module__}.{owner.__qualname__} has no {kind} {missing}"
        )


def signature(fn: Callable[..., Any], *names: str) -> inspect.Signature:
    """Return ``fn``'s signature, raising ``RuntimeError`` if it lacks any of ``names``."""
    found = inspect.signature(fn)
    require(fn, "parameter", found.parameters, *names)
    return found


def bind(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> inspect.BoundArguments | None:
    """Bind a call with defaults applied, or return ``None`` when it doesn't fit."""
    try:
        call = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    call.apply_defaults()
    return call
