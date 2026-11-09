"""Read LangSmith call arguments by name, whatever the installed signature."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def bind(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], *names: str
) -> inspect.BoundArguments | None:
    """Bind a call to ``fn``'s current signature, with defaults applied.

    Returns ``None`` when the call doesn't fit that signature or the signature
    lacks any of ``names``.
    """
    try:
        call = inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    if not all(name in call.signature.parameters for name in names):
        return None
    call.apply_defaults()
    return call
