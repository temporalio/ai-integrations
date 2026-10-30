"""Decorator defined outside the modules of the functions it wraps."""

import functools
from collections.abc import Awaitable, Callable
from typing import Any


def passthrough(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Wraps ``fn`` with ``functools.wraps``; the wrapper has this module's globals."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    return wrapper
