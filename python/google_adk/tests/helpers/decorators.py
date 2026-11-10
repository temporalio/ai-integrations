"""Decorator defined outside the modules of the functions it wraps."""

import functools
import inspect
import types
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


def unwrapped(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Like ``passthrough``, but without ``__wrapped__``: ``fn``'s globals are out of reach."""
    wrapper = passthrough(fn)
    delattr(wrapper, "__wrapped__")
    setattr(wrapper, "__signature__", inspect.signature(fn))
    return wrapper


class WrapOnAccess:
    """Stays in the class; each access returns a new ``functools.wraps`` wrapper, as wrapt does."""

    __wrapped__: Callable[..., Awaitable[Any]]

    def __init__(self, fn: Callable[..., Awaitable[Any]]) -> None:
        self.__wrapped__ = fn

    def __get__(
        self, obj: object, cls: type | None = None
    ) -> Callable[..., Awaitable[Any]]:
        fn = self.__wrapped__
        return passthrough(fn if obj is None else types.MethodType(fn, obj))
