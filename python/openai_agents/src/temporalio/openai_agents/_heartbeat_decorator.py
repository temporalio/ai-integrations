import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, TypeVar, cast, overload

from temporalio import activity

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@overload
def auto_heartbeater(fn: F) -> F: ...


@overload
def auto_heartbeater(*, interval: float) -> Callable[[F], F]: ...


def auto_heartbeater(
    fn: F | None = None, *, interval: float | None = None
) -> F | Callable[[F], F]:
    """Decorator that heartbeats while the activity runs.

    By default it heartbeats at half the activity's heartbeat timeout, and not
    at all without one. With ``interval``, it heartbeats every ``interval``
    seconds whatever the timeout.
    """

    def decorate(fn: F) -> F:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with heartbeating(interval):
                return await fn(*args, **kwargs)

        return cast(F, wrapper)

    return decorate(fn) if fn is not None else decorate


@asynccontextmanager
async def heartbeating(interval: float | None = None) -> AsyncGenerator[None]:
    """Heartbeat in the background while the block runs, as ``auto_heartbeater`` does.

    Stopping the heartbeats on exit awaits the heartbeat task and swallows a cancel
    that arrives meanwhile. An activity that must act on every requested cancel checks
    ``activity.cancellation_details()`` after the block.
    """
    delay = interval
    if delay is None:
        heartbeat_timeout = activity.info().heartbeat_timeout
        if heartbeat_timeout:
            delay = heartbeat_timeout.total_seconds() / 2
    heartbeat_task = None
    if delay:
        heartbeat_task = asyncio.create_task(_heartbeat_every(delay))
    try:
        yield
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


async def _heartbeat_every(delay: float) -> None:
    while True:
        await asyncio.sleep(delay)
        activity.heartbeat()
