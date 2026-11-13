import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar, cast

from temporalio import activity

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def auto_heartbeater(fn: F) -> F:
    """Decorator that heartbeats every half heartbeat timeout while the activity runs.

    ``call_later`` starts the timer at once, where a task would wait for the
    activity's first await, so code that blocks the event loop before then only
    delays the first heartbeat.
    """

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        heartbeat_timeout = activity.info().heartbeat_timeout
        if not heartbeat_timeout:
            return await fn(*args, **kwargs)
        interval = heartbeat_timeout.total_seconds() / 2
        loop = asyncio.get_running_loop()
        handle: asyncio.TimerHandle

        def beat() -> None:
            nonlocal handle
            handle = loop.call_later(interval, beat)
            activity.heartbeat()

        handle = loop.call_later(interval, beat)
        try:
            return await fn(*args, **kwargs)
        finally:
            handle.cancel()

    return cast(F, wrapper)
