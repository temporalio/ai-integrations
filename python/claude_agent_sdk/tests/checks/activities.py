"""Tools for the checks tests. Every real execution is recorded in the shop ledger."""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio import activity
from tests.refund import shop


@activity.defn
async def quick(args: dict[str, Any]) -> dict[str, Any]:
    """Return at once."""
    del args
    shop.log_execution("quick", activity.info().activity_id, "done")
    return {"done": True}


@activity.defn
async def slow(args: dict[str, Any]) -> dict[str, Any]:
    """Run until cancelled (heartbeating, so the cancellation reaches it)."""
    del args
    shop.log_execution("slow", activity.info().activity_id, "started")
    try:
        while True:
            activity.heartbeat()
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        shop.log_execution("slow", activity.info().activity_id, "cancelled")
        raise


ALL = [quick, slow]
