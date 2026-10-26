"""Tools for the long-running agents. Every real execution is recorded."""

from __future__ import annotations

from typing import Any

from temporalio import activity
from tests.refund import shop


@activity.defn
async def count(args: dict[str, Any]) -> dict[str, Any]:
    """Count one step. Returns the step number."""
    shop.log_execution("count", activity.info().activity_id, str(args.get("n")))
    return {"n": args.get("n")}


@activity.defn
async def publish(args: dict[str, Any]) -> dict[str, Any]:
    """Publish the result. Needs a human approval."""
    shop.log_execution("publish", activity.info().activity_id, str(args.get("n")))
    return {"published": args.get("n")}


ALL = [count, publish]
