"""Activities in a module whose annotations are strings (PEP 563)."""

from __future__ import annotations

from typing import Self

from temporalio import activity
from temporalio.google_adk.workflow import ToolContextSnapshot


class PostponedHolder:
    @activity.defn
    async def track(self: Self, parcel_id: str) -> str:
        """Method whose receiver annotation is the string 'Self'."""
        return parcel_id


@activity.defn
class PostponedSnapshotActivity:
    async def __call__(
        self,
        query: str,
        tool_context: ToolContextSnapshot | None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> str:
        return query
