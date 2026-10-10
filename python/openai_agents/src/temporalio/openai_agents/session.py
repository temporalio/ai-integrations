"""Conversation memory for OpenAI Agents running in a Temporal workflow."""

from copy import deepcopy

from agents import SessionSettings, TResponseInputItem

from temporalio import workflow


class WorkflowSession:
    """Store an agent conversation in the state of one workflow execution.

    Create and reuse the same instance inside a workflow. Its contents are rebuilt
    when Temporal replays that workflow, but are not shared with other workflow
    executions. ``session_id`` identifies the conversation to the Agents SDK; it
    does not look up an existing session by ID.
    """

    def __init__(
        self, session_id: str, *, session_settings: SessionSettings | None = None
    ) -> None:
        """Create an empty session inside a Temporal workflow."""
        if not workflow.in_workflow():
            raise RuntimeError("WorkflowSession must be created inside a workflow")
        self.session_id = session_id
        self.session_settings = session_settings
        self._items: list[TResponseInputItem] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return the latest items in chronological order."""
        if limit is None and self.session_settings is not None:
            limit = self.session_settings.limit
        if limit is not None:
            if limit <= 0:
                return []
            return deepcopy(self._items[-limit:])
        return deepcopy(self._items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append a batch of conversation items."""
        self._items.extend(deepcopy(items))

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item, if any."""
        return self._items.pop() if self._items else None

    async def clear_session(self) -> None:
        """Remove every item from the conversation."""
        self._items.clear()
