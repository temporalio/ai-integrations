"""One-line worker setup."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

from temporalio.plugin import SimplePlugin

from ._activity import SegmentRunner, make_segment_activity, make_tool_step_activity


class ClaudeAgentPlugin(SimplePlugin):
    """Registers the Activities that run Claude on the Worker.

    ``run_claude_segment`` runs the model, and ``run_claude_tool_step`` (for runners
    that have ``run_tool_step``) runs one Claude Code tool call as its own Activity.

    Example:
        .. code-block:: python

            runner = ClaudeAgentSdkRunner(cwd="/srv/agent")
            worker = Worker(
                client,
                task_queue="agents",
                workflows=[MyAgentWorkflow],
                activities=[my_tool],
                plugins=[ClaudeAgentPlugin(runner)],
            )
    """

    def __init__(self, runner: SegmentRunner, *, heartbeat_every: float = 5.0) -> None:
        """Create the plugin.

        Args:
            runner: Runs each model segment, for example ``ClaudeAgentSdkRunner``.
            heartbeat_every: Seconds between heartbeats while a segment or a tool
                step runs.
        """
        added = [make_segment_activity(runner, heartbeat_every=heartbeat_every)]
        if callable(getattr(runner, "run_tool_step", None)):
            # Claude Code tools that run as their own Activities.
            added.append(
                make_tool_step_activity(runner, heartbeat_every=heartbeat_every)
            )

        def activities(existing: Sequence[Any] | None) -> list[Any]:
            return [*(existing or []), *added]

        @contextlib.asynccontextmanager
        async def warm_engines_end_with_worker() -> AsyncIterator[None]:
            try:
                yield
            finally:
                end = getattr(runner, "_end_all_warm", None)
                if callable(end):
                    end()  # engines kept for the next segment (warm_engines)

        super().__init__(
            "ClaudeAgentPlugin",
            activities=activities,
            run_context=warm_engines_end_with_worker,
        )
