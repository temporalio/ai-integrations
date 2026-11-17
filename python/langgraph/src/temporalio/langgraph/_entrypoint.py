"""Keep Functional API checkpoints with one Workflow execution instance."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel  # pyright: ignore[reportMissingTypeStubs]

Program = Pregel[Any, Any, Any, Any]
_WORKFLOW_ENTRYPOINTS = "_temporal_langgraph_checkpointed_entrypoints"


def validate_entrypoint_checkpointers(programs: Mapping[str, Program]) -> None:
    """Refuse checkpointers whose construction the plugin cannot reproduce."""
    for name, program in programs.items():
        saver = program.checkpointer
        if (
            saver is not None
            and not isinstance(saver, bool)
            and type(saver) is not InMemorySaver
        ):
            raise ValueError(
                f"Entrypoint {name!r} must use a standard InMemorySaver "
                "checkpointer. Custom checkpointers and InMemorySaver subclasses "
                "cannot be shared between Workflow instances."
            )


@dataclass
class _WorkflowEntrypoints:
    programs: dict[int, Program] = field(default_factory=dict)
    savers: dict[int, InMemorySaver] = field(default_factory=dict)


def workflow_entrypoint(program: Program) -> Program:
    """Return this Workflow instance's entrypoint and in-memory checkpointer.

    The event loop belongs to the Workflow instance, including its constructor
    and handlers, and is replaced when History replays after eviction. Native
    ``Pregel.copy`` preserves the program's configuration and wrapped tasks.
    Aliases of a program or saver keep sharing it within the same instance.
    """
    saver = program.checkpointer
    if not isinstance(saver, InMemorySaver):
        return program
    loop = asyncio.get_running_loop()
    scoped: _WorkflowEntrypoints | None = getattr(loop, _WORKFLOW_ENTRYPOINTS, None)
    if scoped is None:
        scoped = _WorkflowEntrypoints()
        setattr(loop, _WORKFLOW_ENTRYPOINTS, scoped)
    if id(program) not in scoped.programs:
        if id(saver) not in scoped.savers:
            scoped.savers[id(saver)] = InMemorySaver(serde=saver.serde)
        scoped.programs[id(program)] = program.copy(
            update={"checkpointer": scoped.savers[id(saver)]}
        )
    return scoped.programs[id(program)]
