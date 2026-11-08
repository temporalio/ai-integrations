"""Pause a Deep Agent at a model boundary and carry its graph checkpoint.

Checkpoints stay in workflow memory. Only the current, materialized checkpoint
crosses continue-as-new; no database, historical result cache, or pickle is used.
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, get_args, get_type_hints

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from temporalio import workflow
from temporalio.deepagents import _serde
from temporalio.deepagents.workflow import (
    DeepAgentsWorkflowError,
    _merge_files,
    _merge_snapshot,
)

SNAPSHOT_KEY = "__temporal_graph_checkpoint__"


def _serializer(agent: Any) -> JsonPlusSerializer:
    """Allow checkpoint types declared by the graph, including message classes."""
    allowed = {
        (cls.__module__, cls.__name__) for cls in _serde._tool_arg_message_classes()
    }
    pending = [channel.ValueType for channel in agent.channels.values()]
    seen: set[int] = set()
    while pending:
        typ = pending.pop()
        if id(typ) in seen:
            continue
        seen.add(id(typ))
        pending.extend(get_args(typ))
        if isinstance(typ, type):
            allowed.add((typ.__module__, typ.__name__))
            try:
                pending.extend(get_type_hints(typ, include_extras=True).values())
            except Exception:
                # Unresolved annotations are not permission to import a type
                # supplied by the snapshot. Its module must be known here.
                pass
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


async def run(
    agent: Any,
    input: Any,
    *,
    continue_as_new_after: int | None,
    state_snapshot: Mapping[str, Any] | None,
) -> Any:
    """Run until completion, an application interrupt, or a model-boundary rollover."""
    input_original = input
    if state_snapshot and SNAPSHOT_KEY not in state_snapshot:
        # An old transcript snapshot remains usable as input to a fresh graph.
        input = _merge_files(_merge_snapshot(input, state_snapshot), state_snapshot)
    if not isinstance(agent, CompiledStateGraph) or "model" not in agent.nodes:
        if state_snapshot and SNAPSHOT_KEY in state_snapshot:
            raise ValueError("A graph checkpoint requires a compiled Deep Agent.")
        return await agent.ainvoke(input)

    saver = agent.checkpointer
    if saver is None:
        saver = InMemorySaver()
    elif not isinstance(saver, InMemorySaver):
        raise ValueError("run_deep_agent requires an InMemorySaver or no checkpointer.")
    graph = agent.copy({"checkpointer": saver})
    config: RunnableConfig = graph.config.copy() if graph.config is not None else {}
    configurable: dict[str, Any] = dict(config.get("configurable") or {})
    if configurable.get("checkpoint_ns"):
        raise ValueError("run_deep_agent must drive the root agent graph.")
    carried = state_snapshot.get(SNAPSHOT_KEY) if state_snapshot else None
    serde = _serializer(graph)
    budget: int = config.get("recursion_limit", 9999)
    shape = {"nodes": sorted(graph.nodes), "channels": sorted(graph.channels)}
    if carried is not None:
        if not isinstance(carried, Mapping) or carried.get("version") != 1:
            raise ValueError("Unsupported Deep Agents checkpoint snapshot.")
        restored: Mapping[str, Any] = carried
        if restored["graph"] != shape:
            raise ValueError(
                "The graph's nodes or state channels changed since its checkpoint."
            )
        configurable["thread_id"] = restored["thread_id"]
        configurable.pop("checkpoint_id", None)
        checkpoint = serde.loads_typed(
            ("msgpack", base64.b64decode(restored["checkpoint"], validate=True))
        )
        await saver.aput(
            {"configurable": {"thread_id": restored["thread_id"], "checkpoint_ns": ""}},
            checkpoint,
            restored["metadata"],
            checkpoint["channel_versions"],
        )
        budget = restored["remaining_steps"]
        input = None
    else:
        configurable.setdefault("thread_id", str(workflow.uuid4()))
    config["configurable"] = configurable
    # Respect caller-owned debugging breakpoints and HITL interrupts. Only our
    # additional model breakpoint is resumed automatically.
    before = graph.interrupt_before_nodes
    automatic = not before and not graph.interrupt_after_nodes
    breakpoints = ["model"] if automatic else before
    initial = await saver.aget_tuple(config)
    previous_step = initial.metadata.get("step", -2) if initial else -2
    progressed = carried is not None
    while True:
        if budget <= 0:
            raise GraphRecursionError("Deep Agent exceeded its recursion limit.")
        config["recursion_limit"] = budget
        result = await graph.ainvoke(input, config, interrupt_before=breakpoints)
        configurable.pop("checkpoint_id", None)
        state = await graph.aget_state(config)
        if state.interrupts and (
            agent.checkpointer is None
            or not (agent.config or {}).get("configurable", {}).get("thread_id")
        ):
            raise DeepAgentsWorkflowError(
                "Application interrupts require an explicit InMemorySaver and "
                "configurable.thread_id so the caller can resume the agent."
            )
        # A final answer or an application interrupt always belongs to the
        # caller. Neither a large history nor unfinished todos starts it again.
        if not automatic or state.next != ("model",) or state.interrupts:
            return result
        if state.metadata is None:
            raise DeepAgentsWorkflowError("Graph checkpoint metadata is missing.")
        step = state.metadata.get("step", -2)
        budget -= step - previous_step
        previous_step = step
        past_limit = (
            workflow.info().is_continue_as_new_suggested()
            if continue_as_new_after is None
            else workflow.info().get_current_history_length() >= continue_as_new_after
        )
        if progressed and past_limit:
            saved = await saver.aget_tuple(config)
            if saved is None or saved.pending_writes:
                raise DeepAgentsWorkflowError(
                    "Cannot carry an unsettled graph checkpoint."
                )
            # Delta channels normally depend on ancestor writes. Materialize
            # their current values so the next run needs no checkpoint history.
            checkpoint = {
                **saved.checkpoint,
                "channel_values": {
                    **saved.checkpoint["channel_values"],
                    **state.values,
                },
            }
            kind, data = serde.dumps_typed(checkpoint)
            if kind != "msgpack":
                raise DeepAgentsWorkflowError("Unsupported graph checkpoint encoding.")
            snapshot = {
                SNAPSHOT_KEY: {
                    "version": 1,
                    "graph": shape,
                    "thread_id": configurable["thread_id"],
                    "checkpoint": base64.b64encode(data).decode("ascii"),
                    "metadata": saved.metadata,
                    "remaining_steps": budget,
                }
            }
            workflow.continue_as_new(args=[input_original, snapshot])
        # The first breakpoint may precede all model work. Every run is allowed
        # at least one model step, even when its carried input exceeds the limit.
        progressed = True
        input = None
