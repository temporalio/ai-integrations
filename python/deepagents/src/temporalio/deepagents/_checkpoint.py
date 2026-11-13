"""Pause a Deep Agent at a model boundary and carry its graph checkpoint.

Checkpoints stay in workflow memory. Only the current, materialized checkpoint
crosses continue-as-new; no database, historical result cache, or pickle is used.
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import base64
import copy
import dataclasses
import inspect
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Any, cast, get_args, get_type_hints

from langchain_core.runnables import RunnableConfig
from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.base.id import uuid6
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


class _ResumedUntrackedValue(UntrackedValue[Any]):
    """Seed invocation-local state once when continuing an unfinished invocation."""

    def __init__(self, typ: Any, guard: bool = True, key: str = "") -> None:
        super().__init__(typ, guard)
        self.key = key

    def from_checkpoint(self, checkpoint: Any) -> _ResumedUntrackedValue:
        channel = super().from_checkpoint(checkpoint)
        # The parent intentionally ignores checkpoint values. Our carried
        # checkpoint includes live invocation state; subsequent saves retain
        # native UntrackedValue semantics and do not persist these fields.
        if checkpoint is not MISSING:
            channel.update([checkpoint])
        return channel


def _serializer(agent: Any) -> JsonPlusSerializer:
    """Allow checkpoint types declared by the graph, including message classes."""
    allowed = {
        (cls.__module__, cls.__name__) for cls in _serde._tool_arg_message_classes()
    }
    pending = [
        typ
        for channel in agent.channels.values()
        if isinstance(channel, BaseChannel)
        for typ in (channel.ValueType, channel.UpdateType)
    ]
    seen: set[Any] = set()
    unhashable: list[Any] = []
    while pending:
        typ = pending.pop()
        try:
            if typ in seen:
                continue
            seen.add(typ)
        except TypeError:
            if any(typ is previous for previous in unhashable):
                continue
            unhashable.append(typ)
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


def _check_restored_types(original: Any, restored: Any) -> None:
    """Reject serialization that silently replaces a state object with raw data."""
    pending = [(original, restored)]
    seen: set[tuple[int, int]] = set()
    while pending:
        value, decoded = pending.pop()
        pair = (id(value), id(decoded))
        if pair in seen:
            continue
        seen.add(pair)
        if type(value) is not type(decoded):
            raise DeepAgentsWorkflowError(
                f"Checkpoint state type {type(value).__module__}."
                f"{type(value).__qualname__} cannot be restored. Declare importable "
                "state types in the graph schema; local classes are not supported."
            )
        if isinstance(value, Mapping):
            pending.extend((item, decoded[key]) for key, item in value.items())
        elif isinstance(value, (list, tuple)):
            pending.extend(zip(value, decoded, strict=True))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            pending.extend(
                (getattr(value, field.name), getattr(decoded, field.name))
                for field in dataclasses.fields(value)
            )
        elif hasattr(value, "__dict__"):
            pending.append((vars(value), vars(decoded)))


def _validate_workflow_signature(input: Any) -> None:
    """Reject a continued-run signature before scheduling the first model call."""
    if not workflow.in_workflow():
        return
    instance = workflow.instance()
    definition = workflow._Definition.must_from_class(type(instance))
    try:
        inspect.signature(definition.run_fn).bind(instance, input, None)
    except TypeError as exc:
        raise DeepAgentsWorkflowError(
            "run_deep_agent requires a workflow run method accepting "
            "(input, state_snapshot=None) for continue-as-new."
        ) from exc


def _can_resume_interrupts(agent: Any) -> bool:
    return isinstance(agent.checkpointer, InMemorySaver) and bool(
        (agent.config or {}).get("configurable", {}).get("thread_id")
    )


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
            raise DeepAgentsWorkflowError(
                "A graph checkpoint requires a compiled Deep Agent."
            )
        return await agent.ainvoke(input)

    input = copy.deepcopy(input)
    saver = agent.checkpointer
    if saver is None:
        saver = InMemorySaver()
    elif not isinstance(saver, InMemorySaver):
        raise DeepAgentsWorkflowError(
            "run_deep_agent requires an InMemorySaver or no checkpointer."
        )
    config: RunnableConfig = agent.config.copy() if agent.config is not None else {}
    graph = agent.copy({"checkpointer": saver, "config": None})
    configurable: dict[str, Any] = dict(config.get("configurable") or {})
    if configurable.get("checkpoint_ns"):
        raise DeepAgentsWorkflowError("run_deep_agent must drive the root agent graph.")
    if "thread_id" in configurable:
        configurable["thread_id"] = str(configurable["thread_id"])
    carried = state_snapshot.get(SNAPSHOT_KEY) if state_snapshot else None
    budget: int = config.get("recursion_limit", 9999)
    shape = {"nodes": sorted(graph.nodes), "channels": sorted(graph.channels)}
    if carried is not None:
        if not isinstance(carried, Mapping) or carried.get("version") != 1:
            raise DeepAgentsWorkflowError(
                "Unsupported Deep Agents checkpoint snapshot."
            )
        restored: Mapping[str, Any] = carried
        if restored.get("graph") != shape:
            raise DeepAgentsWorkflowError(
                "The graph's nodes or state channels changed since its checkpoint."
            )
        try:
            configurable["thread_id"] = str(restored["thread_id"])
            checkpoint = _serializer(graph).loads_typed(
                ("msgpack", base64.b64decode(restored["checkpoint"], validate=True))
            )
            step = restored["metadata"]["step"]
            budget = restored["remaining_steps"]
            if not isinstance(step, int) or not isinstance(budget, int):
                raise ValueError("Invalid checkpoint recursion metadata.")
            if not isinstance(checkpoint, dict) or not all(
                key in checkpoint
                for key in (
                    "v",
                    "ts",
                    "id",
                    "channel_values",
                    "channel_versions",
                    "versions_seen",
                )
            ):
                raise ValueError("Invalid graph checkpoint.")
        except (KeyError, TypeError, ValueError) as exc:
            raise DeepAgentsWorkflowError(
                "Invalid Deep Agents checkpoint snapshot."
            ) from exc
        configurable.pop("checkpoint_id", None)
        # Native IDs sort by worker wall time. Rebase the carried ID so a worker
        # with an earlier clock can still produce newer checkpoints.
        checkpoint["id"] = str(uuid6(clock_seq=step))
        await saver.aput(
            {
                "configurable": {
                    "thread_id": configurable["thread_id"],
                    "checkpoint_ns": "",
                }
            },
            cast(Checkpoint, cast(object, checkpoint)),
            restored["metadata"],
            checkpoint["channel_versions"],
        )
        input = None
        graph = graph.copy(
            {
                "channels": {
                    key: _ResumedUntrackedValue(
                        channel.ValueType, channel.guard, channel.key
                    )
                    if isinstance(channel, UntrackedValue)
                    else channel
                    for key, channel in graph.channels.items()
                }
            }
        )
    else:
        configurable.setdefault("thread_id", str(workflow.uuid4()))
        if budget < 1:
            raise DeepAgentsWorkflowError("recursion_limit must be at least 1.")
    if budget < 0:
        raise DeepAgentsWorkflowError("Deep Agent exceeded its recursion limit.")
    config["configurable"] = configurable
    config["recursion_limit"] = max(budget, 1)
    automatic = not graph.interrupt_before_nodes and not graph.interrupt_after_nodes
    if automatic:
        _validate_workflow_signature(input_original)
    if not _can_resume_interrupts(agent) and any(
        name.startswith("HumanInTheLoopMiddleware.") for name in graph.nodes
    ):
        raise DeepAgentsWorkflowError(
            "Application interrupts require an explicit InMemorySaver and "
            "configurable.thread_id so the caller can resume the agent."
        )
    initial = await saver.aget_tuple(config)
    previous_step = initial.metadata.get("step", -2) if initial else -2
    progressed = False
    boundary: Any = None
    result: Any = None
    interrupts: list[Any] = []
    # Checkpoint stream events precede execution of the next superstep. Keeping
    # one invocation alive preserves UntrackedValue counters and caches, and
    # avoids restoring delta-backed channels at every model boundary.
    stream = cast(
        AsyncGenerator[tuple[str, Any], None],
        graph.astream(
            input,
            config,
            stream_mode=["values", "updates", "checkpoints"],
            output_keys=graph.output_channels,
        ),
    )
    try:
        async with aclosing(stream):
            async for mode, value in stream:
                if mode == "values":
                    result = value
                elif mode == "updates" and "__interrupt__" in value:
                    interrupts.extend(value["__interrupt__"])
                elif (
                    mode == "checkpoints"
                    and automatic
                    and value["next"] == ["model"]
                    and not any(task.get("interrupts") for task in value["tasks"])
                ):
                    past_limit = (
                        workflow.info().is_continue_as_new_suggested()
                        if continue_as_new_after is None
                        else workflow.info().get_current_history_length()
                        >= continue_as_new_after
                    )
                    if progressed and past_limit:
                        boundary = value
                        break
                    progressed = True
    except GraphRecursionError as exc:
        raise DeepAgentsWorkflowError(
            "Deep Agent exceeded its recursion limit. Increase the agent's "
            "recursion_limit to allow more steps across the entire invocation."
        ) from exc
    if boundary is not None:
        saved = await saver.aget_tuple(boundary["config"])
        if saved is None or saved.pending_writes:
            raise DeepAgentsWorkflowError("Cannot carry an unsettled graph checkpoint.")
        step = saved.metadata.get("step")
        if step is None:
            raise DeepAgentsWorkflowError("Graph checkpoint metadata is missing.")
        checkpoint = {
            **saved.checkpoint,
            "channel_values": {
                **saved.checkpoint["channel_values"],
                **boundary["values"],
            },
        }
        serde = _serializer(graph)
        try:
            kind, data = serde.dumps_typed(checkpoint)
            if kind != "msgpack":
                raise DeepAgentsWorkflowError("Unsupported graph checkpoint encoding.")
            decoded = serde.loads_typed((kind, data))
            _check_restored_types(
                checkpoint["channel_values"], decoded["channel_values"]
            )
        except (TypeError, ValueError) as exc:
            raise DeepAgentsWorkflowError(
                "Graph checkpoint state is not serializable."
            ) from exc
        snapshot = {
            SNAPSHOT_KEY: {
                "version": 1,
                "graph": shape,
                "thread_id": configurable["thread_id"],
                "checkpoint": base64.b64encode(data).decode("ascii"),
                "metadata": saved.metadata,
                "remaining_steps": budget - (step - previous_step),
            }
        }
        if isinstance(input_original, Mapping):
            input_original = {
                key: value
                for key, value in input_original.items()
                if key not in {"messages", "files"}
            }
        workflow.continue_as_new(args=[input_original, snapshot])
    if interrupts:
        if not _can_resume_interrupts(agent):
            raise DeepAgentsWorkflowError(
                "Application interrupts require an explicit InMemorySaver and "
                "configurable.thread_id so the caller can resume the agent."
            )
        return {
            **(result if isinstance(result, dict) else {}),
            "__interrupt__": interrupts,
        }
    return result
