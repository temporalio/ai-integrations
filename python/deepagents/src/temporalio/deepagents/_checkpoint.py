"""Pause a Deep Agent at a model boundary and carry its graph checkpoint.

Checkpoints stay in workflow memory. Only the current, materialized checkpoint
crosses continue-as-new; no database, historical result cache, or pickle is used.
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import inspect
import warnings
import zlib
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast, get_args, get_type_hints

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.channels.base import BaseChannel
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.base import id as checkpoint_ids
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.managed.base import ManagedValue
from langgraph.types import Command
from pydantic import BaseModel

from temporalio import workflow
from temporalio.converter import PayloadConverter
from temporalio.deepagents import _serde
from temporalio.deepagents.workflow import (
    DeepAgentsWorkflowError,
    _merge_files,
    _merge_snapshot,
)

SNAPSHOT_KEY = "__temporal_graph_checkpoint__"


# Leave room for the command envelope and server-side payload accounting.
_MAX_CARRY_BYTES = 1_900_000
_MAX_UNTRACKED_BYTES = 64 * 1024
_UNSET = object()
_METADATA_KEYS = {"source", "step", "parents", "counters_since_delta_snapshot"}


class CheckpointMismatchError(RuntimeError):
    """Retry the Workflow Task on a worker with the matching graph definition."""


class _CheckpointSerializer(JsonPlusSerializer):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            return super().dumps_typed(obj)
        except Exception as exc:
            raise DeepAgentsWorkflowError(
                "Graph checkpoint state is not serializable."
            ) from exc


class _DriverSaver(InMemorySaver):
    """Observe resolved configuration through LangGraph's saver extension point."""

    def __init__(
        self, *, serde: JsonPlusSerializer, backend: InMemorySaver | None = None
    ) -> None:
        super().__init__(serde=serde)
        self.backend = backend
        if backend is not None:
            backend.serde = serde

    def _record_limit(self, config: RunnableConfig) -> None:
        if "recursion_limit" in config and not config.get("configurable", {}).get(
            "checkpoint_ns"
        ):
            self.recursion_limit = config["recursion_limit"]

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        self._record_limit(config)
        if self.backend is not None:
            return self.backend.put(config, checkpoint, metadata, new_versions)
        return super().put(config, checkpoint, metadata, new_versions)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        self._record_limit(config)
        if self.backend is not None:
            return await self.backend.aput(config, checkpoint, metadata, new_versions)
        return self.put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config: RunnableConfig) -> Any:
        if self.backend is not None:
            return self.backend.get_tuple(config)
        return super().get_tuple(config)

    async def aget_tuple(self, config: RunnableConfig) -> Any:
        if self.backend is not None:
            return await self.backend.aget_tuple(config)
        return self.get_tuple(config)

    def list(self, *args: Any, **kwargs: Any) -> Any:
        if self.backend is not None:
            return self.backend.list(*args, **kwargs)
        return super().list(*args, **kwargs)

    def put_writes(self, *args: Any, **kwargs: Any) -> None:
        if self.backend is not None:
            self.backend.put_writes(*args, **kwargs)
        else:
            super().put_writes(*args, **kwargs)

    async def aput_writes(self, *args: Any, **kwargs: Any) -> None:
        if self.backend is not None:
            await self.backend.aput_writes(*args, **kwargs)
        else:
            self.put_writes(*args, **kwargs)

    def get_delta_channel_history(self, *args: Any, **kwargs: Any) -> Any:
        if self.backend is not None:
            return self.backend.get_delta_channel_history(*args, **kwargs)
        return super().get_delta_channel_history(*args, **kwargs)

    async def aget_delta_channel_history(self, *args: Any, **kwargs: Any) -> Any:
        if self.backend is not None:
            return await self.backend.aget_delta_channel_history(*args, **kwargs)
        return self.get_delta_channel_history(*args, **kwargs)

    def get_next_version(self, *args: Any, **kwargs: Any) -> Any:
        if self.backend is not None:
            return self.backend.get_next_version(*args, **kwargs)
        return super().get_next_version(*args, **kwargs)


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
        if hasattr(typ, "__supertype__"):
            pending.append(typ.__supertype__)
        if isinstance(typ, type):
            allowed.add((typ.__module__, typ.__name__))
            try:
                pending.extend(get_type_hints(typ, include_extras=True).values())
            except Exception:
                # Unresolved annotations are not permission to import a type
                # supplied by the snapshot. Its module must be known here.
                pass
    return _CheckpointSerializer(allowed_msgpack_modules=allowed)


def _check_restored_types(original: Any, restored: Any) -> None:
    """Reject lossy state encodings, including mapping keys and hidden fields."""
    pending = [(original, restored)]
    seen: set[tuple[int, int]] = set()
    while pending:
        value, decoded = pending.pop()
        pair = (id(value), id(decoded))
        if pair in seen:
            continue
        seen.add(pair)
        typ, decoded_type = type(value), type(decoded)
        # Msgpack uses arrays for tuples, as does LangGraph's native saver.
        sequence = isinstance(value, (list, tuple)) and isinstance(
            decoded, (list, tuple)
        )
        same_type = typ is decoded_type or (
            typ.__module__ == decoded_type.__module__
            and typ.__qualname__ == decoded_type.__qualname__
            and "<locals>" not in typ.__qualname__
        )
        if not same_type and not sequence:
            raise DeepAgentsWorkflowError(
                f"Checkpoint state type {typ.__module__}.{typ.__qualname__} cannot be restored. "
                "Declare importable state types in the graph schema; local classes are not supported."
            )
        if isinstance(value, Mapping):
            if len(value) != len(decoded):
                raise DeepAgentsWorkflowError("Checkpoint state loses mapping entries.")
            for (key, item), (other_key, other_item) in zip(
                value.items(), cast(Mapping[Any, Any], decoded).items(), strict=True
            ):
                pending.extend(((key, other_key), (item, other_item)))
        elif sequence:
            if len(value) != len(decoded):
                raise DeepAgentsWorkflowError(
                    "Checkpoint state loses sequence entries."
                )
            pending.extend(zip(value, decoded, strict=True))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            pending.extend(
                (getattr(value, field.name), getattr(decoded, field.name))
                for field in dataclasses.fields(value)
            )
        elif hasattr(value, "__dict__"):
            pending.append((vars(value), vars(decoded)))
            if isinstance(value, BaseModel):
                pending.append(
                    (
                        value.__pydantic_private__,
                        cast(BaseModel, decoded).__pydantic_private__,
                    )
                )
        elif value != decoded:
            raise DeepAgentsWorkflowError(
                "Checkpoint state loses a value during serialization."
            )


def _converter() -> PayloadConverter:
    return (
        workflow.payload_converter()
        if workflow.in_workflow()
        else _serde.build_data_converter(None).payload_converter
    )


def _input_digest(input: Any) -> str:
    return hashlib.sha256(
        _converter().to_payloads([input])[0].SerializeToString(deterministic=True)
    ).hexdigest()


def _carry_input(input: Any) -> Any:
    if isinstance(input, Mapping):
        return {
            key: value
            for key, value in input.items()
            if key not in {"messages", "files"}
        }
    return input


def _validate_workflow_signature(input: Any) -> None:
    """Validate both arity and payload decoding before scheduling paid work."""
    if not workflow.in_workflow():
        return
    instance = workflow.instance()
    definition = workflow._Definition.must_from_class(type(instance))
    try:
        inspect.signature(definition.run_fn).bind(instance, input, None)
        if definition.arg_types:
            _converter().from_payloads(
                _converter().to_payloads([input]), [definition.arg_types[0]]
            )
    except (TypeError, ValueError) as exc:
        raise DeepAgentsWorkflowError(
            "run_deep_agent requires a workflow run method accepting (input, state_snapshot=None). "
            "Pass continue_as_new_input matching the workflow's first argument."
        ) from exc


def _can_resume_interrupts(agent: Any) -> bool:
    return isinstance(agent.checkpointer, InMemorySaver) and "thread_id" in (
        agent.config or {}
    ).get("configurable", {})


def _rollover_allowed() -> bool:
    # Signal and Update handlers must finish their current turn. Also wait for
    # other handlers before rolling over from the main workflow coroutine.
    return not workflow.in_workflow() or (
        workflow.current_update_info() is None and workflow.all_handlers_finished()
    )


def _requires_interrupts(agent: Any) -> bool:
    pending = [node.data for node in agent.get_graph().nodes.values()]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        owner = getattr(node, "__self__", None)
        if isinstance(owner, HumanInTheLoopMiddleware) and owner.interrupt_on:
            return True
        pending.extend(getattr(node, "steps", ()))
        for attr in ("bound", "func", "afunc"):
            child = getattr(node, attr, None)
            if child is not None:
                pending.append(child)
    return False


def _round_trip(serde: JsonPlusSerializer, value: Any) -> bytes:
    try:
        kind, data = serde.dumps_typed(value)
        if kind != "msgpack":
            raise DeepAgentsWorkflowError("Unsupported graph checkpoint encoding.")
        _check_restored_types(value, serde.loads_typed((kind, data)))
        return data
    except DeepAgentsWorkflowError:
        raise
    except Exception as exc:
        raise DeepAgentsWorkflowError(
            "Graph checkpoint state cannot be restored losslessly."
        ) from exc


def _untracked_values(
    graph: Any, values: Mapping[str, Any], serde: JsonPlusSerializer
) -> dict[str, Any]:
    carried = {}
    for key, channel in graph.channels.items():
        if not isinstance(channel, UntrackedValue) or key not in values:
            continue
        try:
            data = _round_trip(serde, values[key])
        except (DeepAgentsWorkflowError, TypeError, ValueError):
            continue
        if len(data) <= _MAX_UNTRACKED_BYTES:
            carried[key] = values[key]
    return carried


def _managed_budget(graph: Any, offset: dict[str, int]) -> Any:
    """Keep root managed step values without reducing nested graphs' limits."""
    remaining = get_args(RemainingSteps)[1]
    last = get_args(IsLastStep)[1]

    class Remaining(ManagedValue[int]):
        @staticmethod
        def get(scratchpad: Any) -> int:
            return remaining.get(scratchpad) - offset["steps"]

    class Last(ManagedValue[bool]):
        @staticmethod
        def get(scratchpad: Any) -> bool:
            return Remaining.get(scratchpad) == 1

    return graph.copy(
        {
            "channels": {
                key: Remaining
                if channel is remaining
                else Last
                if channel is last
                else channel
                for key, channel in graph.channels.items()
            }
        }
    )


async def run(
    agent: Any,
    input: Any,
    *,
    continue_as_new_after: int | None,
    state_snapshot: Mapping[str, Any] | None,
    continue_as_new_input: Any = _UNSET,
) -> Any:
    """Drive public static breakpoints; carry only a settled root checkpoint."""
    carried = state_snapshot.get(SNAPSHOT_KEY) if state_snapshot else None
    if state_snapshot and carried is None:
        input = _merge_files(_merge_snapshot(input, state_snapshot), state_snapshot)
    if not isinstance(agent, CompiledStateGraph) or "model" not in agent.nodes:
        if carried is not None:
            raise DeepAgentsWorkflowError(
                "A graph checkpoint requires a compiled Deep Agent."
            )
        warnings.warn(
            "run_deep_agent cannot automatically continue this agent as new; it must expose a compiled graph with a model node.",
            stacklevel=2,
        )
        return await agent.ainvoke(input)

    automatic = (
        not agent.interrupt_before_nodes
        and not agent.interrupt_after_nodes
        and _rollover_allowed()
    )
    carry_input = (
        _carry_input(input)
        if continue_as_new_input is _UNSET
        else continue_as_new_input
    )
    if automatic:
        if isinstance(input, Command) and continue_as_new_input is _UNSET:
            raise DeepAgentsWorkflowError(
                "Command resumes require continue_as_new_input matching the workflow's first argument."
            )
        _validate_workflow_signature(carry_input)
    name = agent.get_name()
    if automatic and name == "LangGraph":
        raise DeepAgentsWorkflowError(
            "run_deep_agent requires a unique, stable agent name; pass name= to create_deep_agent or compile."
        )
    try:
        input = copy.deepcopy(input)
    except Exception as exc:
        raise DeepAgentsWorkflowError(
            "Deep Agent input cannot be copied for checkpoint execution."
        ) from exc
    serde = _serializer(agent)
    owned = agent.checkpointer is None
    if not owned and not isinstance(agent.checkpointer, InMemorySaver):
        raise DeepAgentsWorkflowError(
            "run_deep_agent requires an InMemorySaver or no checkpointer."
        )
    # Use schema-declared types even with LangGraph's strict-msgpack default.
    saver = _DriverSaver(
        serde=serde, backend=cast(InMemorySaver | None, agent.checkpointer)
    )
    config: RunnableConfig = agent.config.copy() if agent.config is not None else {}
    configurable = dict(config.get("configurable") or {})
    if configurable.get("checkpoint_ns"):
        raise DeepAgentsWorkflowError("run_deep_agent must drive the root agent graph.")
    if not owned and "thread_id" not in configurable and carried is None:
        raise DeepAgentsWorkflowError(
            "A caller-supplied InMemorySaver requires configurable.thread_id."
        )
    if "thread_id" in configurable:
        configurable["thread_id"] = str(configurable["thread_id"])
    shape = {
        "name": name,
        "nodes": sorted(agent.nodes),
        "channels": sorted(agent.channels),
    }
    budget = config.get("recursion_limit")
    if budget is not None and budget < 1:
        raise DeepAgentsWorkflowError("recursion_limit must be at least 1.")
    untracked: dict[str, Any] = {}
    if carried is not None:
        if not isinstance(carried, Mapping) or carried.get("version") != 1:
            raise DeepAgentsWorkflowError(
                "Unsupported Deep Agents checkpoint snapshot."
            )
        if carried.get("graph") != shape:
            raise CheckpointMismatchError(
                "The graph's nodes, state channels or name changed since its checkpoint. Use a matching worker version."
            )
        restored: Mapping[str, Any] = carried
        try:
            if input is not None and _input_digest(input) != restored["input_digest"]:
                raise DeepAgentsWorkflowError(
                    "A checkpoint snapshot resumes one unfinished invocation; do not combine it with new input."
                )
            thread_id = str(restored["thread_id"])
            configurable.setdefault("thread_id", thread_id)
            config["configurable"] = configurable
            if await saver.aget_tuple(config) is not None:
                raise DeepAgentsWorkflowError(
                    "Restore a checkpoint snapshot only once, into an empty thread."
                )
            data = base64.b64decode(restored["checkpoint"], validate=True)
            if carried.get("compression") == "zlib":
                data = zlib.decompress(data)
            checkpoint = serde.loads_typed(("msgpack", data))
            metadata = {
                key: value
                for key, value in restored["metadata"].items()
                if key in _METADATA_KEYS
            }
            step = metadata["step"]
            budget = restored["remaining_steps"]
            config["recursion_limit"] = restored["recursion_limit"]
            if (
                not isinstance(config["recursion_limit"], int)
                or config["recursion_limit"] < 1
            ):
                raise ValueError("Invalid original recursion limit.")
            saver.recursion_limit = config["recursion_limit"]
            if not isinstance(step, int) or not isinstance(budget, int) or budget < 0:
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
            raw_untracked = restored.get("untracked")
            decoded_untracked = (
                serde.loads_typed(
                    ("msgpack", base64.b64decode(raw_untracked, validate=True))
                )
                if raw_untracked
                else {}
            )
            if not isinstance(decoded_untracked, dict):
                raise ValueError("Invalid invocation state.")
            untracked = decoded_untracked
        except (KeyError, TypeError, ValueError, zlib.error) as exc:
            raise DeepAgentsWorkflowError(
                "Invalid Deep Agents checkpoint snapshot."
            ) from exc
        configurable.pop("checkpoint_id", None)
        checkpoint["id"] = str(checkpoint_ids.uuid6(clock_seq=step))
        await saver.aput(
            {
                "configurable": {
                    "thread_id": configurable["thread_id"],
                    "checkpoint_ns": "",
                }
            },
            cast(Checkpoint, cast(object, checkpoint)),
            cast(CheckpointMetadata, cast(object, metadata)),
            checkpoint["channel_versions"],
        )
        input = Command(update=untracked) if untracked else None
    elif owned:
        configurable["thread_id"] = str(workflow.uuid4())
    config["configurable"] = configurable
    graph = agent.copy({"checkpointer": saver, "config": None})
    if not _can_resume_interrupts(agent) and _requires_interrupts(agent):
        raise DeepAgentsWorkflowError(
            "Application interrupts require an explicit InMemorySaver and configurable.thread_id so the caller can resume the agent."
        )
    if not automatic:
        return await graph.ainvoke(input, config)

    initial = await saver.aget_tuple(config)
    resuming = (
        carried is None
        and initial is not None
        and (input is None or isinstance(input, Command))
    )
    if resuming and budget is not None:
        # Native resumes have no input superstep, leaving one additional step.
        budget += 1
    managed_offset = {"steps": 0}
    graph = _managed_budget(graph, managed_offset)
    previous_step = initial.metadata.get("step", -1) if initial else -1
    if initial is not None and input is not None and not isinstance(input, Command):
        previous_step += 1
    next_nodes = (await graph.aget_state(initial.config)).next if initial else ()
    progressed = carried is None and initial is not None
    keys = (
        list(graph.output_channels)
        if not isinstance(graph.output_channels, str)
        else [graph.output_channels]
    )
    untracked_keys = [
        key
        for key, channel in graph.channels.items()
        if isinstance(channel, UntrackedValue)
    ]
    state_keys = (
        [graph.stream_channels]
        if isinstance(graph.stream_channels, str)
        else list(graph.stream_channels or keys)
    )
    output_keys = list(dict.fromkeys([*state_keys, *untracked_keys]))
    while True:
        if budget is not None and hasattr(saver, "recursion_limit"):
            managed_offset["steps"] = saver.recursion_limit - budget + 1
        if budget is not None and budget <= 0:
            raise DeepAgentsWorkflowError("Deep Agent exceeded its recursion limit.")
        values: dict[str, Any] = {}
        interrupts: list[Any] = []
        try:
            # Public breakpoints end the stream normally, including callbacks.
            # Exit durability avoids retaining every nested subagent checkpoint.
            stream = cast(
                AsyncIterator[tuple[str, Any]],
                graph.astream(
                    input,
                    config,
                    stream_mode=["values", "updates"],
                    output_keys=output_keys,
                    interrupt_before="*",
                    durability="exit" if owned else None,
                ),
            )
            async for mode, value in stream:
                if mode == "values":
                    values = value
                elif "__interrupt__" in value:
                    interrupts.extend(value["__interrupt__"])
        except GraphRecursionError as exc:
            raise DeepAgentsWorkflowError(
                "Deep Agent exceeded its recursion limit."
            ) from exc
        configurable.pop("checkpoint_id", None)
        saved = await saver.aget_tuple(config)
        if saved is None:
            raise DeepAgentsWorkflowError("Graph checkpoint metadata is missing.")
        step = saved.metadata["step"]
        if budget is None:
            budget = saver.recursion_limit + (1 if resuming else 0)
        if budget is not None:
            budget -= step - previous_step
            if budget < 0:
                raise DeepAgentsWorkflowError(
                    "Deep Agent exceeded its recursion limit."
                )
        previous_step = step
        state = await graph.aget_state(saved.config)
        progressed = progressed or "model" in next_nodes
        next_nodes = state.next
        result = (
            values.get(keys[0])
            if isinstance(graph.output_channels, str)
            else {key: value for key, value in values.items() if key in keys}
        )
        if interrupts:
            if not _can_resume_interrupts(agent):
                raise DeepAgentsWorkflowError(
                    "Application interrupts require an explicit InMemorySaver and configurable.thread_id so the caller can resume the agent."
                )
            return {
                **(result if isinstance(result, dict) else {}),
                "__interrupt__": interrupts,
            }
        if not state.next:
            return result
        # Materialize tracked state and validate before the first paid model
        # call, even when this execution never needs a rollover.
        checkpoint = {
            **saved.checkpoint,
            "channel_values": {
                **saved.checkpoint["channel_values"],
                **{
                    key: value
                    for key, value in values.items()
                    if key not in untracked_keys
                },
            },
        }
        data = _round_trip(serde, checkpoint)
        at_model = state.next == ("model",)
        if at_model:
            past_limit = (
                workflow.info().is_continue_as_new_suggested()
                if continue_as_new_after is None
                else workflow.info().get_current_history_length()
                >= continue_as_new_after
            )
            if progressed and past_limit and _rollover_allowed():
                if saved.pending_writes:
                    raise DeepAgentsWorkflowError(
                        "Cannot carry an unsettled graph checkpoint."
                    )
                snapshot = {
                    SNAPSHOT_KEY: {
                        "version": 1,
                        "graph": shape,
                        "thread_id": configurable["thread_id"],
                        "checkpoint": base64.b64encode(zlib.compress(data)).decode(
                            "ascii"
                        ),
                        "compression": "zlib",
                        "metadata": {
                            key: value
                            for key, value in saved.metadata.items()
                            if key in _METADATA_KEYS
                        },
                        "remaining_steps": budget,
                        "recursion_limit": saver.recursion_limit,
                        "input_digest": _input_digest(carry_input),
                        "untracked": base64.b64encode(
                            _round_trip(serde, _untracked_values(graph, values, serde))
                        ).decode("ascii"),
                    },
                    "messages": [
                        {
                            "role": "__temporal_checkpoint_requires_new_worker__",
                            "content": "Checkpoint snapshots require a compatible Deep Agents worker.",
                        }
                    ],
                }
                try:
                    payloads = _converter().to_payloads([carry_input, snapshot])
                except Exception as exc:
                    raise DeepAgentsWorkflowError(
                        "Continue-as-new input is not serializable."
                    ) from exc
                if sum(payload.ByteSize() for payload in payloads) > _MAX_CARRY_BYTES:
                    raise DeepAgentsWorkflowError(
                        "Deep Agents checkpoint exceeds the continue-as-new payload limit. Reduce state or store large files in TemporalBackend."
                    )
                workflow.continue_as_new(args=[carry_input, snapshot])
            progressed = True
        # Keep every invocation-local value within this run. Only small,
        # losslessly serializable ones cross continue-as-new.
        untracked = {key: values[key] for key in untracked_keys if key in values}
        input = Command(update=untracked) if untracked else None
        configurable.pop("checkpoint_id", None)
