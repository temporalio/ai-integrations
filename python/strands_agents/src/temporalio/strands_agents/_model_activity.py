from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from strands.models import Model
from strands.types.session import decode_bytes_values, encode_bytes_values
from strands.types.streaming import StreamEvent

from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.strands_agents._heartbeat_decorator import auto_heartbeater


# Fields are typed as Any because strands TypedDicts (Message, ToolSpec) use
# NotRequired, which Python < 3.11's get_type_hints leaks through unchanged
# and the default JSON converter then fails to deserialize. Values flow
# through unchanged to ``Model.stream`` which accepts the raw dicts.
#
# Messages can hold ``bytes`` (image and document sources), which the JSON
# payload converter cannot carry: it fails on non-UTF-8 bytes and turns ASCII
# bytes into ``str``. ``__post_init__`` therefore tags every ``bytes`` value as
# base64 with Strands' own ``encode_bytes_values``, and ``_stream`` restores
# them before the model sees the messages. Messages without bytes encode to
# the same JSON as before.
@dataclass
class _InvokeModelInput:
    model_name: str | None
    messages: Any
    invocation_state: dict[str, Any] = field(default_factory=dict)
    tool_specs: Any = None
    system_prompt: str | None = None
    tool_choice: Any = None
    system_prompt_content: Any = None

    def __post_init__(self) -> None:
        self.messages = encode_bytes_values(self.messages)


@dataclass
class _StreamingInvokeModelInput(_InvokeModelInput):
    streaming_topic: str = ""
    streaming_batch_interval_seconds: float = 0.1


class ModelActivity:
    """Holds the registered model factories and exposes the model activities."""

    def __init__(
        self,
        factories: dict[str, Callable[[], Model]],
        *,
        default_name: str | None = None,
    ) -> None:
        """Store the factories; models are constructed lazily on first use.

        ``default_name`` is set only by the plugin's own auto-registered
        ``BedrockModel`` default. User-supplied ``models`` leave it ``None``,
        which forces every ``TemporalAgent`` to specify ``model=`` explicitly.
        """
        self._factories = factories
        self._default_name = default_name
        self._models: dict[str, Model] = {}

    def _get_model(self, name: str | None) -> Model:
        if name is None:
            if self._default_name is None:
                raise ValueError(
                    f"TemporalAgent was constructed without an explicit `model`, "
                    f"but the plugin was configured with user-supplied `models=`. "
                    f"Pass model='...' to TemporalAgent. "
                    f"Known: {sorted(self._factories)}"
                )
            name = self._default_name
        if name not in self._models:
            if name not in self._factories:
                raise ValueError(
                    f"Unknown model name {name!r}. Known: {sorted(self._factories)}"
                )
            self._models[name] = self._factories[name]()
        return self._models[name]

    @activity.defn
    @auto_heartbeater
    async def invoke_model(self, input: _InvokeModelInput) -> list[dict[str, Any]]:
        """Run the named model and return its stream events as a list.

        Events that carry ``bytes`` (``redactedContent``) are returned with
        those values tagged as base64; ``TemporalModel`` restores them. The
        result is typed as plain dicts because the tagged form would not
        validate against ``StreamEvent``, whose ``redactedContent`` is ``bytes``.
        """
        model = self._get_model(input.model_name)
        return [encode_bytes_values(event) async for event in _stream(model, input)]

    @activity.defn
    @auto_heartbeater
    async def invoke_model_streaming(
        self, input: _StreamingInvokeModelInput
    ) -> list[dict[str, Any]]:
        """Run the named model and publish each stream event to a WorkflowStream."""
        model = self._get_model(input.model_name)
        events: list[dict[str, Any]] = []
        stream = WorkflowStreamClient.from_within_activity(
            batch_interval=timedelta(seconds=input.streaming_batch_interval_seconds),
        )
        topic = stream.topic(input.streaming_topic)
        async with stream:
            async for event in _stream(model, input):
                # Subscribers receive the same JSON-safe form the workflow gets.
                encoded = encode_bytes_values(event)
                events.append(encoded)
                topic.publish(encoded)
        return events


def decode_event(event: dict[str, Any]) -> StreamEvent:
    """Restore the ``bytes`` that ``invoke_model`` tagged in a returned stream event."""
    return decode_bytes_values(event)  # type: ignore[no-any-return]


def _stream(model: Model, input: _InvokeModelInput) -> AsyncIterable[StreamEvent]:
    return model.stream(
        decode_bytes_values(input.messages),
        input.tool_specs,
        input.system_prompt,
        tool_choice=input.tool_choice,
        system_prompt_content=input.system_prompt_content,
        invocation_state=input.invocation_state,
    )
