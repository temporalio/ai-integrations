"""The workflow-side chat client: every model round trip is an Activity."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from datetime import timedelta
from typing import Any, ClassVar

from agent_framework import (
    BaseChatClient,
    ChatAndFunctionMiddlewareTypes,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    CompactionStrategy,
    FunctionInvocationConfiguration,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
    TokenizerProtocol,
)

from temporalio import workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.microsoft_agent_framework import _serde
from temporalio.microsoft_agent_framework._activities import (
    INVOKE_MODEL,
    INVOKE_MODEL_STREAMING,
    InvokeModelInput,
    InvokeModelStreamingInput,
)
from temporalio.microsoft_agent_framework.workflow import (
    MicrosoftAgentFrameworkWorkflowError,
)
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    VersioningIntent,
)


class TemporalChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):  # type: ignore[type-arg]
    """An Agent Framework chat client that runs each model call as a Temporal Activity.

    Pass it where you would pass a provider client:
    ``Agent(client=TemporalChatClient("gpt-4o"), ...)`` or
    ``TemporalChatClient("gpt-4o").as_agent(...)``. ``model`` names a factory
    registered with ``MicrosoftAgentFrameworkPlugin(chat_clients=...)`` on the worker;
    the workflow carries only that name. The function-invocation loop, chat and
    function middleware, approvals and compaction run in the workflow as they
    do natively; each provider round trip runs as the
    ``microsoft_agent_framework.invoke_model`` Activity (``microsoft_agent_framework.invoke_model_streaming``
    for ``stream=True``) with the activity options given here. ``summary``
    defaults to the model name; the default timeouts are a 5 minute
    start-to-close and a 10 second heartbeat, and the Activity heartbeats on
    its own at half the heartbeat timeout. A run's ``client_kwargs`` (such as
    ``extra_headers``) cross to the Activity and reach the provider client;
    they are recorded in Event History, so credentials stay in the factory,
    and an ``extra_headers`` entry named ``authorization``, ``x-api-key`` or
    ``api-key`` (in any case) fails the run with
    ``MicrosoftAgentFrameworkWorkflowError``. Unless the caller sets ``store``, the options reach the provider with
    ``store=False``, so the provider keeps no conversation of its own and the
    session's in-memory history is the one transcript; ``store=True`` in the
    agent's or the run's options keeps the service-side conversation path.

    With ``streaming_topic`` set, every call of this client runs as the
    streaming Activity, with or without ``stream=True``, and publishes each
    update (as its dict form) to that topic on the workflow's
    ``WorkflowStream``, batched at ``streaming_batch_interval`` or 256 updates. ``middleware``,
    ``function_invocation_configuration``, ``compaction_strategy`` and
    ``tokenizer`` are the framework's own client options. A synchronous
    ``def`` tool runs inline on the workflow task (the plugin's worker
    runtime stands in for the thread hand-off the framework would use).

    The client only works inside a workflow; calling it elsewhere raises
    ``RuntimeError``. Construction itself does no I/O.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "temporal"
    STORES_BY_DEFAULT: ClassVar[bool] = False

    def __init__(
        self,
        model: str,
        *,
        task_queue: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = timedelta(minutes=5),
        heartbeat_timeout: timedelta | None = timedelta(seconds=10),
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
        versioning_intent: VersioningIntent | None = None,
        summary: str | None = None,
        priority: Priority = Priority.default,
        streaming_topic: str | None = None,
        streaming_batch_interval: timedelta = timedelta(milliseconds=100),
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration
        | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> None:
        """Configure the model name, activity options and streaming settings.

        Raises:
            MicrosoftAgentFrameworkWorkflowError: If ``model`` is not a non-empty string.
        """
        if not isinstance(model, str) or not model:
            raise MicrosoftAgentFrameworkWorkflowError(
                f"TemporalChatClient needs the name of a chat client registered with "
                f"MicrosoftAgentFrameworkPlugin(chat_clients=...), got {model!r}. Credentials and "
                "provider clients live in the worker's factories, not in workflow code."
            )
        super().__init__(
            middleware=middleware,
            function_invocation_configuration=function_invocation_configuration,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
        )
        self._model = model
        self._streaming_topic = streaming_topic
        self._streaming_batch_interval = streaming_batch_interval
        self._activity_options = ActivityConfig(
            task_queue=task_queue,
            schedule_to_close_timeout=schedule_to_close_timeout,
            schedule_to_start_timeout=schedule_to_start_timeout,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy,
            cancellation_type=cancellation_type,
            versioning_intent=versioning_intent,
            summary=summary if summary is not None else model,
            priority=priority,
        )

    @property
    def model(self) -> str:
        """The name of the worker-side chat client this client dispatches to."""
        return self._model

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        if not workflow.in_workflow():
            raise RuntimeError(
                "TemporalChatClient only runs inside a Temporal workflow; outside one, "
                "use the provider's chat client directly."
            )
        # What arrives here is the run's client_kwargs, which the provider
        # client reads (the OpenAI client takes extra_headers from them).
        dumped_kwargs = _serde.dump_kwargs(kwargs)
        dumped_messages = _serde.dump_messages(messages)
        dumped_options = _serde.dump_options(options)
        if dumped_options.get("store") is None:
            # The Agent decides who keeps the transcript from an explicit store
            # only; left unset, a provider that stores by default (the OpenAI
            # Responses API) would return a conversation id, and the next call
            # would carry that id and the session's full history both.
            dumped_options["store"] = False
        if stream or self._streaming_topic is not None:
            streaming_input = InvokeModelStreamingInput(
                model=self._model,
                messages=dumped_messages,
                options=dumped_options,
                kwargs=dumped_kwargs,
                streaming_topic=self._streaming_topic,
                streaming_batch_interval_seconds=self._streaming_batch_interval.total_seconds(),
            )
            if stream:
                return ResponseStream(
                    self._stream_updates(streaming_input),
                    finalizer=lambda updates: ChatResponse.from_updates(
                        updates, output_format_type=options.get("response_format")
                    ),
                )
            # A topic publishes on every call of this client, so a call without
            # stream=True runs the streaming Activity too and is finalized here.
            return self._get_response_from_stream(
                streaming_input, options.get("response_format")
            )
        return self._get_response(
            InvokeModelInput(
                model=self._model,
                messages=dumped_messages,
                options=dumped_options,
                kwargs=dumped_kwargs,
            ),
            options.get("response_format"),
        )

    async def _get_response(
        self, input: InvokeModelInput, response_format: Any
    ) -> ChatResponse:
        raw = await workflow.execute_activity(
            INVOKE_MODEL,
            input,
            result_type=dict[str, Any],
            **self._activity_options,
        )
        # The format does not cross the boundary as a class, so the rebuilt
        # response gets it here and parses ``value`` like the streamed path.
        return _serde.load_response(raw, response_format)

    async def _get_response_from_stream(
        self, input: InvokeModelStreamingInput, response_format: Any
    ) -> ChatResponse:
        updates = [update async for update in self._stream_updates(input)]
        return ChatResponse.from_updates(updates, output_format_type=response_format)

    async def _stream_updates(
        self, input: InvokeModelStreamingInput
    ) -> AsyncIterator[ChatResponseUpdate]:
        raw = await workflow.execute_activity(
            INVOKE_MODEL_STREAMING,
            input,
            result_type=list[dict[str, Any]],
            **self._activity_options,
        )
        for item in raw:
            yield _serde.load_update(item)
