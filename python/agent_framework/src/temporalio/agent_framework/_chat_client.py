"""The workflow-side chat client: every model round trip is an Activity."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
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
    FunctionInvocationContext,
    FunctionInvocationLayer,
    FunctionMiddleware,
    Message,
    ResponseStream,
    TokenizerProtocol,
)

from temporalio import workflow
from temporalio.agent_framework import _serde
from temporalio.agent_framework._activities import (
    INVOKE_MODEL,
    INVOKE_MODEL_STREAMING,
    InvokeModelInput,
    InvokeModelStreamingInput,
)
from temporalio.agent_framework.workflow import AgentFrameworkWorkflowError
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    VersioningIntent,
)


class _SyncToolsOnWorkflowLoop(FunctionMiddleware):
    """Run synchronous tools on the workflow event loop.

    Agent Framework runs a ``def`` tool with ``asyncio.to_thread``, which the
    workflow event loop does not implement, so the tool would fail with
    ``NotImplementedError``. ``_invoke_sync_on_event_loop`` is the framework's
    own switch for running it inline; its harness sets it the same way.
    """

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        context.function._invoke_sync_on_event_loop = True
        await call_next()


class TemporalChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):  # type: ignore[type-arg]
    """An Agent Framework chat client that runs each model call as a Temporal Activity.

    Pass it where you would pass a provider client:
    ``Agent(client=TemporalChatClient("gpt-4o"), ...)`` or
    ``TemporalChatClient("gpt-4o").as_agent(...)``. ``model`` names a factory
    registered with ``AgentFrameworkPlugin(chat_clients=...)`` on the worker;
    the workflow carries only that name. The function-invocation loop, chat and
    function middleware, approvals and compaction run in the workflow as they
    do natively; each provider round trip runs as the
    ``agent_framework.invoke_model`` Activity (``agent_framework.invoke_model_streaming``
    for ``stream=True``) with the activity options given here. ``summary``
    defaults to the model name; the default timeouts are a 5 minute
    start-to-close and a 10 second heartbeat, which the Activity heartbeats
    automatically.

    With ``streaming_topic`` set, a streamed call publishes each update (as its
    dict form) to that topic on the workflow's ``WorkflowStream``, batched at
    ``streaming_batch_interval``. ``middleware``,
    ``function_invocation_configuration``, ``compaction_strategy`` and
    ``tokenizer`` are the framework's own client options.

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
            AgentFrameworkWorkflowError: If ``model`` is not a non-empty string.
        """
        if not isinstance(model, str) or not model:
            raise AgentFrameworkWorkflowError(
                f"TemporalChatClient needs the name of a chat client registered with "
                f"AgentFrameworkPlugin(chat_clients=...), got {model!r}. Credentials and "
                "provider clients live in the worker's factories, not in workflow code."
            )
        super().__init__(
            middleware=[_SyncToolsOnWorkflowLoop(), *(middleware or ())],
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
        if kwargs:
            # The layers above pass their own plumbing here (for example the
            # session); none of it is part of the provider request.
            workflow.logger.debug(
                "Not sending keyword arguments %s to the model Activity", sorted(kwargs)
            )
        dumped_messages = _serde.dump_messages(messages)
        dumped_options = _serde.dump_options(options)
        if stream:
            streaming_input = InvokeModelStreamingInput(
                model=self._model,
                messages=dumped_messages,
                options=dumped_options,
                streaming_topic=self._streaming_topic,
                streaming_batch_interval_seconds=self._streaming_batch_interval.total_seconds(),
            )
            return ResponseStream(
                self._stream_updates(streaming_input),
                finalizer=lambda updates: ChatResponse.from_updates(
                    updates, output_format_type=options.get("response_format")
                ),
            )
        return self._get_response(
            InvokeModelInput(
                model=self._model, messages=dumped_messages, options=dumped_options
            )
        )

    async def _get_response(self, input: InvokeModelInput) -> ChatResponse:
        raw = await workflow.execute_activity(
            INVOKE_MODEL,
            input,
            result_type=dict[str, Any],
            **self._activity_options,
        )
        return _serde.load_response(raw)

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
