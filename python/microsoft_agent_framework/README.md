# Microsoft Agent Framework

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/microsoft-agent-framework).

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs a [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) `Agent` inside a Temporal Workflow. Each model round trip runs as a Temporal Activity with its own timeout and retries. The agent's own function-invocation loop, tools, middleware, context providers and approvals run in the Workflow, where Temporal records every step and replays it after a crash or a worker restart. You keep writing the agent the way Agent Framework documents it; the one change is the chat client.

## Installation

```sh
uv add temporalio-microsoft-agent-framework
```

The plugin depends on `agent-framework-core` (1.23.0 or newer, below 2; tested with 1.23.0 through 1.26.0). Add the provider package you use, for example `agent-framework-openai`. Replay safety rests on a few private names of the framework, so a worker or replayer fails at start with a message naming the name and the tested range if a release moves one of them.

## Quickstart

`workflow.py` defines the workflow and runs the worker:

```python
import asyncio
from datetime import timedelta

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI
from temporalio import workflow
from temporalio.microsoft_agent_framework import MicrosoftAgentFrameworkPlugin, TemporalChatClient
from temporalio.client import Client
from temporalio.worker import Worker


@workflow.defn
class HelloWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            client=TemporalChatClient("gpt-4o", start_to_close_timeout=timedelta(seconds=60)),
            instructions="You are a concise assistant.",
        )
        response = await agent.run(prompt)
        return response.text


async def main() -> None:
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="microsoft-agent-framework",
        workflows=[HelloWorkflow],
        plugins=[
            MicrosoftAgentFrameworkPlugin(
                chat_clients={
                    "gpt-4o": lambda: OpenAIChatClient(
                        model="gpt-4o", async_client=AsyncOpenAI(max_retries=0)
                    ),
                }
            )
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

`client.py` starts the workflow:

```python
import asyncio

from temporalio.client import Client

from workflow import HelloWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        HelloWorkflow.run,
        "Hello",
        id="microsoft-agent-framework-quickstart",
        task_queue="microsoft-agent-framework",
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

The `AsyncOpenAI` client built in the factory reads `OPENAI_API_KEY` on the worker. The workflow only carries the name `"gpt-4o"`.

What runs where: the `Agent`, its loop, your tools, middleware, context providers and approval handling run in the Workflow. Each provider round trip runs as the `microsoft_agent_framework.invoke_model` Activity (`microsoft_agent_framework.invoke_model_streaming` when you stream). The messages and chat options cross to the Activity in their `to_dict()` form, tools cross as their JSON schema specs, and the `ChatResponse` comes back the same way; provider SDK objects (`raw_representation`) never enter history.

## Chat clients and credentials

`MicrosoftAgentFrameworkPlugin(chat_clients=...)` maps a model name to a factory that builds the provider's chat client. The factory runs on the worker, on first use, and its client is cached for the worker's lifetime, so credentials and connections stay on the worker. `TemporalChatClient("name")` in the workflow selects the factory and carries the activity options for that agent's model calls. Register as many names as you have models:

```python
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI


@workflow.defn
class TwoModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        fast = Agent(client=TemporalChatClient("fast"), instructions="Draft.")
        careful = Agent(client=TemporalChatClient("careful"), instructions="Review the draft.")
        draft = await fast.run(prompt)
        return (await careful.run(draft.text)).text


CHAT_CLIENTS = {
    "fast": lambda: OpenAIChatClient(
        model="gpt-4o-mini", async_client=AsyncOpenAI(max_retries=0)
    ),
    "careful": lambda: OpenAIChatClient(
        model="gpt-4o", async_client=AsyncOpenAI(max_retries=0)
    ),
}
plugin = MicrosoftAgentFrameworkPlugin(chat_clients=CHAT_CLIENTS)
```

A name the worker does not know fails the Activity with a non-retryable `ApplicationError` of type `MicrosoftAgentFrameworkUnknownChatClient`, so the workflow fails instead of waiting. A factory that raises (for example `OpenAIChatClient` without `OPENAI_API_KEY` on the worker) or returns `None` fails the Activity the same way, with type `MicrosoftAgentFrameworkChatClientFactoryError` and a message naming the model and the cause; fix the worker and start the workflow again. Headers for hosted MCP tools stay on the worker too, in `hosted_mcp_headers` (see Tools). During a rolling deployment, add the name to every worker on the task queue before workflows start using it.

`TemporalChatClient("gpt-4o").as_agent(instructions=..., tools=...)` is the short form of `Agent(client=TemporalChatClient("gpt-4o"), ...)`. Calling a `TemporalChatClient` outside a workflow raises `RuntimeError`; use the provider's client there.

A run's `client_kwargs`, such as `extra_headers` for the OpenAI client, cross to the Activity and reach the provider client. A value that is not JSON-compatible fails the workflow with `MicrosoftAgentFrameworkWorkflowError`.

## Retries and timeouts

Each `TemporalChatClient` carries the activity options of its model calls: `start_to_close_timeout` (default 5 minutes), `heartbeat_timeout` (default 10 seconds; the Activity heartbeats on its own at half that interval, so a dead worker is noticed within seconds), `schedule_to_close_timeout`, `retry_policy`, `task_queue`, `summary` (defaults to the model name), `schedule_to_start_timeout`, `cancellation_type`, `versioning_intent` and `priority`. Give reasoning models a longer `start_to_close_timeout`.

```python
from temporalio.common import RetryPolicy

chat_client = TemporalChatClient(
    "gpt-4o",
    start_to_close_timeout=timedelta(minutes=10),
    retry_policy=RetryPolicy(maximum_attempts=3),
)
```

Temporal owns retries, so turn the provider SDK's own retries off: `OpenAIChatClient` does not expose `max_retries`, so pass `async_client=AsyncOpenAI(max_retries=0)` as the quickstart does. A factory whose client still retries (`max_retries > 0`) gets one warning in the worker log the first time it runs.

Provider errors are classified by HTTP status. Agent Framework wraps the SDK error in `ChatClientException`; the Activity walks the cause chain for a `status_code`. 408, 409, 429 and 5xx stay retryable and honor `retry-after-ms`, `retry-after` and `x-should-retry` headers through the next retry delay; other 4xx are non-retryable; errors without a status (connection failures) stay retryable. `retry-after-ms` and `retry-after` parse as numbers, fractions included, or as an HTTP-date; a value that does not parse falls through to the next form; a hint above 60 seconds is capped at 60 seconds, and a past date leaves the retry policy's own backoff in charge. The failure reaches the workflow as an `ActivityError` whose cause is an `ApplicationError` with `type` set to the SDK's exception class name, for example `RateLimitError` or `AuthenticationError`. Match on `type`, not on the message.

## Tools

Decide per tool where it runs. Pure logic stays in the Workflow as a plain `@tool`; anything that does I/O or reads a clock belongs in an Activity. The plugin cannot tell them apart, so the choice is yours: decorate the I/O function with `@activity.defn`, register it on the worker, and pass it to the agent with `activity_as_tool`:

```python
from agent_framework import tool
from temporalio import activity
from temporalio.microsoft_agent_framework.workflow import activity_as_tool


@activity.defn
async def fetch_user(user_id: str) -> dict:
    """Fetch a user record."""
    ...


@tool
def shout(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


@workflow.defn
class ToolsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            client=TemporalChatClient("gpt-4o"),
            tools=[
                shout,
                activity_as_tool(fetch_user, start_to_close_timeout=timedelta(seconds=30)),
            ],
        )
        return (await agent.run(prompt)).text


async def run_worker(client: Client) -> None:
    worker = Worker(
        client,
        task_queue="microsoft-agent-framework",
        workflows=[ToolsWorkflow],
        activities=[fetch_user],
        plugins=[plugin],
    )
    await worker.run()
```

`activity_as_tool` takes a function or a method passed as `Class.method` (register the bound method on the worker). Agent Framework infers the tool's parameter schema from the activity's signature, so `Annotated[str, "description"]` parameters and docstrings work as they do for `@tool`. The arguments bind by name and reach the activity by position. `approval_mode="always_require"` works on an activity tool like on any other. A failed activity, including one whose retries ran out, raises its failure cause from the tool, which Agent Framework turns into an error result for the model; cancellation propagates as `asyncio.CancelledError`. Like `workflow.execute_activity`, `activity_as_tool` needs `start_to_close_timeout` or `schedule_to_close_timeout`; without either it raises `ValueError` when the tool is built, not when the model calls it. Agent Framework's loop settings apply unchanged: tool calls of one response run concurrently by default (`allow_concurrent_invocation=True`), a run makes at most `max_iterations` (40) model calls before one last call with `tool_choice="none"`, and after three consecutive rounds of failing tools the loop stops calling tools and asks for a final answer. Set them on `TemporalChatClient(function_invocation_configuration={...})`.

Synchronous `def` tools run inline on the workflow task. Agent Framework hands a `def` tool to `asyncio.to_thread`, which the workflow event loop does not implement, so the plugin's worker runtime rebinds that name in the framework's tool module to a stand-in that calls the function directly inside a workflow and still uses a thread anywhere else. The tool object itself is not changed, so one shared with an Activity or a plain agent in the same process keeps its normal behavior there. A tool that blocks for long blocks the workflow task, so keep sync tools quick.

Declaration-only tools (`FunctionTool(func=None, ...)`) are advertised to the model and the run returns the model's call to you, as in Agent Framework. Hosted tools passed as mappings, such as the provider's web search, pass through to the model Activity and run on the provider's side. A hosted MCP tool (`OpenAIChatClient.get_mcp_tool(...)`) must not carry `headers` or `authorization` in workflow code: the mapping is recorded in history, so the run fails with `MicrosoftAgentFrameworkWorkflowError`. Register the headers on the worker instead, with `MicrosoftAgentFrameworkPlugin(hosted_mcp_headers={"github": {"Authorization": "Bearer ..."}})` keyed by the server label or URL (a callable returning the headers works too); the Activity adds them to the tool before the provider call. `agent.as_tool()` works unchanged: the sub-agent's model calls run as Activities in the same workflow.

A tool object the plugin cannot send to the model (anything that is not a `FunctionTool`, a `@tool` function or a mapping) fails the workflow with `MicrosoftAgentFrameworkWorkflowError`.

## MCP

MCP support runs on [`temporalio-mcp`](https://github.com/temporalio/ai-integrations/tree/main/python/mcp) and the MCP Python SDK v2. Install it with the `mcp` extra:

```sh
uv add "temporalio-microsoft-agent-framework[mcp]"
```

`MicrosoftAgentFrameworkPlugin(mcp_clients=...)` maps a server name to a factory returning an `mcp.Client`, the same shape as `chat_clients`. A factory that raises, or returns something other than an `mcp.Client`, fails the Activity with a non-retryable `ApplicationError` of type `MicrosoftAgentFrameworkMCPClientFactoryError`. The plugin registers the `temporalio.mcp.{name}.list-tools` and `temporalio.mcp.{name}.call-tool` Activities (and the other `temporalio-mcp` operations) for each server. Workflow-side, `TemporalMCPToolProvider("name")` is a context provider: before each agent run it lists the server's tools and adds one `FunctionTool` per tool, with the server's full input schema, whose call runs the `call-tool` Activity. It goes in `context_providers`, not `tools`, because Agent Framework drives an `MCPTool` in `tools=` through a live connection that cannot exist in a workflow; a context provider is the framework's public way to add tools to a run.

```python
from mcp import Client as MCPClient
from mcp import StdioServerParameters, stdio_client
from temporalio.microsoft_agent_framework import TemporalMCPToolProvider


@workflow.defn
class MCPWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            client=TemporalChatClient("gpt-4o"),
            context_providers=[
                TemporalMCPToolProvider("echo", start_to_close_timeout=timedelta(seconds=30))
            ],
        )
        return (await agent.run(prompt)).text


# worker
plugin = MicrosoftAgentFrameworkPlugin(
    chat_clients=CHAT_CLIENTS,
    mcp_clients={
        "echo": lambda: MCPClient(
            stdio_client(StdioServerParameters(command="python", args=["echo_server.py"]))
        ),
    },
)
```

`cache_tools=False` (the default) lists the tools on every run, so a server restarted with different tools is picked up; `cache_tools=True` lists once per provider instance. `tool_name_prefix`, `allowed_tools`, `approval_mode` and `parse_tool_results` work as on Agent Framework's `MCPTool`. Tool names are normalized the same way: a character outside `A-Za-z0-9_.-` becomes `-` and the separators between the prefix and the name are trimmed, so `files/read` reaches the model as `files-read` and prefix `gh_` gives `gh_files-read`. `allowed_tools` matches the server's own names. `approval_mode` is one mode for every tool of the server; the per-tool mapping `MCPTool` also takes (`always_require_approval`, `never_require_approval`) fails the workflow with `MicrosoftAgentFrameworkWorkflowError`, because `FunctionTool` would keep the dict and no tool would ask for approval. Tool results map to content the way `MCPTool` maps them (structured content as JSON text; text, images, audio, resource links and embedded resources to text or URI content; an empty result to the text `null`); a result the server marks as an error becomes an error result for the model, with the server's text in it when `include_detailed_errors` is on.

Tool names must be unique on an agent. A server tool with the name of one of the agent's own tools, or of a tool from another provider, fails the workflow with `MicrosoftAgentFrameworkWorkflowError` before the model call; give that provider a `tool_name_prefix` or an `allowed_tools` list. Tools passed to `agent.run(tools=...)` are merged after context providers run, so a collision with one of those raises Agent Framework's own `ValueError` and fails the workflow task instead; put such tools on the `Agent`.

The activity options on `TemporalMCPToolProvider` apply to its list and call Activities; `summary` defaults to the server name. Without `start_to_close_timeout` or `schedule_to_close_timeout`, `temporalio-mcp`'s one-minute default applies. The default retry policy retries without limit, so a run waits while its MCP server is down; bound it with `retry_policy` or `schedule_to_close_timeout`. A `call-tool` Activity that runs out of attempts becomes an error result for the model; a `list-tools` Activity that runs out of attempts raises from `agent.run`. A workflow cancelled during a call stops, as with `activity_as_tool`: the cancelled Activity reaches the tool as `asyncio.CancelledError`, which Agent Framework does not turn into a tool result. The provider sets no `heartbeat_timeout`; `temporalio-mcp` 0.3 does not heartbeat its Activities. The worker reuses one connection per server across Activities and closes it after `mcp_connection_idle_timeout` (default 5 minutes) of idleness; `None` keeps it until shutdown and `timedelta(0)` closes it after every Activity. Do not register the same server name through `MCPPlugin` on the same worker: the Activity names clash. A server name that no worker registered has no Activity to run, so the Activity attempt fails with a retryable `NotFoundError` and the run waits until a worker with that name polls the task queue.

## Approvals (human in the loop)

Approvals are Agent Framework's own: a tool with `approval_mode="always_require"` is not executed, the run returns with the request in `response.user_input_requests`, and you answer with `request.to_function_approval_response(approved)` on the same session. In a workflow the wait is a Signal, an Update or both, and a Query shows what is pending:

```python
from dataclasses import dataclass

from agent_framework import Agent, tool
from temporalio.exceptions import ApplicationError


@dataclass
class Decision:
    request_id: str
    approved: bool


@tool(approval_mode="always_require")
async def delete_record(record_id: str) -> str:
    """Delete a record. Needs approval."""
    return f"deleted {record_id}"


@workflow.defn
class ApprovalWorkflow:
    def __init__(self) -> None:
        self._pending: list[dict] = []
        self._decisions: dict[str, bool] = {}

    @workflow.signal
    def approve(self, decision: Decision) -> None:
        self._decisions[decision.request_id] = decision.approved

    @workflow.update
    async def decide(self, decision: Decision) -> str:
        if decision.request_id not in {request["id"] for request in self._pending}:
            raise ApplicationError(
                f"unknown or stale request {decision.request_id!r}", non_retryable=True
            )
        self._decisions[decision.request_id] = decision.approved
        return "recorded"

    @decide.validator
    def validate_decide(self, decision: Decision) -> None:
        if not decision.request_id:
            raise ValueError("request_id is required")

    @workflow.query
    def pending_requests(self) -> list[dict]:
        return self._pending

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(client=TemporalChatClient("gpt-4o"), tools=[delete_record])
        session = agent.create_session()
        response = await agent.run(prompt, session=session)
        while response.user_input_requests:
            requests = response.user_input_requests
            self._pending = [request.to_dict() for request in requests]
            await workflow.wait_condition(
                lambda: all(request.id in self._decisions for request in requests)
            )
            self._pending = []
            response = await agent.run(
                [
                    request.to_function_approval_response(self._decisions[request.id])
                    for request in requests
                ],
                session=session,
            )
        return response.text
```

The Update's validator checks only the argument's shape; the handler checks the id against the pending requests and raises `ApplicationError` for an unknown or stale one. Validators run before the handler, and under the time-skipping test server the workflow state a validator reads can lag behind, so a state check there fails runs that pass on a real server. Agent Framework keys approvals by the function call's id and binds the response to the request recorded in the session, so pass the same session to the second run. When a provider omits the id, the provider's client assigns one on the worker side before the response enters history, so the id in a pending request stays the same after a worker restart or a replay. A rejected approval sends the model a rejection result and the tool does not run. Agent Framework decides what happens to a request you do not answer: a response whose id matches no pending request is ignored with a log line and the request stays pending, and a run that brings a new message without answering drops the pending call from the transcript. The .NET plugin fails the run in both cases; here, check `response.user_input_requests` before the next run if a dropped request matters.

## Streaming

`agent.run(prompt, stream=True)` works as it does natively: the model Activity consumes the provider's stream, returns the updates, and the workflow iterates them and finalizes the response. To forward updates to a consumer outside the workflow while the call runs, pass `streaming_topic` and host a `WorkflowStream` on the workflow. With a topic set, every model call of that client runs as the `microsoft_agent_framework.invoke_model_streaming` Activity, with or without `stream=True`, and publishes each update in its `to_dict()` form to the topic; a batch goes out every `streaming_batch_interval` (default 100 milliseconds) or at 256 updates, whichever comes first. A run without `stream=True` gets its response assembled from the same updates:

```python
from agent_framework import ChatResponseUpdate
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient


@workflow.defn
class StreamingWorkflow:
    def __init__(self) -> None:
        self._stream = WorkflowStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(client=TemporalChatClient("gpt-4o", streaming_topic="events"))
        stream = agent.run(prompt, stream=True)
        async for update in stream:
            workflow.logger.debug("update %s", update.text)
        return (await stream.get_final_response()).text


async def follow(client: Client, workflow_id: str) -> None:
    async for item in WorkflowStreamClient.create(client, workflow_id).subscribe(["events"]):
        print(ChatResponseUpdate.from_dict(item.data).text, end="", flush=True)
```

One streamed call records, as its Activity result, one entry per run of text, one entry per streamed function call with its argument deltas merged (the provider's final full-arguments update stays its own entry), and one entry per usage or finish update, so a long answer or a large tool argument does not add one history entry per token; the topic still receives every update. The workflow iterator yields the recorded entries once the Activity completes, plus whatever the framework's own loop adds (releases from 1.25 add one empty update after a tool round).

If the updates cannot all be published, the Activity fails with an `ApplicationError` of type `MicrosoftAgentFrameworkStreamError`. It is non-retryable when a retry would fail the same way: a batch over the server's payload size limit (2 MiB per payload by default) or past the run's Signal limit (each batch is one Signal; a run accepts 10,000 by default). Other publish failures, such as a server outage, are retried, and the retry calls the model again. A retried streamed call publishes from its first update again, so a subscriber sees the earlier attempt's updates followed by the whole retry; a consumer that must not repeat output keys on `response_id` (a new provider response carries a new id) or on the call ids.

## Sessions and continue-as-new

By default Temporal owns the history: unless you set `store` yourself, `TemporalChatClient` sends `store=False` to the provider, so the provider keeps no conversation of its own (the OpenAI Responses API stores one and returns its id when `store` is left unset), and when you pass a session Agent Framework keeps the transcript in the session's in-memory history and resends it on every call. Event history is the durable record and replay rebuilds the session exactly.

A long chat grows the history. Check `workflow.info().is_continue_as_new_suggested()` after each turn and carry the session as `session.to_dict()`; the next run restores it with `AgentSession.from_dict`. The dict carries the transcript, pending approval state and the provider's conversation id. Live `Agent` objects, provider connections and spans do not carry over; build them again in the new run.

```python
from dataclasses import dataclass, field
from typing import Any

from agent_framework import Agent, AgentSession


@dataclass
class ChatInput:
    session: dict[str, Any] | None = None


@workflow.defn
class ChatWorkflow:
    def __init__(self) -> None:
        self._pending: list[str] = []
        self._done = False

    @workflow.signal
    def user_says(self, prompt: str) -> None:
        self._pending.append(prompt)

    @workflow.signal
    def end_chat(self) -> None:
        self._done = True

    @workflow.run
    async def run(self, input: ChatInput) -> None:
        agent = Agent(client=TemporalChatClient("gpt-4o"), instructions="Chat.")
        session = AgentSession.from_dict(input.session) if input.session else agent.create_session()
        while True:
            await workflow.wait_condition(lambda: bool(self._pending) or self._done)
            if not self._pending:
                return
            await agent.run(self._pending.pop(0), session=session)
            if workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(ChatInput(session=session.to_dict()))
```

To let the provider keep the conversation instead, pass `store=True` in the agent's `default_options` or a run's `options`. The returned conversation id lands in `session.service_session_id`, the next call carries it, and only the new turn crosses the boundary.

## Structured output

Pass a Pydantic class as `response_format`; its JSON schema crosses to the model Activity and the workflow parses the reply:

```python
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


@workflow.defn
class StructuredWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> Person:
        agent = Agent(client=TemporalChatClient("gpt-4o"))
        response = await agent.run(prompt, options={"response_format": Person})
        return response.value
```

The plugin sets [`pydantic_data_converter`](https://github.com/temporalio/sdk-python/blob/1.36.0/temporalio/contrib/pydantic.py) as the data converter unless your client already has a payload converter of its own, so Pydantic models cross the workflow boundary. `response.value` parses with the run's `response_format`, from an `Agent` run or from `client.get_response(...)` on the `TemporalChatClient` itself.

## Middleware and context providers

Agent, chat and function middleware run where they run natively: agent middleware in the `Agent`, chat and function middleware in `TemporalChatClient`'s layers, in the workflow. They must be deterministic, like any workflow code. Middleware that needs I/O calls `await workflow.execute_activity(...)`. Context providers run in the workflow too and may await Activities in `before_run` and `after_run`. `require_per_service_call_history_persistence` persists through the in-memory history provider inside the workflow. Compaction strategies run in the workflow; `SummarizationStrategy(client=TemporalChatClient(...))` makes its summary call an Activity.

Agent Framework's own exceptions raised in workflow code, such as `MiddlewareFailure`, fail the workflow rather than the workflow task: the plugin registers `AgentFrameworkException` and `MicrosoftAgentFrameworkWorkflowError` as workflow failure types, because retrying the task cannot change a deterministic failure.

## Limits

- `FileHistoryProvider` and `FileSessionStore` do not work in a workflow: they read and write files on a thread, which a workflow does not have, so the run fails with `MicrosoftAgentFrameworkWorkflowError` naming the call. Use the default in-memory history and carry `session.to_dict()` across continue-as-new.
- `MCPStdioTool`, `MCPStreamableHTTPTool` and `MCPWebsocketTool` cannot be passed in `tools=`: they hold a live connection. Use `TemporalMCPToolProvider`.
- `TemporalMCPToolProvider` exposes tools only: server prompts are not loaded, a result's `_meta` is not carried to the model, and there is no `tool_result_content` choice.
- Agent Framework creates agent spans and records tool metrics from workflow code through the global OpenTelemetry providers. With the OpenTelemetry SDK's providers set directly as the globals, every replay re-emits them; the plugin warns at worker and replayer start. Use `temporalio.contrib.opentelemetry.create_tracer_provider()` and `ReplaySafeMeterProvider`. The provider client's own chat spans run in the Activity with real timings.
- Synchronous tools run inline on the workflow task and block it while they run.
- The Workflows pillar (`WorkflowBuilder`, executors, checkpoints, `agent-framework-orchestrations`), the harness, skills, evaluations, DevUI, declarative agents, hosting, memory and vector providers and the experimental agent hooks are out of scope. Compose agents with Temporal Workflow code: signals, timers, child workflows.
- OpenAI background responses (`continuation_token`) and embedding clients are not routed through Activities.

## Composing with other plugins

Pass `MicrosoftAgentFrameworkPlugin` to the Client or to the Worker, not both: the worker inherits a client's plugins, and a second copy registers the Activities twice. `OpenTelemetryPlugin` composes on the client; set the tracer provider before connecting:

```python
import opentelemetry.trace
from temporalio.contrib.opentelemetry import OpenTelemetryPlugin, create_tracer_provider


async def start_worker() -> None:
    opentelemetry.trace.set_tracer_provider(create_tracer_provider())
    client = await Client.connect("localhost:7233", plugins=[OpenTelemetryPlugin()])
    worker = Worker(
        client,
        task_queue="microsoft-agent-framework",
        workflows=[HelloWorkflow],
        plugins=[MicrosoftAgentFrameworkPlugin(chat_clients=CHAT_CLIENTS)],
    )
    await worker.run()
```

The plugin passes `agent_framework`, `pydantic_core`, `annotated_types`, `msgspec`, `typing_inspection`, `opentelemetry`, `dotenv`, `mcp` and `mcp_types` through the workflow sandbox, and installs workflow-aware stand-ins for the clock, id and thread reads in Agent Framework's agent code (default agent and session ids, the loop's time budget, span durations, the thread hand-off of synchronous tools and of file-backed sessions) so they replay deterministically. Both apply to Workers and Replayers.

## Testing

`temporalio.microsoft_agent_framework.testing` ships a scripted client, so your workflow tests run without a provider:

```python
from temporalio.microsoft_agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)

plugin = MicrosoftAgentFrameworkPlugin(
    chat_clients={
        "gpt-4o": fake_chat_client(
            [
                function_call_response("fetch_user", {"user_id": "u1"}, call_id="call-1"),
                text_response("Done."),
            ]
        )
    }
)
```

Each model call takes the next entry; the last one repeats. `FakeChatClient` is a full Agent Framework client, so it returns the scripted calls the way a provider would, and it records every call in `calls`.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```

`tests/test_replay.py` replays the histories under `tests/histories`. After changing one of the recorded workflows, record them again with `UV_NO_EDITABLE=1 uv run --locked python -m tests.record_histories`.
