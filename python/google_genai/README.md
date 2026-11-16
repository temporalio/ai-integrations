# Google Gemini SDK Integration for Temporal

> Release stage: [Generally Available](https://docs.temporal.io/develop/python/integrations/google-genai).

## Overview

This plugin lets you use the [Google Gemini SDK](https://googleapis.github.io/python-genai/)
(`google-genai`) inside Temporal workflows with durable execution. Every Gemini
API call becomes a **Temporal activity**, so model calls, tool calls, file
operations, interactions, and managed agents are retried, recorded in history,
and survive worker restarts.

Key properties:

- **Credentials never enter the workflow.** The real `genai.Client` lives only
  on the worker, inside activities; no API keys or tokens appear in event
  history.
- **The SDK's automatic function calling (AFC) loop runs in the workflow**, so
  tool wrappers (`activity_as_tool`) work naturally — no manual agent loop.
- **Temporal owns retries.** Configure them via the activity `retry_policy`; the
  SDK's own retry loop is rejected to avoid double-retry (see
  [Retries & errors](#retries--errors)).

## Install

```bash
uv add temporalio-google-genai
```

## Hello World

```python
import os
from datetime import timedelta

from google import genai
from google.genai import types

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.google_genai import (
    GoogleGenAIPlugin,
    TemporalAsyncClient,
    activity_as_tool,
)
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig


# ---- a tool, as a normal Temporal activity (runs on the worker) ----
@activity.defn
async def get_weather(city: str) -> str:
    return f"It's sunny in {city}."


# ---- the workflow (runs in the Temporal sandbox) ----
@workflow.defn
class WeatherAgent:
    @workflow.run
    async def run(self, prompt: str) -> str:
        client = TemporalAsyncClient()
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[
                    activity_as_tool(
                        get_weather,
                        activity_config=ActivityConfig(
                            start_to_close_timeout=timedelta(seconds=30),
                        ),
                    ),
                ],
            ),
        )
        return response.text or ""


# ---- worker setup (outside the sandbox: real client + credentials) ----
async def main() -> None:
    gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    plugin = GoogleGenAIPlugin(gemini)

    client = await Client.connect("localhost:7233", plugins=[plugin])
    async with Worker(
        client,
        task_queue="gemini",
        workflows=[WeatherAgent],
        activities=[get_weather],
    ):
        result = await client.execute_workflow(
            WeatherAgent.run,
            "What's the weather in Tokyo?",
            id="weather-1",
            task_queue="gemini",
        )
        print(result)
```

Construct `TemporalAsyncClient` **inside** the workflow; construct the real
`genai.Client` and `GoogleGenAIPlugin` **on the worker**.

## What this plugin gives you

| Surface | Workflow API | Runs as |
| --- | --- | --- |
| Model calls | `client.models.generate_content` / `generate_content_stream` | activity (AFC loop in workflow) |
| Tools | `activity_as_tool(fn, ...)` | one activity per tool call |
| Files | `client.files.upload` / `download` | activity |
| File search | `client.file_search_stores.upload_to_file_search_store` | activity |
| Interactions | `client.interactions.create` / `get` / `cancel` / `delete` | whole-operation activity |
| Managed agents | `client.agents.create` / `get` / `list` / `delete` | whole-operation activity |
| MCP (client-side) | `TemporalMcpClientSession(name)` in `tools=[...]` | `list_tools` / `call_tool` activities |

Streamed responses are batched: the activity drains the stream and the workflow
iterates the collected chunks/events. `client.webhooks` is not supported in
workflows and raises.

## Tool calling

`activity_as_tool` wraps any `@activity.defn` function as a Gemini tool. When the
model calls it, the AFC loop (running in the workflow) dispatches it as a
durable activity:

```python
activity_as_tool(
    get_weather,
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=30)),
)
```

A timeout is required — `activity_config` must set `start_to_close_timeout` or
`schedule_to_close_timeout` (Temporal needs one; there is no default for tools).

## MCP support

Install the optional dependency for client-side MCP:

```bash
uv add "temporalio-google-genai[mcp]"
```

This adapter works with MCP Python SDK v1 or v2. The base package does not
require MCP. Server-side MCP on Vertex AI
and the Interactions API do not require this extra.

Workflow histories replay across MCP v1 and v2 workers in both directions, with
one exception: tool results that MCP v2 accepts but MCP v1 rejects. MCP v2 lets
a tool return `structuredContent` that is an array or a scalar, and resource
URIs that aren't full URLs, such as `notes.txt`. MCP v1 accepts only an object
and a full URL there. Once a history records such a result, only v2 workers can
replay it: a v1 worker fails the Workflow Task with "Failed decoding arguments".
If your MCP servers return such results, keep v1 off the Task Queue while you
upgrade, and don't roll back to v1 while those Workflows are running.

Client-side MCP (Gemini Developer API) is wired through the plugin: register the
server on the worker and reference it by name in the workflow.

```python
from contextlib import asynccontextmanager
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from temporalio.google_genai import TemporalMcpClientSession


# ---- worker: a factory yielding a connected, initialized session ----
@asynccontextmanager
async def weather_mcp():
    params = StdioServerParameters(command=sys.executable, args=["weather_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


plugin = GoogleGenAIPlugin(
    genai.Client(api_key=os.environ["GOOGLE_API_KEY"]),
    mcp_servers={"weather": weather_mcp},
    mcp_connection_idle_timeout=timedelta(minutes=5),
)


# ---- workflow: reference the server by name in the tools list ----
@workflow.defn
class McpAgent:
    @workflow.run
    async def run(self, prompt: str) -> str:
        client = TemporalAsyncClient()
        session = TemporalMcpClientSession(
            "weather",
            activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=30)),
        )
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(tools=[session]),
        )
        return response.text or ""
```

The MCP connection lives on the worker (pooled, idle-evicted); the workflow only
carries the server name. Tool discovery and calls run as `{name}-list-tools` /
`{name}-call-tool` activities, so the full tool parameter schema reaches the
model. Set `cache_tools=True` to list a server's tools once per workflow instead
of per turn.

A factory that doesn't return an async context manager now fails the Activity
without retries (`MCPServerFactoryTypeError`). Fix it before upgrading: a
corrected Worker won't resume an Activity that already failed.

A factory that isn't callable makes `GoogleGenAIPlugin(...)` raise `TypeError`
during construction, so the Worker won't start, even if no Workflow uses that
server. Fix these factories before upgrading too.

## Streaming

`generate_content_stream` works as usual — the workflow iterates chunks (batched
from the activity). To let an **external** consumer (a chat UI) observe chunks in
real time while the workflow runs durably, set `streaming_topic` on the client
and host a [`WorkflowStream`](https://github.com/temporalio/sdk-python/tree/1.36.0/temporalio/contrib/workflow_streams/) in the workflow. Each
streamed `GenerateContentResponse` is published to that topic as it arrives:

```python
from temporalio.contrib.workflow_streams import WorkflowStream


@workflow.defn
class StreamingAgent:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.stream = WorkflowStream()  # required when streaming_topic is set

    @workflow.run
    async def run(self, prompt: str) -> str:
        client = TemporalAsyncClient(streaming_topic="gemini")
        text = []
        async for chunk in await client.models.generate_content_stream(
            model="gemini-2.5-flash", contents=prompt,
        ):
            text.append(chunk.text or "")
        return "".join(text)
```

Consume the stream from outside the workflow:

```python
from temporalio.contrib.workflow_streams import WorkflowStreamClient


async def consume(client, workflow_id):
    stream = WorkflowStreamClient.create(client, workflow_id)
    async for item in stream.subscribe(
        ["gemini"], result_type=types.GenerateContentResponse,
    ):
        print(item.data.text, end="", flush=True)
```

The workflow's own iteration is unchanged (it still receives batched chunks for
the SDK to parse); the topic is purely for external real-time observation. If
`streaming_topic` is set but the workflow hosts no `WorkflowStream`, the call
raises `GoogleGenAIError`. Tune flush cadence with
`TemporalAsyncClient(streaming_topic=..., streaming_batch_interval=...)`
(default 100ms).

## Heartbeats

The Gemini activities, except `activity_as_tool` and MCP, heartbeat every half
`heartbeat_timeout`, which defaults to 10 seconds, even in an `activity_config`
that leaves it out. So a cancel or a lost worker is noticed within about 10
seconds. In Temporal Cloud each heartbeat that reaches the service is an Action:
with the default, at most one every 8 seconds, and none for calls under 5
seconds. Set `heartbeat_timeout` to `None` or `timedelta(0)` in
`activity_config` to turn heartbeats off.
Without heartbeats, a running call cannot receive cancellation. This includes
interaction streams, which earlier versions heartbeated once per event even
without a heartbeat timeout.

Heartbeats show that the worker is alive; they don't track stream progress.
A quiet interaction stream can stay alive until its HTTP timeout or
`start_to_close_timeout`, instead of failing when no event arrives within
`heartbeat_timeout`.

Temporal caps `heartbeat_timeout` at `start_to_close_timeout`, including one set
per model call with `http_options.timeout`, or with the Interactions and Agents
`timeout` argument, so a shorter one makes beats come sooner. `files.upload`,
`files.download`, `files.register_files`, and file search store uploads pass
`http_options.timeout` to the HTTP client instead.

The timer runs on the worker's event loop. Blocking the loop can cause heartbeat
timeouts in every activity sharing it. On Vertex AI, `interactions` and `agents`
calls refresh credentials on the loop: on the first call, then about once an
hour. A slow refresh can also fail other running Gemini calls. If that can be
slow, set a longer `heartbeat_timeout`, or `None`.

**Breaking change when you upgrade or roll back:**
`temporalio-google-genai` 1.0.0 adds timer heartbeats and the 10-second default.
A call scheduled with this default and run by a worker without timer
heartbeats fails after 10 seconds without a heartbeat, and so does each retry.

- To upgrade to 1.0.0, deploy it to the workers that run the Gemini activities
  first. If the same workers run workflows too, deploy it with
  `heartbeat_timeout=None` in `activity_config` and remove that in a later
  deploy.
- To roll back from 1.0.0, first deploy it with `heartbeat_timeout=None`, then
  wait for calls scheduled before that deploy to finish: they keep the
  10-second timeout.
- Runs pinned with Worker Versioning avoid both, when their Gemini activities
  use the workflow's task queue.

A config you pass replaces the defaults, so keep your activity timeouts when
turning heartbeats off. For example, the upgrade and rollback config above is:

```python
from datetime import timedelta

from temporalio.workflow import ActivityConfig

client = TemporalAsyncClient(
    activity_config=ActivityConfig(
        start_to_close_timeout=timedelta(seconds=60),
        heartbeat_timeout=None,
    ),
)
```

## Retries & errors

Temporal owns retries. Configure them with the activity `retry_policy` via
`activity_config`. The plugin **rejects** the SDK's own retry config so retries
don't compound:

- Constructing the plugin with a `genai.Client` that has
  `http_options.retry_options` raises `ValueError`.
- Setting `http_options.retry_options` on a per-request call raises
  `GoogleGenAIError`.

API-call activities classify failures: transient statuses (408, 429, 5xx) stay
retryable (the activity's `retry_policy` applies); other statuses (e.g. 4xx) are
non-retryable so the workflow fails fast.

## Vertex AI

Pass `vertexai=True` to both the worker-side `genai.Client` and the
workflow-side `TemporalAsyncClient`. On the workflow side you must also set
`project` and `location` **explicitly**:

```python
# worker
genai.Client(vertexai=True, project="my-project", location="us-central1")

# workflow
TemporalAsyncClient(vertexai=True, project="my-project", location="us-central1")
```

Normally the SDK auto-discovers `project`/`location` from the environment
(credentials, ADC, metadata server). That discovery
would be non-deterministic and break replay. Setting them by hand
keeps it deterministic.

## Composing with other plugins

`GoogleGenAIPlugin` is a `temporalio.plugin.SimplePlugin`; pass it in the
`plugins=[...]` list alongside others (e.g. OpenTelemetry). It contributes a
Pydantic data converter, the Gemini activities, a sandbox-passthrough config for
`google.genai` (and `mcp`), and registers `GoogleGenAIError` as a workflow
failure type. When composing data converters, construct the plugins so their
converters are compatible.
