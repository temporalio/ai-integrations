# Temporal MCP integration

> This package is experimental and may change in future versions.

`temporalio.mcp` lets native Temporal workflow code use MCP Python SDK
v2 clients. The workflow sees a durable proxy, while MCP transports, processes,
network connections, and credentials remain in worker-side Activities.

Version 0.2.0 changes both the import root and registered Activity-name prefix
from `temporalio.contrib.mcp` to `temporalio.mcp`. Workflows started with 0.1.x
must finish on 0.1.x workers before those workers are upgraded.

Install the integration with:

```bash
uv add temporalio-mcp
```

## Usage

Create the workflow-side client with a durable name and call the regular MCP
request/response operations:

```python
from mcp.types import TextContent
from temporalio import workflow
from temporalio.mcp import TemporalMCPClient


@workflow.defn
class WeatherWorkflow:
    @workflow.run
    async def run(self, city: str) -> str:
        mcp = TemporalMCPClient("weather")
        tools = await mcp.list_tools()
        assert any(tool.name == "get_weather" for tool in tools.tools)

        result = await mcp.call_tool("get_weather", {"city": city})
        content = result.content[0]
        assert isinstance(content, TextContent)
        return content.text
```

Register a worker-side factory under the same name. Streamable HTTP is the
simplest network transport:

```python
from mcp import Client
from temporalio.mcp import MCPPlugin
from temporalio.worker import Worker

plugin = MCPPlugin(
    {
        "weather": lambda: Client("https://example.com/mcp"),
    }
)

worker = Worker(
    temporal_client,
    task_queue="weather",
    workflows=[WeatherWorkflow],
    plugins=[plugin],
)
```

For a stdio server, return a fresh `Client` and transport from the factory:

```python
from mcp import Client, StdioServerParameters, stdio_client
from temporalio.mcp import MCPPlugin

parameters = StdioServerParameters(
    command="python",
    args=["weather_mcp_server.py"],
)
plugin = MCPPlugin(
    {
        "weather": lambda: Client(stdio_client(parameters)),
    }
)
```

`MCPPlugin` also accepts in-process MCP servers and custom MCP v2 transports
through the same `mcp.Client` API.

## Operations and durability

The proxy exposes these MCP operations, each backed by a named Activity:

| Workflow method | Activity suffix | Result |
| --- | --- | --- |
| `list_tools()` | `list-tools` | `ListToolsResult` |
| `call_tool()` | `call-tool` | `CallToolResult` |
| `list_prompts()` | `list-prompts` | `ListPromptsResult` |
| `get_prompt()` | `get-prompt` | `GetPromptResult` |
| `list_resources()` | `list-resources` | `ListResourcesResult` |
| `list_resource_templates()` | `list-resource-templates` | `ListResourceTemplatesResult` |
| `read_resource()` | `read-resource` | `ReadResourceResult` |

List operations follow every server pagination cursor within one Activity and
return a complete result with `next_cursor=None`. `list_tools()` is cached per
`TemporalMCPClient` instance by default, so repeated calls on the same object
schedule no further Activities (a new instance starts with an empty cache). Set
`cache_tools_list=False` to schedule an Activity for every call. Every Activity
asks the server directly; the MCP client's own response cache is bypassed
because workflow history is the durable record. For a multi-page result,
response metadata is merged in page order, the shortest `ttl_ms` is retained,
and `cache_scope` is `private` when any page is private.

All operations default to a one-minute start-to-close timeout per Activity
attempt. Override this with an `ActivityConfig`; the default is added only when
both `start_to_close_timeout` and `schedule_to_close_timeout` are omitted or
`None`:

```python
from datetime import timedelta
from temporalio.mcp import TemporalMCPClient

mcp = TemporalMCPClient(
    "weather",
    activity_config={"start_to_close_timeout": timedelta(seconds=20)},
)
```

Activities have at-least-once execution semantics. An MCP tool can therefore
run more than once when a worker loses its completion response. Tools with side
effects should be idempotent, usually by accepting a stable idempotency key.

MCP operations do not heartbeat. Do not set `heartbeat_timeout` in
`activity_config`: it does not make these Activities heartbeat or receive
cancellation. Temporal records an Activity timeout in the service, but that
timeout does not cancel an MCP request already running on a worker. Configure
the MCP `Client`'s `read_timeout_seconds` when the request itself must be
bounded.

## Errors and retries

A tool that fails returns a normal `CallToolResult` with `is_error=True`; the
workflow decides what to do with it. A JSON-RPC error response (an unknown
tool, prompt or resource, invalid arguments, a server-side failure) fails the
Activity with an `ApplicationError` of type `MCPProtocolError` whose
`details[0]` is the JSON-RPC error code. When the response includes JSON-RPC
error data, `details[1]` preserves it. Schema-invalid server responses use the
same error type, include Pydantic validation errors in `details[0]`, and are
non-retryable. Errors a retry cannot fix (parse and invalid-request errors,
unknown methods, invalid params, protocol-version, header and capability
mismatches, and URL elicitation) are also non-retryable.

Internal server errors, closed connections, request timeouts and transport
exceptions stay retryable. The default start-to-close timeout limits each
attempt, not the complete series of retries; with Temporal's default retry
policy and no schedule-to-close timeout, retryable failures can retry
indefinitely. Set `schedule_to_close_timeout` to bound the total time including
retries, and/or set a `retry_policy` to bound the number of attempts:

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from temporalio.mcp import TemporalMCPClient

mcp = TemporalMCPClient(
    "weather",
    activity_config={
        "start_to_close_timeout": timedelta(seconds=20),
        "schedule_to_close_timeout": timedelta(minutes=1),
        "retry_policy": RetryPolicy(maximum_attempts=3),
    },
)
```

Most JSON-RPC error responses leave the shared worker connection in place. A
closed connection, request timeout or unsupported negotiated protocol version
makes the worker reconnect before the next operation.

## Connections and configuration

Parameterless factories reuse modern, sessionless MCP connections until they
have been idle for five minutes. Set `connection_idle_timeout=None` to retain
them until plugin shutdown, or `timedelta(0)` to close them whenever they become
idle. Connections using a legacy MCP handshake are not shared between
Activities.

A factory may instead declare one positional parameter. The matching workflow
client passes `factory_argument` to it:

```python
plugin = MCPPlugin(
    {
        "weather": lambda tenant: Client(endpoint_for(tenant)),
    }
)

mcp = TemporalMCPClient("weather", factory_argument="acme")
```

A non-`None` argument creates a fresh client for every Activity. It is recorded
in workflow history, so use only a stable, non-secret identifier. Resolve URLs,
tokens, and other secrets inside the worker-side factory.

Connection reuse is an optimization, not durable session storage. A process
restart creates a new connection, while completed MCP results remain in
workflow history and replay without reconnecting.
