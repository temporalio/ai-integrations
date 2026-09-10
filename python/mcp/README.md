# Temporal MCP integration

> This package is experimental and may change in future versions.

`temporalio.contrib.mcp` lets native Temporal workflow code use MCP Python SDK
v2 clients. The workflow sees a durable proxy, while MCP transports, processes,
network connections, and credentials remain in worker-side Activities.

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
from temporalio.contrib.mcp import TemporalMCPClient


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
from temporalio.contrib.mcp import MCPPlugin
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
from temporalio.contrib.mcp import MCPPlugin

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
because workflow history is the durable record.

All operations default to a one-minute start-to-close timeout. Override this
with an `ActivityConfig`; the default is added only when the config sets
neither `start_to_close_timeout` nor `schedule_to_close_timeout`:

```python
from datetime import timedelta
from temporalio.contrib.mcp import TemporalMCPClient

mcp = TemporalMCPClient(
    "weather",
    activity_config={"start_to_close_timeout": timedelta(seconds=20)},
)
```

Activities have at-least-once execution semantics. An MCP tool can therefore
run more than once when a worker loses its completion response. Tools with side
effects should be idempotent, usually by accepting a stable idempotency key.

MCP operations do not heartbeat. Do not set `heartbeat_timeout` in
`activity_config`; cancellation cannot interrupt an in-flight MCP request and
takes effect when its start-to-close timeout expires.

## Errors and retries

A tool that fails returns a normal `CallToolResult` with `is_error=True`; the
workflow decides what to do with it. A JSON-RPC error response (an unknown
tool, prompt or resource, invalid arguments, a server-side failure) fails the
Activity with an `ApplicationError` of type `MCPProtocolError` whose
`details[0]` is the JSON-RPC error code. Errors a retry cannot fix (parse and
invalid-request errors, unknown methods, invalid params, protocol-version,
header and capability mismatches, and URL elicitation) are non-retryable.
Internal server errors, closed connections, request timeouts and transport
exceptions stay retryable, and the default `activity_config` has no
`retry_policy`, so they retry until the start-to-close timeout expires. Set a
`retry_policy` when that is not what you want:

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from temporalio.contrib.mcp import TemporalMCPClient

mcp = TemporalMCPClient(
    "weather",
    activity_config={
        "start_to_close_timeout": timedelta(seconds=20),
        "retry_policy": RetryPolicy(maximum_attempts=3),
    },
)
```

A JSON-RPC error response leaves the shared worker connection in place; only a
closed connection or a request timeout makes the worker reconnect.

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
