# Microsoft Agent Framework

> Release stage: [Pre-release](https://docs.temporal.io/develop/dotnet/integrations/agent-framework).

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) agents inside Temporal Workflows. You write the Workflow, build the agent in it with Agent Framework and Microsoft.Extensions.AI types, and call `RunAsync`. Every model call, every MCP call and every tool you opt in runs as a Temporal Activity, so a worker crash, a provider outage or a week-long approval wait does not lose the run. The provider client and its credentials live on the worker; Workflow code passes a model name.

The package targets `net10.0` and needs `Temporalio` 1.22.0 or later and `Microsoft.Agents.AI` 1.26.0 or later. .NET 8 and 9 left support on 2026-11-10, so there is no older target.

## Installation

```sh
dotnet add package Temporalio.AgentFramework --prerelease
dotnet add package Microsoft.Extensions.AI.OpenAI
```

The second package is the provider client for the quickstart; any `IChatClient` works.

## Quickstart

`Program.cs` holds the worker, the starter and the Workflow. Run `dotnet run worker` in one terminal and `dotnet run workflow` in another, with `OPENAI_API_KEY` set and a Temporal Service on `localhost:7233`.

```csharp
using System.ClientModel;
using System.ClientModel.Primitives;
using Microsoft.Extensions.AI;
using OpenAI;
using Temporalio.AgentFramework;
using Temporalio.Client;
using Temporalio.Worker;
using Temporalio.Workflows;

var plugin = new AgentFrameworkPlugin(new AgentFrameworkPluginOptions
{
    ChatClients = new Dictionary<string, Func<IChatClient>>
    {
        ["gpt-4o-mini"] = () =>
        {
            var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
                ?? throw new InvalidOperationException("OPENAI_API_KEY is not set");
            // Temporal retries the Activity; the provider client must not retry on its own.
            var options = new OpenAIClientOptions { RetryPolicy = new ClientRetryPolicy(maxRetries: 0) };
            return new OpenAIClient(new ApiKeyCredential(apiKey), options)
                .GetChatClient("gpt-4o-mini")
                .AsIChatClient();
        },
    },
});

var client = await TemporalClient.ConnectAsync(new("localhost:7233") { Plugins = [plugin] });

if (args.ElementAtOrDefault(0) == "worker")
{
    using var tokenSource = new CancellationTokenSource();
    Console.CancelKeyPress += (_, eventArgs) =>
    {
        tokenSource.Cancel();
        eventArgs.Cancel = true;
    };
    using var worker = new TemporalWorker(
        client,
        new TemporalWorkerOptions("agent-framework").AddWorkflow<HelloWorkflow>());
    await worker.ExecuteAsync(tokenSource.Token);
}
else
{
    var result = await client.ExecuteWorkflowAsync(
        (HelloWorkflow wf) => wf.RunAsync("What does Temporal add to an agent?"),
        new(id: $"hello-{Guid.NewGuid()}", taskQueue: "agent-framework"));
    Console.WriteLine(result);
}

[Workflow]
public class HelloWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions
        {
            Name = "assistant",
            ChatOptions = new ChatOptions { Instructions = "Answer in one sentence." },
        });
        var response = await agent.RunAsync(prompt);
        return response.Text;
    }
}
```

The plugin registers three Activities on the worker: `AgentFramework.InvokeModel`, `AgentFramework.McpListTools` and `AgentFramework.McpCallTool`. Pass the plugin to the client or to the worker options, not both; if it reaches the worker twice, the second registration is dropped.

## Chat clients and credentials

`AgentFrameworkPluginOptions.ChatClients` maps a model name to a factory. The factory runs once per worker, on the first model call that names it, and the client is reused for the worker's lifetime. The Workflow only ever sees the name: `new TemporalAgent("gpt-4o-mini")` or `new TemporalChatClient("gpt-4o-mini")`. A name without a factory fails the Activity with error type `AgentFrameworkConfigurationError` and no retry.

Register as many names as you need; two agents in one Workflow can use two models. Any `ChatClientBuilder` pipeline works in the factory (`UseOpenTelemetry`, `UseLogging`, caching) with one exception: a client that contains `FunctionInvokingChatClient` is rejected with `AgentFrameworkConfigurationError`, because it would run tools inside the model Activity instead of in the Workflow.

For Azure OpenAI, build the client the same way and set `AzureOpenAIClientOptions.RetryPolicy = new ClientRetryPolicy(maxRetries: 0)`.

## Retries and timeouts

Each model call is one `AgentFramework.InvokeModel` Activity with a 5-minute start-to-close timeout, a 10-second heartbeat timeout and the summary `agent-framework model call: {model}`. The Activity heartbeats at least every 3 seconds while the provider call runs and on every streamed update, so a dead worker is noticed within the heartbeat timeout. Change the options per agent or per client:

```csharp
using Temporalio.AgentFramework;
using Temporalio.Common;
using Temporalio.Workflows;

public static class ModelOptions
{
    public static TemporalAgent ReasoningAgent() => new("o3-mini", new TemporalAgentOptions
    {
        ModelActivityOptions = new ActivityOptions
        {
            StartToCloseTimeout = TimeSpan.FromMinutes(15),
            HeartbeatTimeout = TimeSpan.FromSeconds(30),
            RetryPolicy = new RetryPolicy { MaximumAttempts = 5, MaximumInterval = TimeSpan.FromSeconds(10) },
        },
    });
}
```

Temporal's default retry policy has no attempt limit, so a retryable provider error retries until the Workflow is closed. Set `MaximumAttempts` when you want the run to fail instead.

Provider errors are classified by HTTP status, never by message text. 400, 401, 403, 404 and 422 fail with error type `AgentFrameworkModelError` and `NonRetryable = true`; every other status, including 408, 429 and 5xx, fails with the same error type and is retried. When the OpenAI or Azure OpenAI exception carries a `Retry-After` header, that delay becomes the next retry interval. Exceptions without a status, such as a provider SDK's own exception types, propagate unchanged and retry. The status list and the two ways of reading it come from Cecil Phillip's draft at https://github.com/temporal-community/temporal-dotnet-agents.

## Tools

Tools go in `ChatOptions.Tools`, as in Agent Framework. Where a tool runs is your choice per tool:

- A plain `AIFunction` from `AIFunctionFactory.Create` runs inside the Workflow. Its code must be deterministic: no clock, no random numbers, no I/O, no `Task.Run`. Use it for string handling, routing and lookups in Workflow state.
- `ActivityAsTool.Create` wraps one of your Temporal Activities. Each invocation is one Activity with the options you pass; the parameter schema comes from the method signature and `[Description]` attributes. Register the Activity on the worker as usual.
- `TemporalMcpTool` runs an MCP tool as an Activity; see the next section.

```csharp
using System.ComponentModel;
using Microsoft.Extensions.AI;
using Temporalio.Activities;
using Temporalio.AgentFramework;
using Temporalio.Workflows;

public class WeatherActivities
{
    [Activity]
    [Description("Gets the current weather for a city")]
    public async Task<string> GetWeatherAsync([Description("City name")] string city)
    {
        using var http = new HttpClient();
        return await http.GetStringAsync(new Uri($"https://wttr.in/{city}?format=3"));
    }
}

[Workflow]
public class ToolsWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var toolOptions = new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(30) };
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions
        {
            AllowConcurrentInvocation = true,
            ChatOptions = new ChatOptions
            {
                Tools =
                [
                    ActivityAsTool.Create((WeatherActivities a) => a.GetWeatherAsync(default!), toolOptions),
                    AIFunctionFactory.Create((string text) => text.Length, "count_characters", "Counts the characters in a text"),
                ],
            },
        });
        return (await agent.RunAsync(prompt)).Text;
    }
}
```

`ActivityAsTool.Create` also takes a static call expression (`() => Activities.Fetch(default!)`), a delegate or a `MethodInfo`. The argument values in the expression are placeholders; the model supplies the real ones.

The loop follows `FunctionInvokingChatClient`. With `AllowConcurrentInvocation = true` the tool calls of one response run at the same time through `Workflow.WhenAllAsync`, with results in call order; the default runs them one after the other. A tool that throws gives the model the result `Error: Function failed.`; after 3 consecutive rounds of failures the run fails with `AgentFrameworkWorkflowError`. A call to a name that is not in `ChatOptions.Tools` gives the model `Error: Requested function "name" not found.` `MaximumIterationsPerRun` (default 40) caps the model calls of one run; at the cap the agent makes one more call without function tools and returns that answer.

Tool arguments and results travel inside the model Activity's input, so a tool result of several megabytes belongs in a blob store with a reference in the result.

## MCP

Register each server's transport on the worker and list its tools from the Workflow:

```csharp
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using Temporalio.AgentFramework;
using Temporalio.AgentFramework.Mcp;
using Temporalio.Workflows;

public static class McpSetup
{
    public static AgentFrameworkPlugin Plugin(Func<IChatClient> openAi) => new(new AgentFrameworkPluginOptions
    {
        ChatClients = new Dictionary<string, Func<IChatClient>> { ["gpt-4o-mini"] = openAi },
        McpServers = new Dictionary<string, Func<IClientTransport>>
        {
            ["docs"] = () => new StdioClientTransport(new StdioClientTransportOptions
            {
                Name = "docs",
                Command = "npx",
                Arguments = ["-y", "@modelcontextprotocol/server-filesystem", "/srv/docs"],
            }),
        },
        McpConnectionIdleTimeout = TimeSpan.FromMinutes(10),
    });
}

[Workflow]
public class McpWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var tools = await TemporalMcpServer.ListToolsAsync("docs", new McpToolOptions
        {
            AllowedTools = ["read_file", "search_files"],
            ToolNamePrefix = "docs_",
            CallToolActivityOptions = new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(30) },
        });
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions { Tools = [.. tools] },
        });
        return (await agent.RunAsync(prompt)).Text;
    }
}
```

`ListToolsAsync` is one `AgentFramework.McpListTools` Activity and returns a `TemporalMcpTool` per tool with the server's full input schema, so the model sees the parameters. Each call is one `AgentFramework.McpCallTool` Activity (60-second start-to-close timeout by default). The worker opens one `McpClient` per server on first use, shares it across Activities, closes it after `McpConnectionIdleTimeout` (default 5 minutes) without a call, and closes everything when the worker stops. `CacheTools = true` lets the worker answer the list Activity from the list it fetched earlier on the same connection; the default lists every time so a changed server is seen. Set `ToolNamePrefix` when two servers share a tool name; calls reach the server under the original name.

A result with `isError`, an unknown tool or bad arguments fails the Activity with error type `AgentFrameworkMcpError` and no retry; the model is told the tool failed. A transport that cannot connect retries under the Activity's retry policy.

The listed catalog says nothing about which tools have side effects. To pin a catalog instead of discovering it, build `TemporalMcpTool` yourself from a `ToolDeclaration` with the schema you expect, and wrap tools that need a human in `ApprovalRequiredAIFunction`. Sessions are stateless: one connection per worker per server, no session affinity.

## Approvals

Agent Framework's approval contract works unchanged. Wrap a tool in `ApprovalRequiredAIFunction` and the run returns a `ToolApprovalRequestContent` instead of calling it; the request id is `ficc_` plus the call id. Your Workflow decides how the answer arrives. The example uses a Query to show the pending request and an Update with a validator to deliver the decision:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.AgentFramework;
using Temporalio.Exceptions;
using Temporalio.Workflows;

[Workflow]
public class ApprovalWorkflow
{
    private readonly Dictionary<string, ToolApprovalResponseContent> decisions = new();
    private ToolApprovalRequestContent? pending;

    [WorkflowQuery]
    public ToolApprovalRequestContent? PendingApproval() => pending;

    [WorkflowUpdateValidator(nameof(DecideAsync))]
    public void ValidateDecision(ToolApprovalResponseContent decision)
    {
        if (pending is null || pending.RequestId != decision.RequestId)
        {
            throw new ApplicationFailureException($"No pending approval {decision.RequestId}");
        }
    }

    [WorkflowUpdate]
    public Task DecideAsync(ToolApprovalResponseContent decision)
    {
        decisions[decision.RequestId] = decision;
        return Task.CompletedTask;
    }

    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var sendEmail = AIFunctionFactory.Create((string to, string body) => $"sent to {to}", "send_email", "Sends an email");
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions { Tools = [new ApprovalRequiredAIFunction(sendEmail)] },
        });
        var session = await agent.CreateSessionAsync();
        var response = await agent.RunAsync(prompt, session);
        var request = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().FirstOrDefault();
        while (request is not null)
        {
            pending = request;
            var requestId = request.RequestId;
            await Workflow.WaitConditionAsync(() => decisions.ContainsKey(requestId));
            pending = null;
            response = await agent.RunAsync(new ChatMessage(ChatRole.User, [decisions[requestId]]), session);
            request = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().FirstOrDefault();
        }

        return response.Text;
    }
}
```

The client reads `PendingApproval`, shows the `FunctionCallContent` in `request.ToolCall`, and sends `request.CreateResponse(approved: true)` or `request.CreateResponse(false, "reason")` through the Update. A Signal works the same way when no validation or return value is needed. The next run on the same session executes approved calls, gives the model `Tool call invocation was rejected by the approver and must not be retried.` plus your reason for rejected ones, and never sends the approval contents to the provider. A run that starts with an unanswered request on the session fails with `AgentFrameworkWorkflowError`.

While the Workflow waits, no Activity is open: the wait costs nothing and survives worker restarts. When one response mixes approval-required calls with plain ones, the plain calls run immediately and the response carries the requests for the rest.

## Streaming

`RunStreamingAsync` works in a Workflow. The model Activity consumes the provider's stream, heartbeats on every update, and returns the whole response; the Workflow then yields that response as `AgentResponseUpdate`s, one per message, plus one per tool result. One model call is still one Activity.

To get the updates as the provider produces them, name a Workflow Stream topic. The Activity publishes each `ChatResponseUpdate` through `Temporalio.Extensions.WorkflowStreams`, batched every `StreamingBatchInterval` (default 100 milliseconds), and the Workflow hosts the stream:

```csharp
using Microsoft.Extensions.AI;
using Temporalio.AgentFramework;
using Temporalio.Client;
using Temporalio.Extensions.WorkflowStreams;
using Temporalio.Workflows;

public record StreamingInput(string Prompt, WorkflowStreamState? StreamState = null);

[Workflow]
public class StreamingWorkflow
{
    private readonly WorkflowStream stream;

    [WorkflowInit]
    public StreamingWorkflow(StreamingInput input) => stream = new(input.StreamState);

    [WorkflowRun]
    public async Task<string> RunAsync(StreamingInput input)
    {
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions { StreamingTopic = "deltas" });
        var response = await agent.RunAsync(input.Prompt);
        await Workflow.DelayAsync(TimeSpan.FromMilliseconds(500));
        return response.Text;
    }
}

public static class StreamReader
{
    public static async Task PrintAsync(ITemporalClient client, string workflowId)
    {
        await using var streams = new WorkflowStreamClient(client, workflowId);
        await foreach (var item in streams.SubscribeAsync<ChatResponseUpdate>(new WorkflowStreamSubscribeOptions { Topics = ["deltas"] }))
        {
            Console.Write(item.Value.Text);
        }
    }
}
```

The delay before returning gives subscribers time to poll the last batch; a closed Workflow cannot be polled. Workflow Streams is an experimental package and its wire format may change.

## Sessions and continue-as-new

`agent.CreateSessionAsync()` returns a `TemporalAgentSession`. The default history provider stores the transcript in the session's state bag, which lives in Workflow state and is rebuilt from Event History on replay; nothing is stored anywhere else. To carry a conversation across continue-as-new, serialize the session into the next run's input:

```csharp
using System.Text.Json;
using Temporalio.AgentFramework;
using Temporalio.Workflows;

public record ChatInput(IList<string> Prompts, int Index = 0, JsonElement? Session = null);

[Workflow]
public class ChatWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(ChatInput input)
    {
        var agent = new TemporalAgent("gpt-4o-mini");
        var session = input.Session is { } carried
            ? await agent.DeserializeSessionAsync(carried)
            : await agent.CreateSessionAsync();
        var response = await agent.RunAsync(input.Prompts[input.Index], session);
        if (input.Index + 1 < input.Prompts.Count && Workflow.ContinueAsNewSuggested)
        {
            var next = input with { Index = input.Index + 1, Session = await agent.SerializeSessionAsync(session) };
            throw Workflow.CreateContinueAsNewException((ChatWorkflow wf) => wf.RunAsync(next));
        }

        return response.Text;
    }
}
```

What carries over: the serialized session (transcript and any provider state in the bag) and whatever you put in the input. What does not: the `TemporalAgent` instance (build a new one), MCP connections (the worker reconnects), and an approval request you have not answered yet (answer it before continuing, or carry the pending request in your own input). The serialized form is `{"stateBag": ...}`, the shape `ChatClientAgent` writes, so a session saved by one can be read by the other.

## Structured output

`RunAsync<T>` sends the JSON schema of `T` as the response format and parses the answer:

```csharp
using Temporalio.AgentFramework;
using Temporalio.Workflows;

public record PersonInfo(string Name, int Age);

[Workflow]
public class ExtractWorkflow
{
    [WorkflowRun]
    public async Task<PersonInfo> RunAsync(string text)
    {
        var agent = new TemporalAgent("gpt-4o-mini");
        var response = await agent.RunAsync<PersonInfo>($"Extract the person from: {text}");
        return response.Result;
    }
}
```

The schema comes from `AgentAbstractionsJsonUtilities.DefaultOptions` (camelCase property names) unless you pass your own `JsonSerializerOptions`. A `T` whose schema is not an object, such as `string` or an array, is wrapped in `{"data": ...}` and unwrapped on the way back, as `AIAgent.RunAsync<T>` does.

## Context and history providers

`TemporalAgentOptions.ChatHistoryProvider` and `AIContextProviders` run inside the Workflow around each run, through the same `InvokingAsync` and `InvokedAsync` calls `ChatClientAgent` makes. They must complete synchronously and deterministically; a provider that returns an incomplete `ValueTask` fails the run with `AgentFrameworkWorkflowError` before it can leave the Workflow scheduler. Providers that read a database or a memory service belong in an Activity whose result you add to the prompt.

The default history provider is `InMemoryChatHistoryProvider`. Setting another one logs a warning and runs, because it persists the conversation next to Event History, which already holds it. Two providers that share a state key are rejected at construction.

## Why the plugin owns the loop

`AIAgent.RunAsync` and `ChatClientAgent.RunCoreAsync` await with `ConfigureAwait(false)`. Inside a Workflow that moves their continuation off the Workflow scheduler, the Workflow Task completes without a result, and the run stalls until it is terminated. `TemporalAgent` therefore runs the agent loop itself, over Agent Framework's own types and extension points, and hides the `AIAgent` run methods with overloads that stay on the scheduler. Cecil Phillip's draft at https://github.com/temporal-community/temporal-dotnet-agents named this hazard first.

Two things follow for you. Call `RunAsync`, `RunAsync<T>` and `RunStreamingAsync` on a `TemporalAgent`-typed reference; a call through an `AIAgent`-typed reference reaches the base method and fails the Workflow with `AgentFrameworkWorkflowError` and a message that says so. Agent middleware (`AIAgentBuilder`, `DelegatingAIAgent`, `AsAIFunction()`, `OpenTelemetryAgent`) wraps that base entry point and is not supported inside a Workflow; put telemetry and logging on the provider client in the worker factory instead. To use an agent as a tool, wrap `subAgent.RunAsync(...)` in an `AIFunction` of your own.

## Limits

- `ChatClientAgent`, `IChatClient.AsAIAgent()` and `ChatClientBuilder.BuildAIAgent()` stall inside a Workflow for the reason above. They keep working inside Activities and outside Temporal.
- Background responses are not supported: `AgentRunOptions.AllowBackgroundResponses`, `ContinuationToken` and `ChatClientAgentRunOptions.ChatClientFactory` fail the run with `AgentFrameworkWorkflowError`. `ChatOptions.RawRepresentationFactory` and `ContinuationToken` do not cross the Activity boundary and are dropped with a warning.
- `ChatOptions.Tools` may hold `AIFunction`s, `ActivityAsTool` and `TemporalMcpTool` wrappers, `ApprovalRequiredAIFunction` and the hosted tool types (`HostedWebSearchTool`, `HostedCodeInterpreterTool`, `HostedFileSearchTool`, `HostedMcpServerTool`). Hosted tools are passed through to the provider client by kind and options; any other `AITool` type fails the run.
- Service-managed conversations (`ConversationId`) are not used; the Workflow holds the history.
- `AIAgent` generates a random id on construction. `TemporalAgent` never reads it: `Id` is `TemporalAgentOptions.Id`, then `Name`, then `temporal-agent`, so `AgentResponse.AgentId` is the same on every replay.
- The plugin calls Agent Framework members marked experimental (`MAAI001`): the `ChatHistoryProvider.InvokingContext`, `ChatHistoryProvider.InvokedContext`, `AIContextProvider.InvokingContext` and `AIContextProvider.InvokedContext` constructors. The dependency floor is `Microsoft.Agents.AI` 1.26.0; a release that changes them shows up in this repository's nightly build.
- The whole conversation is the input of every model Activity. Long transcripts cost history size on every call; summarize or continue-as-new before Temporal's payload limits are near.

## Composing with other plugins

The plugin adds Activities, the Workflow failure type `AgentFrameworkWorkflowException`, and a shutdown hook that closes MCP connections. It installs no data converter, no interceptor and no header codec, so it composes with other plugins in any order. Pass it once, to `TemporalClientConnectOptions.Plugins` or to `TemporalWorkerOptions.Plugins`. Tracing from `Temporalio.Extensions.OpenTelemetry` works as with any Workflow: the model and MCP calls appear as Activity spans; provider spans come from `UseOpenTelemetry()` on the client you build in the factory.

## Develop

```sh
cd dotnet/agent-framework
make sync
make lint
make test
make build
```

`make test` builds in Release and runs the suite under Microsoft.Testing.Platform with a local Temporal dev server that `WorkflowEnvironment.StartLocalAsync` downloads on first run. No provider credentials are needed: the suite uses `Temporalio.AgentFramework.Testing.FakeChatClient`, `ScriptedResponse` and `InMemoryMcpServer`, which are shipped in the package for your own tests. The replay test reads the histories under `tests/Temporalio.AgentFramework.Tests/Histories`; set `AGENT_FRAMEWORK_RECORD_HISTORIES=1` while running the suite to write fresh copies under `TestResults/recorded-histories`.
