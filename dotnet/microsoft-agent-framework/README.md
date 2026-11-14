# Microsoft Agent Framework

> Release stage: [Pre-release](https://docs.temporal.io/develop/dotnet/integrations/microsoft-agent-framework).

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) agents inside Temporal Workflows. You write the Workflow, build the agent in it with Agent Framework and Microsoft.Extensions.AI types, and call `RunAsync`. Every model call, every MCP call and every tool you opt in runs as a Temporal Activity, so a worker crash, a provider outage or a week-long approval wait does not lose the run. The provider client and its credentials live on the worker; Workflow code passes a model name.

The package targets `net8.0`, so it runs on .NET 8, 9 and 10 (the test suite runs on .NET 8 and .NET 10), and needs `Temporalio` 1.22.0 or later and `Microsoft.Agents.AI` from 1.26.0 up to, and not including, 2.0.0. The plugin calls the provider context constructors that package marks experimental (`MAAI001`), so it checks for them when it is constructed and fails with `MicrosoftAgentFrameworkConfigurationError` naming the tested version, 1.26.0, if a release has changed them; this repository's nightly build runs the suite against the newest release. .NET 8 and 9 left support on 2026-11-10, so there is no older target.

## Installation

```sh
dotnet add package Temporalio.Extensions.MicrosoftAgentFramework --prerelease
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
using Temporalio.Extensions.MicrosoftAgentFramework;
using Temporalio.Client;
using Temporalio.Worker;
using Temporalio.Workflows;

var plugin = new MicrosoftAgentFrameworkPlugin(new MicrosoftAgentFrameworkPluginOptions
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
        new TemporalWorkerOptions("microsoft-agent-framework").AddWorkflow<HelloWorkflow>());
    await worker.ExecuteAsync(tokenSource.Token);
}
else
{
    var result = await client.ExecuteWorkflowAsync(
        (HelloWorkflow wf) => wf.RunAsync("What does Temporal add to an agent?"),
        new(id: $"hello-{Guid.NewGuid()}", taskQueue: "microsoft-agent-framework"));
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

The plugin registers three Activities on the worker: `MicrosoftAgentFramework.InvokeModel`, `MicrosoftAgentFramework.McpListTools` and `MicrosoftAgentFramework.McpCallTool`. Pass the plugin to the client or to the worker options, not both; if it reaches the worker twice, the second registration is dropped.

## Chat clients and credentials

`MicrosoftAgentFrameworkPluginOptions.ChatClients` maps a model name to a factory. The factory runs once per worker, on the first model call that names it, and the client is reused for the worker's lifetime. The Workflow only ever sees the name: `new TemporalAgent("gpt-4o-mini")` or `new TemporalChatClient("gpt-4o-mini")`. A name without a factory, or a factory that returns null, fails the Activity with error type `MicrosoftAgentFrameworkConfigurationError` and no retry. A factory that throws (for example when `OPENAI_API_KEY` is not set) fails the attempt with the same error type, logs the exception on the worker, and is retried under the Activity's retry policy, so a worker fixed in place picks the run up; set `MaximumAttempts` or a schedule-to-close timeout when you want the run to fail instead.

Register as many names as you need; two agents in one Workflow can use two models. Any `ChatClientBuilder` pipeline works in the factory (`UseOpenTelemetry`, `UseLogging`, caching) with one exception: a client that contains `FunctionInvokingChatClient` is rejected with `MicrosoftAgentFrameworkConfigurationError`, because it would run tools inside the model Activity instead of in the Workflow.

For Azure OpenAI, build the client the same way and set `AzureOpenAIClientOptions.RetryPolicy = new ClientRetryPolicy(maxRetries: 0)`.

Credentials stay in the factory. `ChatOptions.AdditionalProperties` and `AgentRunOptions.AdditionalProperties` are recorded in Event History with the model Activity's input, so an entry named `authorization`, `x-api-key` or `api-key`, at the top level or inside a nested headers dictionary, fails the run with `MicrosoftAgentFrameworkWorkflowError` before any Activity is scheduled.

The quickstart's Chat Completions client stores nothing on OpenAI's side. The Responses client (`GetResponsesClient().AsIChatClient(model)`) stores every response there unless told otherwise, and the plugin never reads those copies: it sends the transcript on every call and never sets `previous_response_id` or a conversation id, because the Workflow holds the history. Microsoft.Extensions.AI exposes that switch only through `ChatOptions.RawRepresentationFactory` and the OpenAI SDK's own types, and the plugin references no provider SDK, so turn storage off in the factory with a wrapper:

```csharp
using System.ClientModel;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;

#pragma warning disable OPENAI001 // The Responses client is marked experimental in OpenAI 2.13 and Microsoft.Extensions.AI.OpenAI 10.10.

public static class ResponsesClients
{
    // Temporal holds the transcript, so the Responses API does not need to keep a copy of each response.
    public static IChatClient WithoutStoredResponses(string apiKey, string model) =>
        new OpenAIClient(new ApiKeyCredential(apiKey)).GetResponsesClient().AsIChatClient(model)
            .AsBuilder()
            .Use((messages, options, next, cancellationToken) =>
            {
                options ??= new ChatOptions();
                options.RawRepresentationFactory = _ => new CreateResponseOptions { StoredOutputEnabled = false };
                return next(messages, options, cancellationToken);
            })
            .Build();
}
```

A reasoning model needs `IncludedProperties` with the encrypted reasoning content on the same options once storage is off, or its reasoning does not carry across the tool rounds of a run; that choice belongs to the factory, which is why the plugin does not set the switch for you. A `RawRepresentationFactory` set in Workflow code is dropped with a warning, because it cannot cross the Activity boundary.

The headers of a hosted MCP server tool follow the same rule: set them on the worker, in `MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaders`, never on the `HostedMcpServerTool` in the Workflow. The Hosted tools section below has the details.

To keep an MCP connection open for the worker's lifetime, set `McpConnectionIdleTimeout` to a value longer than the worker runs; zero and negative values are rejected.

## Retries and timeouts

Each model call is one `MicrosoftAgentFramework.InvokeModel` Activity with a 5-minute start-to-close timeout, a 10-second heartbeat timeout and the model name as its summary. The Activity heartbeats at least every 3 seconds while the provider call runs and on every streamed update, so a dead worker is noticed within the heartbeat timeout. Change the options per agent or per client:

```csharp
using Temporalio.Extensions.MicrosoftAgentFramework;
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

Provider errors are classified by HTTP status, never by message text, with the rule the Python plugin uses: 408, 409, 429 and 5xx fail with error type `MicrosoftAgentFrameworkModelError` and are retried; every other 4xx fails with the same error type and `NonRetryable = true`. A provider `x-should-retry` header of `true` or `false` overrides the status rule. When the OpenAI or Azure OpenAI exception carries `retry-after-ms`, or else `Retry-After` as seconds, with a fraction, or as an HTTP date, that delay becomes the next retry interval, capped at 60 seconds as the OpenAI SDK caps it; a value the plugin cannot read is skipped. The status is read from `HttpRequestException` or from an `int` `Status` or `StatusCode` property of the provider's exception type (`ClientResultException`, `Azure.RequestFailedException`), without a reference to the provider package; an exception without one propagates unchanged and retries. The two ways of reading the status come from Cecil Phillip's draft at https://github.com/temporal-community/temporal-dotnet-agents.

## Tools

Tools go in `ChatOptions.Tools`, as in Agent Framework. Where a tool runs is your choice per tool:

- A plain `AIFunction` from `AIFunctionFactory.Create` runs inside the Workflow. Its code must be deterministic: no clock, no random numbers, no I/O, no `Task.Run`. It runs again on every replay, so keep its effects in Workflow state. Use it for string handling, routing and lookups in Workflow state.
- `ActivityAsTool.Create` wraps one of your Temporal Activities. Each invocation is one Activity with the options you pass; the parameter schema comes from the method signature and `[Description]` attributes. Register the Activity on the worker as usual.
- `TemporalMcpTool` runs an MCP tool as an Activity; see the next section.

```csharp
using System.ComponentModel;
using Microsoft.Extensions.AI;
using Temporalio.Activities;
using Temporalio.Extensions.MicrosoftAgentFramework;
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

`ActivityAsTool.Create` also takes a static call expression (`() => Activities.Fetch(default!)`), a delegate or a `MethodInfo`. The argument values in the expression are placeholders; the model supplies the real ones. A trailing `CancellationToken` parameter with a default value stays out of the schema and of the call, and the Activity reads cancellation from `ActivityExecutionContext.Current.CancellationToken`; a token parameter anywhere else is rejected, because a token cannot cross the Activity boundary.

The loop follows `FunctionInvokingChatClient`. With `AllowConcurrentInvocation = true` the tool calls of one response run at the same time through `Workflow.WhenAllAsync`, with results in call order; the default runs them one after the other. A tool that throws gives the model the result `Error: Function failed.`; three consecutive failing rounds are tolerated and the fourth fails the run with `MicrosoftAgentFrameworkWorkflowError`. A call to a name that is not in `ChatOptions.Tools` gives the model `Error: Requested function "name" not found.` `MaximumIterationsPerRun` (default 40) caps the model calls of one run; at the cap the agent makes one more call without function tools and returns that answer.

A call that leaves out a required argument of an `ActivityAsTool` fails before any Activity is scheduled, with the message `AIFunctionFactory` gives, so the model reads `Error: Function failed.` and can call again. Tool results cross the Activity boundary as the JSON the provider client would have sent: a string as is, an `AIContent` with its type, any other object as camel-cased JSON, and `FunctionResultContent.Result` in the response messages holds that JSON.

Tool arguments and results travel inside the model Activity's input, so a tool result of several megabytes belongs in a blob store with a reference in the result.

## MCP

Register each server's transport on the worker and list its tools from the Workflow:

```csharp
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using Temporalio.Extensions.MicrosoftAgentFramework;
using Temporalio.Extensions.MicrosoftAgentFramework.Mcp;
using Temporalio.Workflows;

public static class McpSetup
{
    public static MicrosoftAgentFrameworkPlugin Plugin(Func<IChatClient> openAi) => new(new MicrosoftAgentFrameworkPluginOptions
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

`ListToolsAsync` is one `MicrosoftAgentFramework.McpListTools` Activity and returns a `TemporalMcpTool` per tool with the server's full input schema, so the model sees the parameters. Each call is one `MicrosoftAgentFramework.McpCallTool` Activity (60-second start-to-close timeout by default). The plugin opens one `McpClient` per server on first use, shares it across Activities and across every worker running with the same plugin instance, closes it after `McpConnectionIdleTimeout` (default 5 minutes) without a call, and closes everything when the last of those workers (or a replayer) stops; stopping one worker leaves the others' calls running on the connection. `CacheTools = true` lets the worker answer the list Activity from the list it fetched earlier on the same connection; the default lists every time so a changed server is seen. Set `ToolNamePrefix` when two servers share a tool name; calls reach the server under the original name. The names the model sees follow the rule of the Python Agent Framework's `MCPTool`, so one rule serves both plugins (`Microsoft.Agents.AI` has no prefix rule and the MCP C# SDK's `McpClientTool` keeps the server's name): characters outside `A-Za-z0-9_.-` become `-`, and the prefix is joined with one `_`, so `docs` and `docs_` both give `docs_read_file`. Names in `ChatOptions.Tools` must be unique; when two servers or a server and a local tool share a name, the first match runs.

Results reach the model the way the MCP C# SDK's `McpClientTool` delivers them: a plain text result becomes a `TextContent`, several content blocks become a list, and a result with `isError`, structured content or application metadata goes back whole, so the model reads the server's error text and can correct the call; no Activity fails and the consecutive-error count does not move. A protocol error, for an unknown tool or bad arguments, fails the Activity with error type `MicrosoftAgentFrameworkMcpError` and no retry, as `McpClientTool` throws, and the model is told the tool failed. A transport that cannot connect retries under the Activity's retry policy, and a call that fails on the transport drops the cached connection so the next attempt reconnects. A transport factory that throws is logged and retried under the Activity's retry policy, like a chat client factory that throws; one that returns null fails with `MicrosoftAgentFrameworkConfigurationError` and no retry.

The listed catalog says nothing about which tools have side effects. To pin a catalog instead of discovering it, build `TemporalMcpTool` yourself from a `ToolDeclaration` with the schema you expect, and wrap tools that need a human in `ApprovalRequiredAIFunction`. Sessions are stateless: one connection per plugin instance per server, no session affinity.

## Hosted tools

`HostedWebSearchTool`, `HostedCodeInterpreterTool`, `HostedFileSearchTool` and `HostedMcpServerTool` run on the provider's side, inside the model call. Put them in `ChatOptions.Tools` as usual; the plugin carries each one to the model Activity by kind and options and rebuilds it there, so the provider client receives the same tool.

The exception is `HostedMcpServerTool.Headers`. The model Activity's input is recorded in Event History, so a header set on the tool in the Workflow, an `Authorization` or API key header in practice, would be recorded with it. The plugin sends no hosted MCP server headers from the Workflow: a tool whose `Headers` are set fails the run with `MicrosoftAgentFrameworkWorkflowError` before any Activity is scheduled. Register the headers on the worker by server name instead, and the Activity adds them when it rebuilds the tool:

```csharp
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework;
using Temporalio.Workflows;

public static class HostedMcpSetup
{
    public static MicrosoftAgentFrameworkPlugin Plugin(Func<IChatClient> openAi, string docsToken) => new(new MicrosoftAgentFrameworkPluginOptions
    {
        ChatClients = new Dictionary<string, Func<IChatClient>> { ["gpt-4o-mini"] = openAi },
        HostedMcpServerHeaders = new Dictionary<string, IReadOnlyDictionary<string, string>>
        {
            ["docs"] = new Dictionary<string, string> { ["Authorization"] = $"Bearer {docsToken}" },
        },
    });
}

[Workflow]
public class HostedMcpWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions
            {
                Tools = [new HostedMcpServerTool("docs", "https://example.com/mcp") { AllowedTools = ["search"] }],
            },
        });
        return (await agent.RunAsync(prompt)).Text;
    }
}
```

The key is the tool's `ServerName`; a server without an entry gets no headers. For a token that expires, register a provider in `HostedMcpServerHeaderProviders` instead: the worker calls it before every model call that carries the tool, and a server name goes in one of the two dictionaries, not both. Nothing else on a hosted tool is treated as a secret: `AdditionalProperties`, inputs and allowed tool names travel in the Activity input like the rest of the request.

## Approvals

Agent Framework's approval contract applies, with the two differences described below. Wrap a tool in `ApprovalRequiredAIFunction` and the run returns a `ToolApprovalRequestContent` instead of calling it; the request id is `ficc_` plus the call id. Your Workflow decides how the answer arrives. The example uses a Query to show the pending requests and an Update to deliver each decision:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework;
using Temporalio.Exceptions;
using Temporalio.Workflows;

public record ApprovalDecision(string RequestId, bool Approved, string? Reason = null);

[Workflow]
public class ApprovalWorkflow
{
    private readonly Dictionary<string, ToolApprovalRequestContent> pending = new();
    private readonly Dictionary<string, ApprovalDecision> decisions = new();
    private bool awaitingReply;

    [WorkflowQuery]
    public IReadOnlyList<ToolApprovalRequestContent> PendingApprovals() => pending.Values.ToList();

    [WorkflowUpdateValidator(nameof(DecideAsync))]
    public void ValidateDecision(ApprovalDecision decision)
    {
        // Validators run before the Workflow sees the model's reply, so this one checks the input
        // and not the state. Bad input never enters history.
        if (string.IsNullOrEmpty(decision.RequestId))
        {
            throw new ApplicationFailureException("RequestId is required");
        }
    }

    [WorkflowUpdate]
    public async Task<string> DecideAsync(ApprovalDecision decision)
    {
        // A decision can arrive in the same Workflow Task as the model's reply, before RunAsync has
        // read the requests in that reply. Judge the id once the reply has been read.
        await Workflow.WaitConditionAsync(() => !awaitingReply);
        if (decisions.ContainsKey(decision.RequestId))
        {
            throw new ApplicationFailureException($"Approval request '{decision.RequestId}' is already decided");
        }

        if (!pending.ContainsKey(decision.RequestId))
        {
            throw new ApplicationFailureException($"No pending approval request '{decision.RequestId}'");
        }

        decisions[decision.RequestId] = decision;
        return decision.Approved ? "approved" : "rejected";
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
        var response = await ReplyAsync(agent, new ChatMessage(ChatRole.User, prompt), session);
        var requests = ApprovalRequests(response);
        while (requests.Count > 0)
        {
            foreach (var request in requests)
            {
                pending[request.RequestId] = request;
            }

            // No Activity is open while the Workflow waits: the wait costs nothing and survives a worker restart.
            await Workflow.WaitConditionAsync(() => pending.Keys.All(decisions.ContainsKey));
            var replies = requests
                .Select(request => (AIContent)request.CreateResponse(decisions[request.RequestId].Approved, decisions[request.RequestId].Reason))
                .ToList();
            pending.Clear();
            response = await ReplyAsync(agent, new ChatMessage(ChatRole.User, replies), session);
            requests = ApprovalRequests(response);
        }

        await Workflow.WaitConditionAsync(() => Workflow.AllHandlersFinished);
        return response.Text;
    }

    private static List<ToolApprovalRequestContent> ApprovalRequests(AgentResponse response) =>
        response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();

    private async Task<AgentResponse> ReplyAsync(TemporalAgent agent, ChatMessage message, AgentSession session)
    {
        awaitingReply = true;
        try
        {
            return await agent.RunAsync(message, session);
        }
        finally
        {
            awaitingReply = false;
        }
    }
}
```

The client reads `PendingApprovals`, shows the `FunctionCallContent` in each `request.ToolCall`, and sends an `ApprovalDecision` through the Update; the Workflow turns it into `request.CreateResponse(approved, reason)` and runs the agent again. The validator checks only that an id is present, because validators run before the Workflow has read the model's reply; the handler waits for that, then fails the Update, not the Workflow, for an id that is unknown or already decided. A Signal works the same way when no return value is needed. The next run on the same session executes approved calls, gives the model `Tool call invocation rejected.` plus your reason for rejected ones, the text `FunctionInvokingChatClient` uses, and never sends the approval contents to the provider.

The plugin binds each response to the request it surfaced, as Agent Framework's default middleware does: the call that runs is the one the session recorded under that request id, with the name and arguments the human saw, so a response that carries a changed `FunctionCallContent` cannot redirect the call. Your Workflow only has to check the request id, as the Update handler above does. A request that appears only in a run's input cannot authorize anything.

Two things differ from `ChatClientAgent`, which rejects an unanswered request on the caller's behalf and ignores a response it did not record. Here a request that reaches the next run without a response, and a response whose request id the session never surfaced, both fail the run with `MicrosoftAgentFrameworkWorkflowError`: a durable Workflow should not drop a pending decision silently. A run that fails after the decision, for example because the final model call failed for good, stores nothing in the session, as under `ChatClientAgent`; retry the same run with the same decision, and expect the approved call to run again.

While the Workflow waits, no Activity is open: the wait costs nothing and survives worker restarts. When one response mixes approval-required calls with plain ones, nothing runs until the human answers: the response carries the requests that need a decision, the plain calls are held in the session, and the next run executes both groups in one tool round, which is what Agent Framework's default middleware does.

## Streaming

`RunStreamingAsync` works in a Workflow. The model Activity consumes the provider's stream, heartbeats on every update, and returns the whole response; the Workflow then yields that response as `AgentResponseUpdate`s: for each model call, one per response message and one carrying its usage, plus one per tool result message. One model call is still one Activity.

To get the updates as the provider produces them, name a Workflow Stream topic; a topic on an agent whose Workflow hosts no `WorkflowStream` fails the run with `MicrosoftAgentFrameworkWorkflowError` before the first model call. The Activity publishes each `ChatResponseUpdate` through `Temporalio.Extensions.WorkflowStreams`, batched every `StreamingBatchInterval` (default 100 milliseconds) or every 256 updates, whichever comes first, and the Workflow hosts the stream:

```csharp
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework;
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

The delay before returning gives subscribers time to poll the last batch; a closed Workflow cannot be polled. Each batch is one Signal and one Workflow Task in Event History, so a long stream at the default interval adds several hundred events. A model Activity that retries starts a new publisher, and subscribers see the first attempt's updates again; key your display on `ResponseId` or clear it when a new attempt begins. Workflow Streams is an experimental package and its wire format may change.

## Sessions and continue-as-new

`agent.CreateSessionAsync()` returns a `TemporalAgentSession`. The default history provider stores the transcript in the session's state bag, which lives in Workflow state and is rebuilt from Event History on replay; nothing is stored anywhere else. To carry a conversation across continue-as-new, serialize the session into the next run's input:

```csharp
using System.Text.Json;
using Temporalio.Extensions.MicrosoftAgentFramework;
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

What carries over: the serialized session and whatever you put in the input. The session holds the transcript, any provider state in the bag, and the calls held back behind an approval. What does not: the `TemporalAgent` instance (build a new one), MCP connections (the worker reconnects), and your Workflow's own record of a pending approval. The request itself rides in the transcript, so answer it on the first run of the new execution or that run fails with `MicrosoftAgentFrameworkWorkflowError`; carry the request in your input so the new run can show it and wait for the decision. The serialized form is `{"stateBag": ...}`, the shape `ChatClientAgent` writes, so a transcript saved by one can be read by the other. Approvals in flight do not carry across to `ChatClientAgent`: the two keep pending and held requests under different state keys, so finish the approval round before you hand a session to the other side.

## Structured output

`RunAsync<T>` sends the JSON schema of `T` as the response format and parses the answer:

```csharp
using Temporalio.Extensions.MicrosoftAgentFramework;
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

`TemporalAgentOptions.ChatHistoryProvider` and `AIContextProviders` run inside the Workflow around each run, through the same `InvokingAsync` and `InvokedAsync` calls `ChatClientAgent` makes, with the same request list: history, your input and the messages the context providers added, so their messages are stored the way `ChatClientAgent` stores them. Providers must complete synchronously and deterministically, and they run under the SDK's scheduler checks like any Workflow code. A provider that awaits an Activity, `Workflow.DelayAsync` or another task that is not complete when it returns fails the run with `MicrosoftAgentFrameworkWorkflowError` naming the provider. A provider that leaves the Workflow scheduler, with `Task.Run`, `Task.Delay` or a blocking `.Result` on thread-pool work, fails the Workflow Task with the SDK's own error, as the same line would anywhere in a Workflow, and Temporal retries the task until the code is fixed or the run times out; a provider that throws does the same. Throw `ApplicationFailureException` from a provider to fail the run instead. Providers that read a database or a memory service belong in an Activity your Workflow runs before the agent, with the result added to the prompt or the provider's state.

The default history provider is `InMemoryChatHistoryProvider`. Setting another one logs a warning and runs, because it persists the conversation next to Event History, which already holds it. Two providers that share a state key are rejected at construction.

## Why the plugin owns the loop

`AIAgent.RunAsync` and `ChatClientAgent.RunCoreAsync` await with `ConfigureAwait(false)`. Inside a Workflow that moves their continuation off the Workflow scheduler, the Workflow Task completes without a result, and the run stalls until it is terminated. `TemporalAgent` therefore runs the agent loop itself, over Agent Framework's own types and extension points, and hides the `AIAgent` run methods with overloads that stay on the scheduler. Cecil Phillip's draft at https://github.com/temporal-community/temporal-dotnet-agents named this hazard first.

Two things follow for you. Call `RunAsync`, `RunAsync<T>` and `RunStreamingAsync` on a `TemporalAgent`-typed reference; a call through an `AIAgent`-typed reference reaches the base method and fails the Workflow with `MicrosoftAgentFrameworkWorkflowError` and a message that says so. Agent middleware (`AIAgentBuilder`, `DelegatingAIAgent`, `AsAIFunction()`, `OpenTelemetryAgent`) wraps that base entry point and is not supported inside a Workflow; put telemetry and logging on the provider client in the worker factory instead. To use an agent as a tool, wrap `subAgent.RunAsync(...)` in an `AIFunction` of your own.

## Limits

- `ChatClientAgent`, `IChatClient.AsAIAgent()` and `ChatClientBuilder.BuildAIAgent()` stall inside a Workflow for the reason above. They keep working inside Activities and outside Temporal.
- Background responses are not supported: `AllowBackgroundResponses` and `ContinuationToken`, on `AgentRunOptions` or on `ChatOptions`, and `ChatClientAgentRunOptions.ChatClientFactory` fail the run with `MicrosoftAgentFrameworkWorkflowError`. `ChatOptions.RawRepresentationFactory` does not cross the Activity boundary and is dropped with a warning.
- `ChatOptions.Tools` may hold `AIFunction`s, `ActivityAsTool` and `TemporalMcpTool` wrappers, `ApprovalRequiredAIFunction` and the four hosted tool types, with the header rule from the Hosted tools section; any other `AITool` type fails the run.
- Service-managed conversations are not used: the Workflow holds the history and sends it on every call, so a `ConversationId` on `ChatOptions` fails the run with `MicrosoftAgentFrameworkWorkflowError` instead of making the model see every turn twice.
- `AIAgent` generates a random id on construction. `TemporalAgent` never reads it: `Id` is `TemporalAgentOptions.Id`, then `Name`, then `temporal-agent`, so `AgentResponse.AgentId` is the same on every replay.
- The plugin calls Agent Framework members marked experimental (`MAAI001`): the `ChatHistoryProvider.InvokingContext`, `ChatHistoryProvider.InvokedContext`, `AIContextProvider.InvokingContext` and `AIContextProvider.InvokedContext` constructors. The package reference accepts `Microsoft.Agents.AI` 1.26.0 up to, and not including, 2.0.0; constructing the plugin checks that those constructors are present and fails with `MicrosoftAgentFrameworkConfigurationError` naming the tested version if a release changed them, and this repository's nightly build runs the suite against the newest release.
- The whole conversation is the input of every model Activity, so a long transcript grows both the per-call payload and the Event History. Continue-as-new bounds the History; only a shorter transcript (a trimmed session, or a summary your code produces) bounds the payload, which Temporal caps at 2 MiB.
- Values in `AdditionalProperties` cross the Activity boundary as JSON and are recorded in Event History; an entry named like a credential header fails the run, see Chat clients and credentials. The plugin turns JSON primitives in `ChatOptions.AdditionalProperties` back into `bool`, `long`, `double` and `string` before the provider client sees them; nested objects there, and the `AdditionalProperties` of messages and responses, arrive as `JsonElement`.
- Only the `AIContent` types Microsoft.Extensions.AI declares serialize. A custom `AIContent` subclass in a message or a provider response fails serialization at the Activity boundary.

## Composing with other plugins

The plugin adds Activities and run hooks that count the workers using the instance and close MCP connections when the last one stops; `MicrosoftAgentFrameworkWorkflowException` is an `ApplicationFailureException`, so it fails a Workflow with no registration. It installs no data converter, no interceptor and no header codec, so it composes with other plugins in any order. Pass it once, to `TemporalClientConnectOptions.Plugins` or to `TemporalWorkerOptions.Plugins`. The same instance on both registers its Activities once. Two instances on one worker, or two of your own Activities with one name, fail the worker with the SDK's `Duplicate activity named ...` error: the plugin removes only repeated copies of its own definitions. Tracing from `Temporalio.Extensions.OpenTelemetry` works as with any Workflow: the model and MCP calls appear as Activity spans; provider spans come from `UseOpenTelemetry()` on the client you build in the factory.

## Testing

`Temporalio.Extensions.MicrosoftAgentFramework.Testing` ships `FakeChatClient`, `ScriptedResponse` and `InMemoryMcpServer`, so your Workflow tests run without a provider or a network:

```csharp
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework;
using Temporalio.Extensions.MicrosoftAgentFramework.Testing;

var fake = new FakeChatClient(
    ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })),
    ScriptedResponse.Text("Sunny."));
var plugin = new MicrosoftAgentFrameworkPlugin(new MicrosoftAgentFrameworkPluginOptions
{
    ChatClients = new Dictionary<string, Func<IChatClient>> { ["gpt-4o-mini"] = () => fake },
});
```

Each model call takes the next scripted response; the last one repeats. `fake.Calls` holds every request the model received, with tools as declarations. `ScriptedResponse.Fail(HttpStatusCode)` stands in for a provider error and `InMemoryMcpServer` for an MCP server.

## Develop

```sh
cd dotnet/microsoft-agent-framework
make sync
make lint
make test
make build
```

`make test` builds in Release and runs the suite under Microsoft.Testing.Platform with a local Temporal dev server that `WorkflowEnvironment.StartLocalAsync` downloads on first run. No provider credentials are needed: the suite uses the fakes from the Testing section. The replay test reads the histories under `tests/Temporalio.Extensions.MicrosoftAgentFramework.Tests/Histories`; set `MICROSOFT_AGENT_FRAMEWORK_RECORD_HISTORIES=1` while running the suite to write fresh copies under `TestResults/recorded-histories`.
