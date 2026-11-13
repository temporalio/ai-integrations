using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using Temporalio.Api.Enums.V1;
using Temporalio.Client;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class McpTests : WorkflowTestBase
{
    private const string Server = "calc";

    public McpTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task ListTools_InActivity_CallTool_ViaProxy_FullSchema()
    {
        await using var mcp = NewServer();
        var fake = AddFake();
        var (result, handle) = await RunAsync(TestPlugin.WithMcp(fake, Server, mcp), (McpWorkflow wf) => wf.RunAsync(new McpInput(Server, "add 2 and 3")));
        await HistoryRecorder.SaveAsync(handle, "mcp");
        Assert.Equal("Final: 5", result.Text);
        Assert.Equal(["add", "fail", "subtract"], result.ToolNames);
        Assert.Contains("integer", result.AddSchema, StringComparison.Ordinal);
        Assert.Equal(
            [ModelActivityNames.McpListTools, ModelActivityNames.InvokeModel, ModelActivityNames.McpCallTool, ModelActivityNames.InvokeModel],
            HistoryAssert.ActivityTypesInOrder(await handle.FetchHistoryAsync()));
        Assert.Equal(1, mcp.CallToolRequestCount);
        var advertised = fake.Calls[0].Options!.Tools!.Cast<AIFunctionDeclaration>().Single(tool => tool.Name == "add");
        Assert.Contains("integer", advertised.JsonSchema.GetRawText(), StringComparison.Ordinal);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Contains("5", toolResult.Result?.ToString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 1)]
    public async Task CacheTools_ControlsServerListRequests(bool cacheTools, int expectedListRequests)
    {
        await using var mcp = NewServer();
        var input = new McpInput(Server, "add", CacheTools: cacheTools, ListTwice: true);
        var (_, handle) = await RunAsync(TestPlugin.WithMcp(AddFake(), Server, mcp), (McpWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(expectedListRequests, mcp.ListToolsRequestCount);
        Assert.Equal(2, HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync())[ModelActivityNames.McpListTools]);
    }

    [Fact]
    public async Task ThrowingTransportFactory_ConfigurationError_RetriedUnderThePolicy()
    {
        var attempts = 0;
        var plugin = new MicrosoftAgentFrameworkPlugin(new MicrosoftAgentFrameworkPluginOptions
        {
            ChatClients = new Dictionary<string, Func<IChatClient>> { [TestPlugin.Model] = () => AddFake() },
            McpServers = new Dictionary<string, Func<IClientTransport>>
            {
                [Server] = () =>
                {
                    attempts++;
                    throw new InvalidOperationException("MCP_URL is not set");
                },
            },
        });
        var input = new McpInput(Server, "add 2 and 3", MaxAttempts: 3);
        var (error, handle) = await RunFailingAsync(plugin, (McpWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.ConfigurationError, failure.ErrorType);
        Assert.False(failure.NonRetryable);
        Assert.Contains("InvalidOperationException: MCP_URL is not set", failure.Message, StringComparison.Ordinal);
        Assert.Equal(3, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.McpListTools));
        Assert.Equal(3, attempts);
    }

    [Fact]
    public async Task NullTransportFactory_ConfigurationError_NoRetry()
    {
        var plugin = new MicrosoftAgentFrameworkPlugin(new MicrosoftAgentFrameworkPluginOptions
        {
            ChatClients = new Dictionary<string, Func<IChatClient>> { [TestPlugin.Model] = () => AddFake() },
            McpServers = new Dictionary<string, Func<IClientTransport>> { [Server] = () => null! },
        });
        var input = new McpInput(Server, "add 2 and 3", MaxAttempts: 3);
        var (error, handle) = await RunFailingAsync(plugin, (McpWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.ConfigurationError, failure.ErrorType);
        Assert.True(failure.NonRetryable);
        Assert.Contains("returned null", failure.Message, StringComparison.Ordinal);
        Assert.Equal(1, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.McpListTools));
    }

    [Fact]
    public async Task IdleTimeout_EvictsConnection_Reconnects()
    {
        await using var mcp = NewServer();
        var fake = new FakeChatClient(ScriptedResponse.Text("no tools needed"));
        using var worker = Fixture.NewWorker(TestPlugin.WithMcp(fake, Server, mcp, TimeSpan.FromSeconds(1)));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new McpInput(Server, "hi");
            await Client.ExecuteWorkflowAsync((McpWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "mcp-idle-1"));
            Assert.Equal(1, mcp.ConnectionCount);
            await Task.Delay(TimeSpan.FromSeconds(3), TestContext.Current.CancellationToken);
            await Client.ExecuteWorkflowAsync((McpWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "mcp-idle-2"));
            Assert.Equal(2, mcp.ConnectionCount);
        });
    }

    [Fact]
    public async Task TwoWorkersOnePlugin_StoppingOneLeavesTheOthersCallRunning()
    {
        await using var mcp = new InMemoryMcpServer(
            McpServerTool.Create(
                async (int a, int b) =>
                {
                    await Task.Delay(TimeSpan.FromSeconds(2));
                    return a + b;
                },
                new McpServerToolCreateOptions { Name = "add", Description = "Adds two integers after a pause" }));
        var fake = new FakeChatClient(call => call.Messages.Any(message => message.Contents.OfType<FunctionResultContent>().Any())
            ? new ChatResponse(new ChatMessage(ChatRole.Assistant, "Final: 5"))
            : new ChatResponse(new ChatMessage(ChatRole.Assistant, [new FunctionCallContent("call_add", "add", new Dictionary<string, object?> { ["a"] = 2, ["b"] = 3 })])));
        var plugin = TestPlugin.WithMcp(fake, Server, mcp);
        var input = new McpInput(Server, "add 2 and 3");
        using var first = Fixture.NewWorker(plugin);
        using var second = Fixture.NewWorker(plugin);
        var stopFirst = new TaskCompletionSource();
        var firstRun = first.ExecuteAsync(() => stopFirst.Task, TestContext.Current.CancellationToken);
        try
        {
            await WithWorkerAsync(second, async () =>
            {
                // The first worker opens the shared connection.
                await Client.ExecuteWorkflowAsync((McpWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(first, "mcp-shared-first"));
                Assert.Equal(1, mcp.ConnectionCount);

                // The second worker's tool call is on the wire when the first worker stops.
                var handle = await Client.StartWorkflowAsync((McpWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(second, "mcp-shared-second"));
                while (mcp.CallToolRequestCount < 2)
                {
                    await Task.Delay(20, TestContext.Current.CancellationToken);
                }

                stopFirst.SetResult();
                await firstRun;

                var result = await handle.GetResultAsync();
                Assert.Equal("Final: 5", result.Text);
                var history = await handle.FetchHistoryAsync();
                Assert.Equal(0, HistoryAssert.Count(history, EventType.ActivityTaskFailed));
                Assert.Equal(1, HistoryAssert.LastAttempt(history, ModelActivityNames.McpCallTool));

                // The connection the first worker opened stays open for the second worker's later calls.
                await Client.ExecuteWorkflowAsync((McpWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(second, "mcp-shared-second-again"));
                Assert.Equal(1, mcp.ConnectionCount);
            });
        }
        finally
        {
            stopFirst.TrySetResult();
            await firstRun;
        }

        // Both runs have ended, so the connection was closed; the next worker reconnects.
        using var third = Fixture.NewWorker(plugin);
        await WithWorkerAsync(third, () => Client.ExecuteWorkflowAsync((McpWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(third, "mcp-shared-third")));
        Assert.Equal(2, mcp.ConnectionCount);
    }

    [Fact]
    public async Task AllowedToolsAndPrefix_FilterAndRename()
    {
        await using var mcp = NewServer();
        var fake = AddFake("calc_add");
        var input = new McpInput(Server, "add", AllowedTools: ["add"], ToolNamePrefix: "calc_");
        var (result, _) = await RunAsync(TestPlugin.WithMcp(fake, Server, mcp), (McpWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(["calc_add"], result.ToolNames);
        Assert.Equal("Final: 5", result.Text);
        Assert.Equal(1, mcp.CallToolRequestCount);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Contains("5", toolResult.Result?.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task PinnedCatalog_CallsWithoutListing()
    {
        await using var mcp = NewServer();
        var input = new McpInput(Server, "add", PinnedCatalog: true);
        var (result, handle) = await RunAsync(TestPlugin.WithMcp(AddFake(), Server, mcp), (McpWorkflow wf) => wf.RunAsync(input));
        Assert.Equal("Final: 5", result.Text);
        Assert.Equal(0, mcp.ListToolsRequestCount);
        Assert.Equal([ModelActivityNames.InvokeModel, ModelActivityNames.McpCallTool, ModelActivityNames.InvokeModel], HistoryAssert.ActivityTypesInOrder(await handle.FetchHistoryAsync()));
    }

    [Fact]
    public async Task ToolError_NonRetryable_McpError_ModelToldFunctionFailed()
    {
        await using var mcp = NewServer();
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("fail", new Dictionary<string, object?>())),
            ScriptedResponse.Text("Recovered"));
        var (result, handle) = await RunAsync(TestPlugin.WithMcp(fake, Server, mcp), (McpWorkflow wf) => wf.RunAsync(new McpInput(Server, "fail")));
        Assert.Equal("Recovered", result.Text);
        var history = await handle.FetchHistoryAsync();
        var failed = Assert.Single(history.Events, e => e.ActivityTaskFailedEventAttributes is not null);
        var info = failed.ActivityTaskFailedEventAttributes!.Failure.ApplicationFailureInfo;
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.McpError, info.Type);
        Assert.True(info.NonRetryable);
        Assert.Equal(1, HistoryAssert.LastAttempt(history, ModelActivityNames.McpCallTool));
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("Error: Function failed.", toolResult.Result?.ToString());
    }

    [Fact]
    public async Task ServerDown_RetriedToTheLimit_ThenWorkflowFails()
    {
        var plugin = TestPlugin.For(
            new Dictionary<string, IChatClient> { [TestPlugin.Model] = AddFake() },
            new Dictionary<string, IClientTransport> { [Server] = new BrokenTransport() });
        var input = new McpInput(Server, "add", MaxAttempts: 2);
        var (error, handle) = await RunFailingAsync(plugin, (McpWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal("IOException", failure.ErrorType);
        Assert.Equal(2, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.McpListTools));
    }

    [Fact]
    public async Task UnknownServer_ConfigurationError()
    {
        await using var mcp = NewServer();
        var input = new McpInput("nope", "add", MaxAttempts: 3);
        var (error, handle) = await RunFailingAsync(TestPlugin.WithMcp(AddFake(), Server, mcp), (McpWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.ConfigurationError, failure.ErrorType);
        Assert.True(failure.NonRetryable);
        Assert.Equal(1, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.McpListTools));
    }

    private sealed class BrokenTransport : IClientTransport
    {
        public string Name => "broken";

        public Task<ITransport> ConnectAsync(CancellationToken cancellationToken = default) =>
            throw new IOException("connection refused");
    }

    private static InMemoryMcpServer NewServer() => new(
        McpServerTool.Create((int a, int b) => a + b, new McpServerToolCreateOptions { Name = "add", Description = "Adds two integers" }),
        McpServerTool.Create((int a, int b) => a - b, new McpServerToolCreateOptions { Name = "subtract", Description = "Subtracts b from a" }),
        McpServerTool.Create((Func<string>)(() => throw new InvalidOperationException("broken tool")), new McpServerToolCreateOptions { Name = "fail", Description = "Always fails" }));

    private static FakeChatClient AddFake(string toolName = "add") => new(
        ScriptedResponse.ToolCalls((toolName, new Dictionary<string, object?> { ["a"] = 2, ["b"] = 3 })),
        ScriptedResponse.Text("Final: 5"));
}
