using Microsoft.Extensions.AI;
using Temporalio.Api.Enums.V1;
using Temporalio.Client;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class AgentRunTests : WorkflowTestBase
{
    public AgentRunTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task RunAsync_InWorkflowTool_TwoModelActivities_NoToolActivity()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston"))),
            ScriptedResponse.Text("Final: sunny in Kingston"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather in Kingston?", "in-workflow")));
        await HistoryRecorder.SaveAsync(handle, "tools-in-workflow");

        Assert.Equal("Final: sunny in Kingston", result.Text);
        var counts = HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync());
        Assert.Equal(2, counts[ModelActivityNames.InvokeModel]);
        Assert.Single(counts);
        Assert.Equal(2, fake.CallCount);
        var declaration = Assert.Single(fake.Calls[0].Options!.Tools!);
        Assert.Equal("get_weather", declaration.Name);
        Assert.Contains("city", ((AIFunctionDeclaration)declaration).JsonSchema.GetRawText(), StringComparison.Ordinal);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("call_1_0", toolResult.CallId);
        Assert.Equal("sunny in Kingston (workflow)", toolResult.Result?.ToString());
    }

    [Fact]
    public async Task RunAsync_ActivityAsTool_Expression_RunsToolAsActivity()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston"))),
            ScriptedResponse.Text("Final"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", "activity-expression")));
        var history = await handle.FetchHistoryAsync();
        var counts = HistoryAssert.ActivityCounts(history);
        Assert.Equal(2, counts[ModelActivityNames.InvokeModel]);
        Assert.Equal(1, counts["GetWeather"]);
        var scheduled = Assert.Single(HistoryAssert.Scheduled(history, "GetWeather"));
        Assert.Equal("get_weather", HistoryAssert.Summary(scheduled, Client));
        Assert.Equal(TimeSpan.FromSeconds(30), scheduled.ActivityTaskScheduledEventAttributes!.StartToCloseTimeout.ToTimeSpan());
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("sunny in Kingston (activity)", toolResult.Result?.ToString());
        Assert.Equal("Final", result.Text);
    }

    [Theory]
    [InlineData("activity-static", "LookupTimeZone")]
    [InlineData("activity-delegate", "LookupTimeZone")]
    [InlineData("activity-methodinfo", "GetWeather")]
    public async Task RunAsync_ActivityAsTool_OtherOverloads_RunToolAsActivity(string kind, string activityType)
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston"))),
            ScriptedResponse.Text("Final"));
        var (_, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", kind)));
        var counts = HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync());
        Assert.Equal(1, counts[activityType]);
        Assert.Equal(2, counts[ModelActivityNames.InvokeModel]);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Contains("Kingston", toolResult.Result?.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task RunAsync_ActivityAsTool_TypedResult_ReachesModel()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_report", new Dictionary<string, object?> { ["city"] = "Kingston", ["days"] = 3 })),
            ScriptedResponse.Text("Final"));
        var (_, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Report?", "activity-report")));
        Assert.Equal(1, HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync())["GetReport"]);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        var json = System.Text.Json.JsonSerializer.Serialize(toolResult.Result, AIJsonUtilities.DefaultOptions);
        Assert.Contains("sunny for 3 days", json, StringComparison.Ordinal);
        var schema = ((AIFunctionDeclaration)fake.Calls[0].Options!.Tools![0]).JsonSchema.GetRawText();
        Assert.Contains("days", schema, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RunAsync_ActivityAsTool_TrailingToken_LeftOutOfSchemaAndCall()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston"))),
            ScriptedResponse.Text("Final"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Alerts?", "activity-token")));
        Assert.Equal("Final", result.Text);
        Assert.Equal(1, HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync())["GetAlerts"]);
        var schema = ((AIFunctionDeclaration)fake.Calls[0].Options!.Tools![0]).JsonSchema.GetRawText();
        Assert.Contains("city", schema, StringComparison.Ordinal);
        Assert.DoesNotContain("cancellationToken", schema, StringComparison.OrdinalIgnoreCase);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("no alerts for Kingston", toolResult.Result?.ToString());
    }

    [Fact]
    public async Task RunAsync_ConcurrentTools_BothScheduledBeforeEitherCompletes()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston")), ("get_weather", City("Toronto"))),
            ScriptedResponse.Text("Final"));
        var input = new ToolsInput("Weather?", "activity-expression", Concurrent: true);
        var (_, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(input));
        await HistoryRecorder.SaveAsync(handle, "tools-concurrent");
        var history = await handle.FetchHistoryAsync();
        var events = history.Events.ToList();
        var scheduled = events.Where(e => e.ActivityTaskScheduledEventAttributes?.ActivityType.Name == "GetWeather").ToList();
        Assert.Equal(2, scheduled.Count);
        var scheduledIds = scheduled.Select(e => e.EventId).ToHashSet();
        var firstCompleted = events.First(e => e.ActivityTaskCompletedEventAttributes is { } c && scheduledIds.Contains(c.ScheduledEventId));
        Assert.All(scheduled, e => Assert.True(events.IndexOf(e) < events.IndexOf(firstCompleted)));
        var results = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Select(r => r.CallId).ToList();
        Assert.Equal(["call_1_0", "call_1_1"], results);
    }

    [Fact]
    public async Task RunAsync_SequentialTools_SecondScheduledAfterFirstCompletes()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston")), ("get_weather", City("Toronto"))),
            ScriptedResponse.Text("Final"));
        var (_, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", "activity-expression")));
        var events = (await handle.FetchHistoryAsync()).Events.ToList();
        var scheduled = events.Where(e => e.ActivityTaskScheduledEventAttributes?.ActivityType.Name == "GetWeather").ToList();
        Assert.Equal(2, scheduled.Count);
        var firstCompleted = events.First(e => e.ActivityTaskCompletedEventAttributes is { } c && c.ScheduledEventId == scheduled[0].EventId);
        Assert.True(events.IndexOf(firstCompleted) < events.IndexOf(scheduled[1]));
    }

    [Theory]
    [InlineData(null, null, "temporal-agent")]
    [InlineData("ops-agent", "ops", "ops-agent")]
    [InlineData(null, "ops", "ops")]
    public async Task RunAsync_AgentId_IsStableAcrossReplay(string? id, string? name, string expectedId)
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", City("Kingston"))),
            ScriptedResponse.Text("Final"));
        var input = new ToolsInput("Weather?", "in-workflow", AgentId: id, AgentName: name);
        var (result, _) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(input), options => options.MaxCachedWorkflows = 0);
        Assert.Equal(expectedId, result.AgentId);
        Assert.Equal(3, result.MessageCount);
        Assert.Equal(name ?? string.Empty, result.AuthorNames[0]);
        Assert.Equal(30, result.TotalTokens);
    }

    [Fact]
    public async Task RunAsync_TwoAgentsTwoModels_RouteToTheirClients()
    {
        var alpha = new FakeChatClient(ScriptedResponse.Text("from alpha"));
        var beta = new FakeChatClient(ScriptedResponse.Text("from beta"));
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { ["alpha"] = alpha, ["beta"] = beta });
        var (result, handle) = await RunAsync(plugin, (TwoAgentsWorkflow wf) => wf.RunAsync("hi"));
        Assert.Equal("from alpha|from beta", result);
        Assert.Equal(1, alpha.CallCount);
        Assert.Equal(1, beta.CallCount);
        var summaries = HistoryAssert.Scheduled(await handle.FetchHistoryAsync()).Select(e => HistoryAssert.Summary(e, Client)).ToList();
        Assert.Equal(["alpha", "beta"], summaries);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TemporalChatClient_Direct_OneActivity(bool streaming)
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("direct answer"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (DirectChatClientWorkflow wf) => wf.RunAsync(new DirectInput("hi", streaming)));
        Assert.Equal("direct answer", result);
        var history = await handle.FetchHistoryAsync();
        Assert.Equal([ModelActivityNames.InvokeModel], HistoryAssert.ActivityTypesInOrder(history));
    }

    [Fact]
    public async Task ModelActivityOptions_AppearInHistory()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("ok"));
        var (_, handle) = await RunAsync(TestPlugin.For(fake), (ModelOptionsWorkflow wf) => wf.RunAsync(new ModelOptionsInput("hi", 42, "custom summary", 7)));
        var scheduled = Assert.Single(HistoryAssert.Scheduled(await handle.FetchHistoryAsync()));
        var attributes = scheduled.ActivityTaskScheduledEventAttributes!;
        Assert.Equal(TimeSpan.FromSeconds(42), attributes.StartToCloseTimeout.ToTimeSpan());
        Assert.Equal(TimeSpan.FromSeconds(7), attributes.HeartbeatTimeout.ToTimeSpan());
        Assert.Equal("custom summary", HistoryAssert.Summary(scheduled, Client));
    }

    [Fact]
    public async Task ModelActivityDefaults_AppearInHistory()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("ok"));
        var (_, handle) = await RunAsync(TestPlugin.For(fake), (ModelOptionsWorkflow wf) => wf.RunAsync(new ModelOptionsInput("hi")), runTimeout: TimeSpan.FromMinutes(10));
        var scheduled = Assert.Single(HistoryAssert.Scheduled(await handle.FetchHistoryAsync()));
        var attributes = scheduled.ActivityTaskScheduledEventAttributes!;
        Assert.Equal(TimeSpan.FromMinutes(5), attributes.StartToCloseTimeout.ToTimeSpan());
        Assert.Equal(TimeSpan.FromSeconds(10), attributes.HeartbeatTimeout.ToTimeSpan());
        Assert.Equal("fake", HistoryAssert.Summary(scheduled, Client));
        Assert.Equal(0, HistoryAssert.Count(await handle.FetchHistoryAsync(), EventType.WorkflowTaskFailed));
    }

    [Fact]
    public async Task HostedToolsAndAdditionalProperties_ReachProviderClient()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("searched"));
        var (result, _) = await RunAsync(TestPlugin.For(fake), (HostedToolWorkflow wf) => wf.RunAsync(new HostedToolInput("search the docs")));
        Assert.Equal("searched", result);
        var properties = fake.Calls[0].Options!.AdditionalProperties!;
        Assert.Equal(true, properties["strict"]);
        Assert.Equal(7L, properties["seed"]);
        Assert.Equal("ops", properties["user"]);
        var tools = fake.Calls[0].Options!.Tools!;
        Assert.Contains(tools, tool => tool is HostedWebSearchTool);
        var mcp = Assert.IsType<HostedMcpServerTool>(tools.Single(tool => tool is HostedMcpServerTool));
        Assert.Equal("docs", mcp.ServerName);
        Assert.Equal("https://example.com/mcp", mcp.ServerAddress);
        Assert.Equal("Documentation search", mcp.ServerDescription);
        Assert.Equal(["search"], mcp.AllowedTools);
        Assert.IsType<HostedMcpServerToolNeverRequireApprovalMode>(mcp.ApprovalMode);
        Assert.Null(mcp.Headers);
    }

    [Fact]
    public async Task HostedMcpServerHeaders_AddedOnTheWorker_NeverRecorded()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("searched"));
        var plugin = TestPlugin.For(
            new Dictionary<string, IChatClient> { [TestPlugin.Model] = fake },
            hostedMcpServerHeaders: new Dictionary<string, IReadOnlyDictionary<string, string>>
            {
                ["docs"] = new Dictionary<string, string> { ["Authorization"] = "Bearer worker-side-secret" },
            });
        var (result, handle) = await RunAsync(plugin, (HostedToolWorkflow wf) => wf.RunAsync(new HostedToolInput("search the docs")));
        Assert.Equal("searched", result);
        var mcp = Assert.IsType<HostedMcpServerTool>(fake.Calls[0].Options!.Tools!.Single(tool => tool is HostedMcpServerTool));
        Assert.Equal("Bearer worker-side-secret", mcp.Headers!["Authorization"]);
        var text = HistoryAssert.AllText(await handle.FetchHistoryAsync());
        Assert.Contains("example.com/mcp", text, StringComparison.Ordinal);
        Assert.DoesNotContain("worker-side-secret", text, StringComparison.Ordinal);
        Assert.DoesNotContain("Authorization", text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task HostedMcpServerHeaderProviders_AreCalledBeforeEveryModelCall()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("searched"));
        var calls = 0;
        var plugin = TestPlugin.For(
            new Dictionary<string, IChatClient> { [TestPlugin.Model] = fake },
            hostedMcpServerHeaderProviders: new Dictionary<string, Func<IReadOnlyDictionary<string, string>>>
            {
                ["docs"] = () => new Dictionary<string, string> { ["Authorization"] = $"Bearer token-{Interlocked.Increment(ref calls)}" },
            });
        using var worker = Fixture.NewWorker(plugin);
        await WithWorkerAsync(worker, async () =>
        {
            for (var i = 1; i <= 2; i++)
            {
                var input = new HostedToolInput("search the docs");
                await Client.ExecuteWorkflowAsync((HostedToolWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, $"hosted-provider-{i}"));
            }
        });

        var headers = fake.Calls.Select(call => ((HostedMcpServerTool)call.Options!.Tools!.Single(tool => tool is HostedMcpServerTool)).Headers!["Authorization"]);
        Assert.Equal(["Bearer token-1", "Bearer token-2"], headers);
    }

    [Fact]
    public async Task FunctionAdditionalProperties_ReachTheProviderClient()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("ok"));
        await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", "in-workflow-strict")));
        var declaration = Assert.IsAssignableFrom<AIFunctionDeclaration>(Assert.Single(fake.Calls[0].Options!.Tools!));
        Assert.Equal("get_weather", declaration.Name);
        Assert.Equal(true, declaration.AdditionalProperties["strict"]);
    }

    [Fact]
    public async Task ActivityTool_MissingRequiredArgument_FailsTheCallBeforeAnyActivity()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?>())),
            ScriptedResponse.Text("Final"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", "activity-expression")));
        Assert.Equal("Final", result.Text);
        Assert.False(HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync()).ContainsKey("GetWeather"));
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("Error: Function failed.", toolResult.Result?.ToString());
    }

    [Fact]
    public async Task HostedMcpServerTool_WithHeadersInTheWorkflow_FailsTheRunBeforeAnyActivity()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("never"));
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (HostedToolWorkflow wf) => wf.RunAsync(new HostedToolInput("search the docs", WorkflowSideHeaders: true)));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Contains("HostedMcpServerTool 'docs' has 1 HTTP header(s) set in the Workflow", failure.Message, StringComparison.Ordinal);
        Assert.Contains("MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaders[\"docs\"]", failure.Message, StringComparison.Ordinal);
        Assert.Equal(0, fake.CallCount);
        var history = await handle.FetchHistoryAsync();
        Assert.Empty(HistoryAssert.Scheduled(history));
        var text = HistoryAssert.AllText(history);
        Assert.DoesNotContain("workflow-side-secret", text, StringComparison.Ordinal);
        Assert.DoesNotContain("Authorization", text, StringComparison.Ordinal);
    }

    private static Dictionary<string, object?> City(string city) => new() { ["city"] = city };
}
