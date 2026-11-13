using ModelContextProtocol.Server;
using Temporalio.Api.Enums.V1;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class SideEffectTests : WorkflowTestBase
{
    public SideEffectTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task MaxCachedWorkflowsZero_ExactActivityCounts()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(
                ("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" }),
                ("get_weather", new Dictionary<string, object?> { ["city"] = "Toronto" })),
            ScriptedResponse.Text("Final"));
        var (result, handle) = await RunAsync(
            TestPlugin.For(fake),
            (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", "activity-expression", Concurrent: true)),
            options => options.MaxCachedWorkflows = 0);
        Assert.Equal("Final", result.Text);
        var history = await handle.FetchHistoryAsync();
        var counts = HistoryAssert.ActivityCounts(history);
        Assert.Equal(2, counts[ModelActivityNames.InvokeModel]);
        Assert.Equal(2, counts["GetWeather"]);
        Assert.Equal(2, counts.Count);
        Assert.True(HistoryAssert.Count(history, EventType.WorkflowTaskCompleted) >= 3);
        Assert.Equal(0, HistoryAssert.Count(history, EventType.WorkflowTaskFailed));
        Assert.Equal(2, fake.CallCount);
    }

    [Fact]
    public async Task MaxCachedWorkflowsZero_McpActivityCounts()
    {
        await using var mcp = new InMemoryMcpServer(
            McpServerTool.Create((int a, int b) => a + b, new McpServerToolCreateOptions { Name = "add", Description = "Adds two integers" }));
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("add", new Dictionary<string, object?> { ["a"] = 2, ["b"] = 3 })),
            ScriptedResponse.Text("Final: 5"));
        var (result, handle) = await RunAsync(
            TestPlugin.WithMcp(fake, "calc", mcp),
            (McpWorkflow wf) => wf.RunAsync(new McpInput("calc", "add 2 and 3")),
            options => options.MaxCachedWorkflows = 0);
        Assert.Equal("Final: 5", result.Text);
        var counts = HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync());
        Assert.Equal(1, counts[ModelActivityNames.McpListTools]);
        Assert.Equal(1, counts[ModelActivityNames.McpCallTool]);
        Assert.Equal(2, counts[ModelActivityNames.InvokeModel]);
        Assert.Equal(1, mcp.CallToolRequestCount);
    }
}
