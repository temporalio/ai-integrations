using Temporalio.AgentFramework.Testing;
using Temporalio.AgentFramework.Tests.Workflows;
using Temporalio.Api.Enums.V1;

namespace Temporalio.AgentFramework.Tests;

public class HeartbeatTests : WorkflowTestBase
{
    public HeartbeatTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task ModelActivity_HeartbeatsDuringLongCall_WithDefaultTimeout()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("slow answer").WithDelay(TimeSpan.FromSeconds(12)));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (ModelOptionsWorkflow wf) => wf.RunAsync(new ModelOptionsInput("hi")));
        Assert.Equal("slow answer", result);
        var history = await handle.FetchHistoryAsync();
        Assert.Equal(1, HistoryAssert.LastAttempt(history, ModelActivityNames.InvokeModel));
        Assert.Equal(0, HistoryAssert.Count(history, EventType.ActivityTaskTimedOut));
        Assert.Equal(1, fake.CallCount);
    }

    [Fact]
    public async Task ModelActivity_HeartbeatsDuringLongCall_WithShortTimeout()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("slow answer").WithDelay(TimeSpan.FromSeconds(3)));
        var input = new ModelOptionsInput("hi", HeartbeatSeconds: 1);
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (ModelOptionsWorkflow wf) => wf.RunAsync(input));
        Assert.Equal("slow answer", result);
        var history = await handle.FetchHistoryAsync();
        Assert.Equal(1, HistoryAssert.LastAttempt(history, ModelActivityNames.InvokeModel));
        Assert.Equal(0, HistoryAssert.Count(history, EventType.ActivityTaskTimedOut));
    }
}
