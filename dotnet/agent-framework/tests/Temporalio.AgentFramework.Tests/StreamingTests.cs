using Microsoft.Extensions.AI;
using Temporalio.AgentFramework.Testing;
using Temporalio.AgentFramework.Tests.Workflows;
using Temporalio.Extensions.WorkflowStreams;

namespace Temporalio.AgentFramework.Tests;

public class StreamingTests : WorkflowTestBase
{
    public StreamingTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task RunStreamingAsync_YieldsUpdates_OneActivityPerModelCall()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("Hello there from the fake"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (StreamingWorkflow wf) => wf.RunAsync(new StreamingInput("hi")));
        Assert.Equal("Hello there from the fake", result.FinalText);
        Assert.True(result.UpdateCount >= 1);
        Assert.Equal([ModelActivityNames.InvokeModel], HistoryAssert.ActivityTypesInOrder(await handle.FetchHistoryAsync()));
        Assert.Single(fake.Calls);
    }

    [Fact]
    public async Task RunStreamingAsync_WithTools_ToolCallsRunBetweenRounds()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })),
            ScriptedResponse.Text("Final: sunny"));
        var input = new StreamingInput("Weather?", WithTool: true);
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (StreamingWorkflow wf) => wf.RunAsync(input));
        Assert.Equal("Final: sunny", result.FinalText);
        Assert.Equal(1, result.ToolUpdates);
        Assert.Equal(2, HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync())[ModelActivityNames.InvokeModel]);
    }

    [Fact]
    public async Task StreamingTopic_PublishesUpdatesFromActivity()
    {
        const string text = "The quick brown fox jumps over the lazy dog";
        var fake = new FakeChatClient(ScriptedResponse.Text(text));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await Client.StartWorkflowAsync(
                (StreamingWorkflow wf) => wf.RunAsync(new StreamingInput("hi", "deltas")),
                Fixture.WorkflowOptions(worker, "streaming-topic"));
            var received = new List<ChatResponseUpdate>();
            await using var streams = new WorkflowStreamClient(Client, handle.Id);
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(30));
            var items = streams.SubscribeAsync<ChatResponseUpdate>(new WorkflowStreamSubscribeOptions { Topics = ["deltas"] });
            await foreach (var item in items.WithCancellation(timeout.Token))
            {
                received.Add(item.Value);
                if (string.Concat(received.Select(update => update.Text)) == text)
                {
                    break;
                }
            }

            var result = await handle.GetResultAsync();
            Assert.Equal(text, result.FinalText);
            Assert.True(received.Count > 1, $"expected several streamed updates, got {received.Count}");
            Assert.Equal(text, string.Concat(received.Select(update => update.Text)));
            await HistoryRecorder.SaveAsync(handle, "streaming");
        });
    }
}
