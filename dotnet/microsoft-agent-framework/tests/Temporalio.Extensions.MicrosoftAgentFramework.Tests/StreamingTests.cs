using System.Text.Json;
using Microsoft.Extensions.AI;
using Temporalio.Api.Enums.V1;
using Temporalio.Extensions.MicrosoftAgentFramework.Testing;
using Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Extensions.WorkflowStreams;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

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
    public async Task RunStreamingAsync_UsageSummedOnceAcrossRounds()
    {
        var call = ("get_weather", (IDictionary<string, object?>)new Dictionary<string, object?> { ["city"] = "Kingston" });
        var fake = new FakeChatClient(ScriptedResponse.ToolCalls(call), ScriptedResponse.ToolCalls(call), ScriptedResponse.Text("Final"));
        var input = new StreamingInput("Weather?", WithTool: true);
        var (result, _) = await RunAsync(TestPlugin.For(fake), (StreamingWorkflow wf) => wf.RunAsync(input));
        Assert.Equal("Final", result.FinalText);
        Assert.Equal(3, fake.CallCount);
        Assert.Equal(45, result.TotalTokens);
    }

    [Fact]
    public async Task StreamingTopic_PublishesUpdatesFromActivity()
    {
        const string text = "The quick brown fox jumps over the lazy dog";
        var fake = new FakeChatClient(ScriptedResponse.Text(text).WithStreamingDelay(TimeSpan.FromMilliseconds(100)));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await Client.StartWorkflowAsync(
                (StreamingWorkflow wf) => wf.RunAsync(new StreamingInput("hi", "deltas")),
                Fixture.WorkflowOptions(worker, "streaming-topic"));
            var received = new List<ChatResponseUpdate>();
            var activityDoneAtFirstItem = true;
            await using var streams = new WorkflowStreamClient(Client, handle.Id);
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(30));
            var items = streams.SubscribeAsync<ChatResponseUpdate>(new WorkflowStreamSubscribeOptions { Topics = ["deltas"] });
            await foreach (var item in items.WithCancellation(timeout.Token))
            {
                if (received.Count == 0)
                {
                    // The stream is nine words at 100 ms each; the first batch must arrive while the Activity still runs.
                    activityDoneAtFirstItem = HistoryAssert.Count(await handle.FetchHistoryAsync(), EventType.ActivityTaskCompleted) > 0;
                }

                received.Add(item.Value);
                if (string.Concat(received.Select(update => update.Text)) == text)
                {
                    break;
                }
            }

            var result = await handle.GetResultAsync();
            Assert.Equal(text, result.FinalText);
            Assert.False(activityDoneAtFirstItem, "the first update arrived only after the model Activity completed");
            Assert.True(received.Count > 1, $"expected several streamed updates, got {received.Count}");
            Assert.Equal(text, string.Concat(received.Select(update => update.Text)));
            await HistoryRecorder.SaveAsync(handle, "streaming");
        });
    }

    [Fact]
    public async Task StreamingTopic_WithoutAHostedStream_FailsTheRunBeforeTheModelCall()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("never"));
        var input = new StreamingInput("hi", "deltas", HostStream: false);
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (StreamingWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Contains("hosts no WorkflowStream", failure.Message, StringComparison.Ordinal);
        Assert.Equal(0, fake.CallCount);
        Assert.Empty(HistoryAssert.Scheduled(await handle.FetchHistoryAsync()));
    }

    [Fact]
    public async Task StreamingTopic_LongStream_FlushesAtMaxBatchSize()
    {
        var text = string.Join(' ', Enumerable.Range(1, 300).Select(i => $"w{i}"));
        var fake = new FakeChatClient(ScriptedResponse.Text(text).WithStreamingDelay(TimeSpan.FromMilliseconds(1)));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new StreamingInput("hi", "deltas", BatchInterval: TimeSpan.FromSeconds(30));
            var handle = await Client.StartWorkflowAsync((StreamingWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "streaming-batches"));
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

            // 301 updates (300 text pieces and the usage) against a bound of 256. The bound wakes the publisher's
            // background flusher, which takes the whole buffer, so the first Signal carries at least 256 updates and
            // possibly more (items published before the swap join it); a stale wake can add a Signal for the items
            // that arrived during the first Signal's RPC, so the count is at least two. The 30-second interval timer
            // starts on the first publish and cannot fire inside the subscriber's own 30-second window.
            var history = await handle.FetchHistoryAsync();
            var batches = new List<int>();
            foreach (var signaled in history.Events.Select(e => e.WorkflowExecutionSignaledEventAttributes).Where(a => a?.SignalName == WorkflowStreamConstants.PublishSignalName))
            {
                // The publish Signal's input is the extension's internal batch record; its items array is all the test needs.
                var batch = (JsonElement)Client.Options.DataConverter.PayloadConverter.ToValue(signaled!.Input.Payloads_.Single(), typeof(JsonElement))!;
                var entries = batch.EnumerateObject().Single(property => string.Equals(property.Name, "items", StringComparison.OrdinalIgnoreCase)).Value;
                batches.Add(entries.GetArrayLength());
            }

            Assert.True(batches.Count >= 2, $"the size bound must split the stream into at least two Signals, got {batches.Count}");
            Assert.True(batches[0] >= Activities.ModelActivities.MaxStreamingBatchSize, $"the first Signal is the size-triggered flush and carries at least the bound, got {batches[0]}");
            Assert.Equal(301, batches.Sum());
        });
    }
}
