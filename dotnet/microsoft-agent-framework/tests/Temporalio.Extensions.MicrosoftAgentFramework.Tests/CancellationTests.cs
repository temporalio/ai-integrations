using Temporalio.Api.Enums.V1;
using Temporalio.Api.WorkflowService.V1;
using Temporalio.Client;
using Temporalio.Exceptions;
using Temporalio.Extensions.MicrosoftAgentFramework.Testing;
using Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>
/// A cancelled Workflow cancels the Activity that is running and makes no further model call. The
/// Workflow waits for the cancellation to be confirmed, so the Activity's own cancellation is in history.
/// </summary>
public class CancellationTests : WorkflowTestBase
{
    public CancellationTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task CancelDuringModelCall_CancelsTheActivity()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("slow").WithDelay(TimeSpan.FromSeconds(60)), ScriptedResponse.Text("never"));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await Client.StartWorkflowAsync((CancellationWorkflow wf) => wf.RunAsync("hi"), Fixture.WorkflowOptions(worker, "cancel-model"));
            await WaitForActivityStartAsync(handle, ModelActivityNames.InvokeModel);
            await handle.CancelAsync();
            var error = await Assert.ThrowsAsync<WorkflowFailedException>(() => handle.GetResultAsync());
            Assert.IsType<CanceledFailureException>(error.InnerException);

            var history = await handle.FetchHistoryAsync();
            Assert.Equal(1, HistoryAssert.Count(history, EventType.ActivityTaskCancelRequested));
            Assert.Equal(1, HistoryAssert.Count(history, EventType.ActivityTaskCanceled));
            Assert.Equal(1, HistoryAssert.Count(history, EventType.WorkflowExecutionCanceled));
            Assert.True(EventId(history, EventType.ActivityTaskCanceled) < EventId(history, EventType.WorkflowExecutionCanceled));
            Assert.Equal(1, fake.CallCount);
        });
    }

    [Fact]
    public async Task CancelDuringActivityTool_CancelsTheActivity_NoFurtherModelCall()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })),
            ScriptedResponse.Text("never"));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await Client.StartWorkflowAsync((CancellationWorkflow wf) => wf.RunAsync("Weather?"), Fixture.WorkflowOptions(worker, "cancel-tool"));
            await WaitForActivityStartAsync(handle, "SlowWeather");
            await handle.CancelAsync();
            var error = await Assert.ThrowsAsync<WorkflowFailedException>(() => handle.GetResultAsync());
            Assert.IsType<CanceledFailureException>(error.InnerException);

            var history = await handle.FetchHistoryAsync();
            var scheduled = Assert.Single(HistoryAssert.Scheduled(history, "SlowWeather"));
            var cancelRequested = Assert.Single(history.Events, e => e.ActivityTaskCancelRequestedEventAttributes is not null);
            Assert.Equal(scheduled.EventId, cancelRequested.ActivityTaskCancelRequestedEventAttributes!.ScheduledEventId);
            Assert.Equal(1, HistoryAssert.Count(history, EventType.ActivityTaskCanceled));
            Assert.Equal(1, HistoryAssert.Count(history, EventType.WorkflowExecutionCanceled));
            Assert.True(EventId(history, EventType.ActivityTaskCanceled) < EventId(history, EventType.WorkflowExecutionCanceled));
            Assert.Equal(1, HistoryAssert.ActivityCounts(history)[ModelActivityNames.InvokeModel]);
            Assert.Equal(1, fake.CallCount);
        });
    }

    private static long EventId(Temporalio.Common.WorkflowHistory history, EventType type) => history.Events.Single(e => e.EventType == type).EventId;

    // History records ActivityTaskStarted only when the Activity ends; the service's pending activities show it running.
    private async Task WaitForActivityStartAsync(WorkflowHandle handle, string activityType)
    {
        var request = new DescribeWorkflowExecutionRequest
        {
            Namespace = Client.Options.Namespace,
            Execution = new Api.Common.V1.WorkflowExecution { WorkflowId = handle.Id },
        };
        for (var i = 0; i < 400; i++)
        {
            var description = await Client.Connection.WorkflowService.DescribeWorkflowExecutionAsync(request);
            if (description.PendingActivities.Any(a => a.ActivityType.Name == activityType && a.State == PendingActivityState.Started))
            {
                return;
            }

            if (description.WorkflowExecutionInfo.Status != WorkflowExecutionStatus.Running)
            {
                throw new InvalidOperationException($"The Workflow is {description.WorkflowExecutionInfo.Status} before Activity {activityType} started.");
            }

            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        throw new TimeoutException($"Activity {activityType} did not start.");
    }
}
