using Microsoft.Extensions.AI;
using Temporalio.AgentFramework.Testing;
using Temporalio.AgentFramework.Tests.Workflows;
using Temporalio.Client;
using Temporalio.Exceptions;

namespace Temporalio.AgentFramework.Tests;

public class ApprovalTests : WorkflowTestBase
{
    public ApprovalTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task ApprovalRequiredTool_SurfacesRequest_ResumesOnSignal()
    {
        var fake = EmailFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Email ops", "workflow");
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-signal"));
            var pending = await WaitForPendingAsync(handle);
            Assert.Equal("ficc_call_1_0", pending.RequestId);
            var call = Assert.IsType<FunctionCallContent>(pending.ToolCall);
            Assert.Equal("send_email", call.Name);
            Assert.Equal(1, fake.CallCount);
            Assert.Equal(0, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));

            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true, "approved")));
            Assert.Equal("Done", await handle.GetResultAsync());
            await HistoryRecorder.SaveAsync(handle, "approval-signal");

            Assert.Equal(1, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));
            Assert.Equal(2, fake.CallCount);
            var providerView = fake.Calls[1].Messages.ToList();
            Assert.DoesNotContain(providerView, m => m.Contents.Any(c => c is ToolApprovalRequestContent or ToolApprovalResponseContent));
            var callMessage = providerView.Single(m => m.Contents.OfType<FunctionCallContent>().Any());
            var resultMessage = providerView[providerView.IndexOf(callMessage) + 1];
            Assert.Equal(ChatRole.Assistant, callMessage.Role);
            Assert.Equal(ChatRole.Tool, resultMessage.Role);
            var result = Assert.IsType<FunctionResultContent>(Assert.Single(resultMessage.Contents));
            Assert.Equal("call_1_0", result.CallId);
            Assert.Equal("email sent to ops@example.com", result.Result?.ToString());
        });
    }

    [Fact]
    public async Task RejectedApproval_ToolNotInvoked_ModelToldWhy()
    {
        var fake = EmailFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Email ops", "workflow");
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-rejected"));
            var pending = await WaitForPendingAsync(handle);
            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(false, "not now")));
            Assert.Equal("Done", await handle.GetResultAsync());
            Assert.Equal(0, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));
            var result = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
            Assert.Equal("Tool call invocation was rejected by the approver and must not be retried. not now", result.Result?.ToString());
        });
    }

    [Fact]
    public async Task Approval_ThroughUpdate_WithValidator()
    {
        var fake = EmailFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await Client.StartWorkflowAsync((ApprovalUpdateWorkflow wf) => wf.RunAsync("Email ops"), Fixture.WorkflowOptions(worker, "approval-update"));
            ToolApprovalRequestContent? pending = null;
            for (var i = 0; i < 200 && pending is null; i++)
            {
                await Task.Delay(50, TestContext.Current.CancellationToken);
                pending = await handle.QueryAsync(wf => wf.PendingApproval());
            }

            Assert.NotNull(pending);
            var bogus = new ToolApprovalResponseContent("ficc_nope", true, pending.ToolCall);
            var rejected = await Assert.ThrowsAsync<WorkflowUpdateFailedException>(() => handle.ExecuteUpdateAsync(wf => wf.DecideAsync(bogus)));
            Assert.Contains("ficc_nope", rejected.InnerException?.Message, StringComparison.Ordinal);

            var text = await handle.ExecuteUpdateAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", text);
            Assert.Equal("Done", await handle.GetResultAsync());
        });
    }

    [Fact]
    public async Task ApprovedActivityTool_RunsExactlyOnce()
    {
        var fake = EmailFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Email ops", "activity");
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-activity"));
            var pending = await WaitForPendingAsync(handle);
            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", await handle.GetResultAsync());
            var counts = HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync());
            Assert.Equal(1, counts["SendEmail"]);
            Assert.Equal(2, counts[ModelActivityNames.InvokeModel]);
            var result = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
            Assert.Equal("email sent to ops@example.com", result.Result?.ToString());
        });
    }

    [Fact]
    public async Task MixedResponse_PlainToolRunsNow_ApprovalToolWaits()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(
                ("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" }),
                ("send_email", new Dictionary<string, object?> { ["to"] = "ops@example.com" })),
            ScriptedResponse.Text("Done"));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Weather and email", "workflow", WithPlainTool: true);
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-mixed"));
            var pending = await WaitForPendingAsync(handle);
            Assert.Equal("send_email", ((FunctionCallContent)pending.ToolCall).Name);
            Assert.Equal(1, fake.CallCount);
            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", await handle.GetResultAsync());
            var results = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().ToList();
            Assert.Equal(["call_1_0", "call_1_1"], results.Select(r => r.CallId));
            Assert.Equal("sunny in Kingston", results[0].Result?.ToString());
            Assert.Equal("email sent to ops@example.com", results[1].Result?.ToString());
        });
    }

    [Fact]
    public async Task UnansweredApproval_FailsWorkflowWithWorkflowError()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })),
            ScriptedResponse.Text("Done"));
        var input = new FailureInput("unanswered-approval");
        var (error, _) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Equal(1, fake.CallCount);
    }

    private static FakeChatClient EmailFake() => new(
        ScriptedResponse.ToolCalls(("send_email", new Dictionary<string, object?> { ["to"] = "ops@example.com" })),
        ScriptedResponse.Text("Done"));

    private static async Task<ToolApprovalRequestContent> WaitForPendingAsync(WorkflowHandle<ApprovalWorkflow> handle)
    {
        for (var i = 0; i < 200; i++)
        {
            if (await handle.QueryAsync(wf => wf.PendingApproval()) is { } pending)
            {
                return pending;
            }

            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        throw new TimeoutException("No approval request became pending.");
    }
}
