using System.Net;
using Microsoft.Extensions.AI;
using Temporalio.Api.Enums.V1;
using Temporalio.Client;
using Temporalio.Exceptions;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests;

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
            Assert.Equal("Tool call invocation rejected. not now", result.Result?.ToString());
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
            IReadOnlyList<ToolApprovalRequestContent> pending = [];
            for (var i = 0; i < 200 && pending.Count == 0; i++)
            {
                await Task.Delay(50, TestContext.Current.CancellationToken);
                pending = await handle.QueryAsync(wf => wf.PendingApprovals());
            }

            var request = Assert.Single(pending);
            Assert.Equal("send_email", ((FunctionCallContent)request.ToolCall).Name);

            // The validator rejects an empty id; the handler rejects an id that is not pending.
            var empty = new ApprovalDecision(string.Empty, true);
            var invalid = await Assert.ThrowsAsync<WorkflowUpdateFailedException>(() => handle.ExecuteUpdateAsync(wf => wf.DecideAsync(empty)));
            Assert.Contains("RequestId is required", invalid.InnerException?.Message, StringComparison.Ordinal);
            var bogus = new ApprovalDecision("ficc_nope", true);
            var rejected = await Assert.ThrowsAsync<WorkflowUpdateFailedException>(() => handle.ExecuteUpdateAsync(wf => wf.DecideAsync(bogus)));
            Assert.Contains("No pending approval request 'ficc_nope'", rejected.InnerException?.Message, StringComparison.Ordinal);

            var text = await handle.ExecuteUpdateAsync(wf => wf.DecideAsync(new ApprovalDecision(request.RequestId, true)));
            Assert.Equal("approved", text);
            Assert.Equal("Done", await handle.GetResultAsync());
            Assert.Equal(2, fake.CallCount);
            var result = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
            Assert.Equal("email sent to ops@example.com", result.Result?.ToString());
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

    [Theory]
    [InlineData("arguments")]
    [InlineData("name")]
    public async Task TamperedResponse_RunsTheCallTheSessionSurfaced(string tampered)
    {
        var fake = EmailFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Email ops", "workflow", WithPlainTool: true);
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-tampered"));
            var pending = await WaitForPendingAsync(handle);
            var surfaced = Assert.IsType<FunctionCallContent>(pending.ToolCall);

            // Same request id and call id, different arguments or a different tool: only the id binds.
            var swapped = tampered == "arguments"
                ? new FunctionCallContent(surfaced.CallId, surfaced.Name, new Dictionary<string, object?> { ["to"] = "attacker@example.com" })
                : new FunctionCallContent(surfaced.CallId, "get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" });
            await handle.SignalAsync(wf => wf.DecideAsync(new ToolApprovalResponseContent(pending.RequestId, true, swapped)));
            Assert.Equal("Done", await handle.GetResultAsync());

            Assert.Equal(1, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));
            Assert.Equal(0, await handle.QueryAsync(wf => wf.PlainToolInvocations()));
            var providerView = fake.Calls[1].Messages.ToList();
            var call = providerView.SelectMany(m => m.Contents).OfType<FunctionCallContent>().Single();
            Assert.Equal("send_email", call.Name);
            Assert.Equal("ops@example.com", call.Arguments!["to"]?.ToString());
            var result = providerView.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
            Assert.Equal("email sent to ops@example.com", result.Result?.ToString());
        });
    }

    [Fact]
    public async Task ForgedRequestId_FailsRunWithWorkflowError()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })),
            ScriptedResponse.Text("Done"));
        var input = new FailureInput("forged-approval");
        var (error, _) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Contains("'ficc_forged' matches no ToolApprovalRequestContent this session surfaced", failure.Message, StringComparison.Ordinal);
        Assert.Contains("'ficc_call_1_0' has no matching ToolApprovalResponseContent", failure.Message, StringComparison.Ordinal);
        Assert.Equal(1, fake.CallCount);
    }

    [Fact]
    public async Task RunFailingAfterTheApprovedTool_RetryWithTheSameDecision_SeesACleanSession()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("send_email", new Dictionary<string, object?> { ["to"] = "ops@example.com" })),
            ScriptedResponse.Fail(HttpStatusCode.BadRequest, "payload too large"),
            ScriptedResponse.Text("Done"));
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Email ops", "workflow", RetryAfterModelFailure: true);
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-retry"));
            var pending = await WaitForPendingAsync(handle);
            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", await handle.GetResultAsync());

            Assert.Equal(1, await handle.QueryAsync(wf => wf.ModelFailures()));
            Assert.Equal(3, fake.CallCount);

            // The failed run stored nothing, so the retry processes the decision again, including the approved call.
            Assert.Equal(2, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));
            var providerView = fake.Calls[2].Messages.ToList();
            Assert.DoesNotContain(providerView, m => m.Contents.Any(c => c is ToolApprovalRequestContent or ToolApprovalResponseContent));
            var call = providerView.SelectMany(m => m.Contents).OfType<FunctionCallContent>().Single();
            Assert.Equal("call_1_0", call.CallId);
            var result = providerView.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
            Assert.Equal("email sent to ops@example.com", result.Result?.ToString());
            Assert.Equal(0, HistoryAssert.Count(await handle.FetchHistoryAsync(), EventType.WorkflowTaskFailed));
        });
    }

    [Fact]
    public async Task MixedResponse_NothingRunsUntilTheDecision()
    {
        var fake = MixedFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake), options => options.MaxCachedWorkflows = 0);
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Weather and email", "workflow", WithPlainTool: true);
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-mixed"));
            var pending = await WaitForPendingAsync(handle);
            Assert.Equal("send_email", ((FunctionCallContent)pending.ToolCall).Name);
            Assert.Equal(1, fake.CallCount);
            Assert.Equal(0, await handle.QueryAsync(wf => wf.PlainToolInvocations()));
            Assert.Equal(0, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));

            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", await handle.GetResultAsync());
            await HistoryRecorder.SaveAsync(handle, "approval-mixed");
            Assert.Equal(1, await handle.QueryAsync(wf => wf.PlainToolInvocations()));
            Assert.Equal(1, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));
            AssertOneToolRound(fake);
            Assert.Equal(0, HistoryAssert.Count(await handle.FetchHistoryAsync(), EventType.WorkflowTaskFailed));
        });
    }

    [Fact]
    public async Task MixedResponse_AcrossContinueAsNew_HeldCallSurvivesSerialization()
    {
        var fake = MixedFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Weather and email", "workflow", WithPlainTool: true, ContinueAsNew: true);
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "approval-mixed-can"));
            var pending = await WaitForPendingAsync(handle);
            Assert.Equal("send_email", ((FunctionCallContent)pending.ToolCall).Name);
            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", await handle.GetResultAsync());
            Assert.Equal(1, await handle.QueryAsync(wf => wf.PlainToolInvocations()));
            Assert.Equal(1, await handle.QueryAsync(wf => wf.WorkflowToolInvocations()));
            AssertOneToolRound(fake);

            var first = Client.GetWorkflowHandle(handle.Id, runId: handle.ResultRunId);
            var last = Client.GetWorkflowHandle(handle.Id);
            Assert.Equal(1, HistoryAssert.Count(await first.FetchHistoryAsync(), EventType.WorkflowExecutionContinuedAsNew));
            var lastHistory = await last.FetchHistoryAsync();
            Assert.Equal(1, HistoryAssert.Count(lastHistory, EventType.WorkflowExecutionSignaled));
            Assert.Equal(0, HistoryAssert.Count(lastHistory, EventType.WorkflowTaskFailed));
            await HistoryRecorder.SaveAsync(first, "approval-mixed-first");
            await HistoryRecorder.SaveAsync(last, "approval-mixed-last");
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
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Equal(1, fake.CallCount);
    }

    private static FakeChatClient EmailFake() => new(
        ScriptedResponse.ToolCalls(("send_email", new Dictionary<string, object?> { ["to"] = "ops@example.com" })),
        ScriptedResponse.Text("Done"));

    private static FakeChatClient MixedFake() => new(
        ScriptedResponse.ToolCalls(
            ("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" }),
            ("send_email", new Dictionary<string, object?> { ["to"] = "ops@example.com" })),
        ScriptedResponse.Text("Done"));

    // The provider sees one tool round with both calls, as it does under ChatClientAgent.
    private static void AssertOneToolRound(FakeChatClient fake)
    {
        Assert.Equal(2, fake.CallCount);
        var providerView = fake.Calls[1].Messages.ToList();
        var callMessage = providerView.Single(m => m.Contents.OfType<FunctionCallContent>().Any());
        Assert.Equal(["send_email", "get_weather"], callMessage.Contents.OfType<FunctionCallContent>().Select(c => c.Name));
        var resultMessage = providerView[providerView.IndexOf(callMessage) + 1];
        var results = resultMessage.Contents.OfType<FunctionResultContent>().ToList();
        Assert.Equal(["call_1_1", "call_1_0"], results.Select(r => r.CallId));
        Assert.Equal("email sent to ops@example.com", results[0].Result?.ToString());
        Assert.Equal("sunny in Kingston", results[1].Result?.ToString());
        Assert.DoesNotContain(providerView, m => m.Contents.Any(c => c is ToolApprovalRequestContent or ToolApprovalResponseContent));
    }

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
