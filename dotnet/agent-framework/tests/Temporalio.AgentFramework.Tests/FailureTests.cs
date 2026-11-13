using System.Net;
using Microsoft.Extensions.AI;
using Temporalio.AgentFramework.Testing;
using Temporalio.AgentFramework.Tests.Workflows;
using Temporalio.Api.Enums.V1;

namespace Temporalio.AgentFramework.Tests;

public class FailureTests : WorkflowTestBase
{
    public FailureTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task Model4xx_NonRetryable_OneAttempt_ModelError()
    {
        var fake = new FakeChatClient(ScriptedResponse.Fail(HttpStatusCode.BadRequest, "bad request"));
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput("model")));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.ModelError, failure.ErrorType);
        Assert.True(failure.NonRetryable);
        Assert.Equal(1, fake.CallCount);
        Assert.Equal(1, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.InvokeModel));
    }

    [Fact]
    public async Task Model5xx_Retried_ThenWorkflowFails()
    {
        var fake = new FakeChatClient(ScriptedResponse.Fail(HttpStatusCode.InternalServerError));
        var input = new FailureInput("model", MaxAttempts: 3);
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.ModelError, failure.ErrorType);
        Assert.False(failure.NonRetryable);
        Assert.Equal(3, fake.CallCount);
        Assert.Equal(3, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.InvokeModel));
    }

    [Fact]
    public async Task Model503ThenSuccess_RecoversOnRetry()
    {
        var fake = new FakeChatClient(ScriptedResponse.Fail(HttpStatusCode.ServiceUnavailable), ScriptedResponse.Text("recovered"));
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput("model")));
        Assert.Equal("recovered", result);
        Assert.Equal(2, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.InvokeModel));
    }

    [Fact]
    public async Task Model429WithRetryAfter_SetsNextRetryDelay()
    {
        var fake = new FakeChatClient(_ => throw new System.ClientModel.ClientResultException("rate limited", new FakePipelineResponse(429, "2")));
        var input = new FailureInput("model", MaxAttempts: 1);
        var (error, _) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.ModelError, failure.ErrorType);
        Assert.False(failure.NonRetryable);
        Assert.Equal(TimeSpan.FromSeconds(2), failure.NextRetryDelay);
    }

    [Fact]
    public async Task UnknownModel_ConfigurationError()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("never"));
        var input = new FailureInput("model", Model: "missing");
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.ConfigurationError, failure.ErrorType);
        Assert.True(failure.NonRetryable);
        Assert.Equal(1, HistoryAssert.LastAttempt(await handle.FetchHistoryAsync(), ModelActivityNames.InvokeModel));
        Assert.Equal(0, fake.CallCount);
    }

    [Fact]
    public async Task UnknownTool_ModelToldNotFound()
    {
        var fake = new FakeChatClient(
            ScriptedResponse.ToolCalls(("launch_rockets", new Dictionary<string, object?> { ["count"] = 1 })),
            ScriptedResponse.Text("ok"));
        var (result, _) = await RunAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput("unknown-tool")));
        Assert.Equal("ok", result);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("Error: Requested function \"launch_rockets\" not found.", toolResult.Result?.ToString());
    }

    [Fact]
    public async Task IterationCap_LastCallWithoutFunctionTools()
    {
        var call = ("get_weather", (IDictionary<string, object?>)new Dictionary<string, object?> { ["city"] = "Kingston" });
        var fake = new FakeChatClient(ScriptedResponse.ToolCalls(call), ScriptedResponse.ToolCalls(call), ScriptedResponse.Text("gave up"));
        var input = new FailureInput("iteration-cap", MaximumIterations: 2);
        var (result, handle) = await RunAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(input));
        Assert.Equal("gave up", result);
        Assert.Equal(3, fake.CallCount);
        Assert.NotNull(fake.Calls[1].Options!.Tools);
        Assert.Null(fake.Calls[2].Options!.Tools);
        Assert.Equal(3, HistoryAssert.ActivityCounts(await handle.FetchHistoryAsync())[ModelActivityNames.InvokeModel]);
    }

    [Fact]
    public async Task ConsecutiveToolErrors_FailWorkflowWithWorkflowError()
    {
        var fake = new FakeChatClient(ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })));
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput("tool-errors")));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Equal(4, fake.CallCount);
        var toolResult = fake.Calls[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("Error: Function failed.", toolResult.Result?.ToString());
        Assert.Equal(0, HistoryAssert.Count(await handle.FetchHistoryAsync(), EventType.WorkflowTaskFailed));
    }

    [Fact]
    public async Task RunAsync_ThroughAIAgentBase_FailsFastWithWorkflowError()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("never"));
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput("base-entry")));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Contains("TemporalAgent-typed", failure.Message, StringComparison.Ordinal);
        Assert.Equal(0, fake.CallCount);
        var history = await handle.FetchHistoryAsync();
        Assert.Equal(0, HistoryAssert.Count(history, EventType.WorkflowTaskFailed));
        Assert.Equal(1, HistoryAssert.Count(history, EventType.WorkflowExecutionFailed));
    }

    [Theory]
    [InlineData("unsupported-run-options", "ChatClientFactory")]
    [InlineData("wrong-session", "OtherSession")]
    public async Task Misuse_FailsWorkflowWithWorkflowError(string kind, string expectedMessagePart)
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("never"));
        var (error, _) = await RunFailingAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput(kind)));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(AgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.Contains(expectedMessagePart, failure.Message, StringComparison.Ordinal);
        Assert.Equal(0, fake.CallCount);
    }

    [Fact]
    public async Task DeclarationOnlyTool_ReturnsCallsToCaller()
    {
        var fake = new FakeChatClient(ScriptedResponse.ToolCalls(("get_weather", new Dictionary<string, object?> { ["city"] = "Kingston" })));
        var (result, _) = await RunAsync(TestPlugin.For(fake), (FailureWorkflow wf) => wf.RunAsync(new FailureInput("declaration-only")));
        Assert.Equal("calls returned to caller: 1", result);
        Assert.Equal(1, fake.CallCount);
    }
}
