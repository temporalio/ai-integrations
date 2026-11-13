using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.Api.Enums.V1;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class SessionTests : WorkflowTestBase
{
    public SessionTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task MultiTurn_SameSession_ModelSeesFullHistory()
    {
        var fake = EchoFake();
        var input = new SessionInput(["first", "second", "third"]);
        var (replies, _) = await RunAsync(TestPlugin.For(fake), (SessionWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(["echo[1]: first", "echo[2]: second", "echo[3]: third"], replies);
        Assert.Equal(5, fake.Calls[2].Messages.Count);
        Assert.Equal("Be brief.", fake.Calls[2].Options!.Instructions);
    }

    [Fact]
    public async Task ContinueAsNew_CarriesSerializedSession()
    {
        var fake = EchoFake();
        var input = new SessionInput(["first", "second", "third"], ContinueAsNew: true);
        var (replies, handle) = await RunAsync(TestPlugin.For(fake), (SessionWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(["echo[1]: first", "echo[2]: second", "echo[3]: third"], replies);
        Assert.Equal(5, fake.Calls[2].Messages.Count);
        var runs = 0;
        await foreach (var execution in Client.ListWorkflowsAsync($"WorkflowId = '{handle.Id}'").WithCancellation(TestContext.Current.CancellationToken))
        {
            runs += execution.Id == handle.Id ? 1 : 0;
        }

        Assert.Equal(3, runs);
        await HistoryRecorder.SaveAsync(Client.GetWorkflowHandle(handle.Id, runId: handle.ResultRunId), "session-continue-as-new-first");
        await HistoryRecorder.SaveAsync(Client.GetWorkflowHandle(handle.Id), "session-continue-as-new-last");
    }

    [Fact]
    public async Task CustomHistoryProvider_LogsWarning_StillRuns()
    {
        var fake = EchoFake();
        var logs = new CapturingLoggerProvider();
        var input = new SessionInput(["first", "second"], HistoryProvider: "custom");
        var (replies, _) = await RunAsync(TestPlugin.For(fake), (SessionWorkflow wf) => wf.RunAsync(input), options => options.LoggerFactory = logs);
        Assert.Equal(["echo[1]: first", "echo[2]: second"], replies);
        Assert.Equal(3, fake.Calls[1].Messages.Count);
        Assert.Contains(logs.Entries, entry => entry.Level == LogLevel.Warning && entry.Message.Contains(nameof(RecordingChatHistoryProvider), StringComparison.Ordinal));
    }

    [Fact]
    public async Task ContextProvider_AddsInstructionsAndMessages()
    {
        var fake = EchoFake();
        var input = new SessionInput(["first"], ContextProvider: "prefix");
        var (replies, _) = await RunAsync(TestPlugin.For(fake), (SessionWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(["echo[1]: first"], replies);
        var call = fake.Calls[0];
        Assert.Equal($"Be brief.\n{PrefixContextProvider.Instruction}", call.Options!.Instructions);
        Assert.Equal(["first", PrefixContextProvider.ContextMessage], call.Messages.Select(m => m.Text));
    }

    [Fact]
    public async Task ContextProviderMessages_PersistedLikeChatClientAgent()
    {
        var fake = EchoFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new SessionInput(["first", "second"], ContextProvider: "prefix");
            var handle = await Client.StartWorkflowAsync((SessionWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "session-context"));
            Assert.Equal(["echo[1]: first", "echo[2]: second"], await handle.GetResultAsync());
            Assert.Equal(
                ["first", PrefixContextProvider.ContextMessage, "echo[1]: first", "second", PrefixContextProvider.ContextMessage],
                fake.Calls[1].Messages.Select(m => m.Text));
            Assert.Equal(
                [
                    "user:first",
                    $"user:{PrefixContextProvider.ContextMessage}",
                    "assistant:echo[1]: first",
                    "user:second",
                    $"user:{PrefixContextProvider.ContextMessage}",
                    "assistant:echo[2]: second",
                ],
                await handle.QueryAsync(wf => wf.Transcript()));
        });
    }

    [Theory]
    [InlineData("delaying", null, ProviderWaits.TaskDelay, nameof(DelayingChatHistoryProvider), "did not complete synchronously")]
    [InlineData("memory", "delaying", ProviderWaits.TaskDelay, nameof(DelayingContextProvider), "did not complete synchronously")]
    [InlineData("delaying", null, ProviderWaits.WorkflowTimer, nameof(DelayingChatHistoryProvider), "scheduled Workflow work")]
    [InlineData("memory", "delaying", ProviderWaits.WorkflowTimer, nameof(DelayingContextProvider), "scheduled Workflow work")]
    [InlineData("delaying", null, ProviderWaits.WorkflowActivity, nameof(DelayingChatHistoryProvider), "scheduled Workflow work")]
    [InlineData("memory", "delaying", ProviderWaits.WorkflowActivity, nameof(DelayingContextProvider), "scheduled Workflow work")]
    public async Task ProviderThatWaits_FailsRunWithWorkflowError(string historyProvider, string? contextProvider, string wait, string providerName, string expectedPhrase)
    {
        var fake = EchoFake();
        var input = new SessionInput(["first"], HistoryProvider: historyProvider, ContextProvider: contextProvider, ProviderWait: wait);
        var (error, handle) = await RunFailingAsync(TestPlugin.For(fake), (SessionWorkflow wf) => wf.RunAsync(input));
        var failure = AppFailure(error);
        Assert.NotNull(failure);
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.WorkflowError, failure.ErrorType);
        Assert.StartsWith($"{providerName}.InvokingAsync", failure.Message, StringComparison.Ordinal);
        Assert.Contains(expectedPhrase, failure.Message, StringComparison.Ordinal);
        Assert.Equal(0, fake.CallCount);
        var history = await handle.FetchHistoryAsync();
        Assert.Equal(0, HistoryAssert.Count(history, EventType.WorkflowTaskFailed));
        Assert.Equal(1, HistoryAssert.Count(history, EventType.WorkflowExecutionFailed));
    }

    [Fact]
    public async Task Query_Transcript_ReturnsMessages()
    {
        var fake = EchoFake();
        using var worker = Fixture.NewWorker(TestPlugin.For(fake));
        await WithWorkerAsync(worker, async () =>
        {
            var input = new SessionInput(["first", "second"]);
            var handle = await Client.StartWorkflowAsync((SessionWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "session-query"));
            await handle.GetResultAsync();
            var transcript = await handle.QueryAsync(wf => wf.Transcript());
            Assert.Equal(["user:first", "assistant:echo[1]: first", "user:second", "assistant:echo[2]: second"], transcript);
        });
    }

    private static FakeChatClient EchoFake() => new(call =>
    {
        var users = call.Messages.Where(m => m.Role == ChatRole.User && !m.Text.StartsWith("Context:", StringComparison.Ordinal)).ToList();
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, $"echo[{users.Count}]: {users[^1].Text}"));
    });
}
