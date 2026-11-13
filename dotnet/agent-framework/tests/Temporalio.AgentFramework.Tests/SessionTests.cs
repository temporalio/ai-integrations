using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.AgentFramework.Testing;
using Temporalio.AgentFramework.Tests.Workflows;

namespace Temporalio.AgentFramework.Tests;

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
        var input = new SessionInput(["first"], WithContextProvider: true);
        var (replies, _) = await RunAsync(TestPlugin.For(fake), (SessionWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(["echo[1]: first"], replies);
        var call = fake.Calls[0];
        Assert.Equal($"Be brief.\n{PrefixContextProvider.Instruction}", call.Options!.Instructions);
        Assert.Equal(["first", PrefixContextProvider.ContextMessage], call.Messages.Select(m => m.Text));
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
