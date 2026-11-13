using Temporalio.Client;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Worker;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class PluginTests : WorkflowTestBase
{
    public PluginTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task PluginOnClient_WorkerGetsActivities()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("hello from the client plugin"));
        var client = ClientWithPlugin(TestPlugin.For(fake, "gpt-4o-mini"));
        var (result, _) = await RunAsync(null, (HelloWorkflow wf) => wf.RunAsync("hi"), client: client);
        Assert.Equal("hello from the client plugin", result);
    }

    [Fact]
    public async Task PluginOnBoth_NoDuplicateActivityError()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("hello twice"));
        var plugin = TestPlugin.For(fake, "gpt-4o-mini");
        var client = ClientWithPlugin(plugin);
        var (result, _) = await RunAsync(plugin, (HelloWorkflow wf) => wf.RunAsync("hi"), client: client);
        Assert.Equal("hello twice", result);
    }

    [Fact]
    public async Task Readme_Quickstart_Runs()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("Temporal keeps the agent going."));
        var (result, handle) = await RunAsync(TestPlugin.For(fake, "gpt-4o-mini"), (HelloWorkflow wf) => wf.RunAsync("What does Temporal do for agents?"));
        Assert.Equal("Temporal keeps the agent going.", result);
        Assert.Equal("Answer in one sentence.", fake.Calls[0].Options!.Instructions);
        Assert.Equal([ModelActivityNames.InvokeModel], HistoryAssert.ActivityTypesInOrder(await handle.FetchHistoryAsync()));
    }

    [Fact]
    public void ConfigureWorker_RegistersActivitiesOnceAndFailureTypes()
    {
        var plugin = TestPlugin.For(new FakeChatClient(ScriptedResponse.Text("unused")));
        var options = new TemporalWorkerOptions("tq");
        plugin.ConfigureWorker(options);
        plugin.ConfigureWorker(options);
        Assert.Equal(
            [ModelActivityNames.InvokeModel, ModelActivityNames.McpListTools, ModelActivityNames.McpCallTool],
            options.Activities.Select(a => a.Name).OrderBy(n => n == ModelActivityNames.InvokeModel ? 0 : n == ModelActivityNames.McpListTools ? 1 : 2));
        Assert.Equal(3, options.Activities.Count);
        Assert.Equal([typeof(MicrosoftAgentFrameworkWorkflowException)], options.WorkflowFailureExceptionTypes);
        Assert.Equal("microsoft.AgentFrameworkPlugin", plugin.Name);
    }

    [Theory]
    [InlineData("", true)]
    [InlineData("  ", true)]
    [InlineData("gpt", false)]
    public void Options_WithEmptyNameOrNullFactory_AreRejected(string name, bool withFactory)
    {
        var options = new MicrosoftAgentFrameworkPluginOptions
        {
            ChatClients = new Dictionary<string, Func<Microsoft.Extensions.AI.IChatClient>> { [name] = withFactory ? () => new FakeChatClient(ScriptedResponse.Text("x")) : null! },
        };
        var error = Assert.Throws<ArgumentException>(() => new MicrosoftAgentFrameworkPlugin(options));
        Assert.Contains(withFactory ? "empty model name" : "no factory", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Options_WithEmptyServerName_AreRejected()
    {
        var options = new MicrosoftAgentFrameworkPluginOptions
        {
            McpServers = new Dictionary<string, Func<ModelContextProtocol.Client.IClientTransport>> { [string.Empty] = () => null! },
        };
        var error = Assert.Throws<ArgumentException>(() => new MicrosoftAgentFrameworkPlugin(options));
        Assert.Contains("empty server name", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void Options_WithNonPositiveIdleTimeout_AreRejected(int seconds)
    {
        var options = new MicrosoftAgentFrameworkPluginOptions
        {
            ChatClients = new Dictionary<string, Func<Microsoft.Extensions.AI.IChatClient>> { ["gpt"] = () => new FakeChatClient(ScriptedResponse.Text("x")) },
            McpConnectionIdleTimeout = TimeSpan.FromSeconds(seconds),
        };
        var error = Assert.Throws<ArgumentException>(() => new MicrosoftAgentFrameworkPlugin(options));
        Assert.Contains("McpConnectionIdleTimeout must be positive", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Replayer_ConfiguredByPlugin_HasFailureTypes()
    {
        var plugin = TestPlugin.For(new FakeChatClient(ScriptedResponse.Text("unused")));
        var replayer = new WorkflowReplayer(new WorkflowReplayerOptions { Plugins = [plugin] }.AddWorkflow<HelloWorkflow>());
        Assert.Contains(typeof(MicrosoftAgentFrameworkWorkflowException), replayer.Options.WorkflowFailureExceptionTypes ?? []);
    }

    private TemporalClient ClientWithPlugin(MicrosoftAgentFrameworkPlugin plugin) => new(
        Client.Connection,
        new TemporalClientOptions { Namespace = Client.Options.Namespace, Plugins = [plugin] });
}
