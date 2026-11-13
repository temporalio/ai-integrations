using Temporalio.Activities;
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
    public void ConfigureWorker_RegistersActivitiesOnce()
    {
        var plugin = TestPlugin.For(new FakeChatClient(ScriptedResponse.Text("unused")));
        var options = new TemporalWorkerOptions("tq");
        plugin.ConfigureWorker(options);
        plugin.ConfigureWorker(options);
        Assert.Equal(
            [ModelActivityNames.InvokeModel, ModelActivityNames.McpListTools, ModelActivityNames.McpCallTool],
            options.Activities.Select(a => a.Name).OrderBy(n => n == ModelActivityNames.InvokeModel ? 0 : n == ModelActivityNames.McpListTools ? 1 : 2));
        Assert.Equal(3, options.Activities.Count);
        Assert.Null(options.WorkflowFailureExceptionTypes);
        Assert.Equal("microsoft.AgentFrameworkPlugin", plugin.Name);
    }

    [Fact]
    public void ConfigureWorker_LeavesApplicationDuplicatesToTheSdk()
    {
        // A fresh queue: the SDK registers the native worker before its duplicate check throws.
        var options = new TemporalWorkerOptions($"tq-{Guid.NewGuid():N}")
            .AddActivity(Colliding.First)
            .AddActivity(Colliding.Second);
        options.Plugins = [TestPlugin.For(new FakeChatClient(ScriptedResponse.Text("unused")))];
        var error = Assert.Throws<ArgumentException>(() => new TemporalWorker(Client, options));
        Assert.Contains("Duplicate activity named Lookup", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoPluginInstances_FailTheWorkerWithTheSdkError()
    {
        var fake = new FakeChatClient(ScriptedResponse.Text("unused"));
        var options = new TemporalWorkerOptions($"tq-{Guid.NewGuid():N}").AddWorkflow<HelloWorkflow>();
        options.Plugins = [TestPlugin.For(fake), TestPlugin.For(fake)];
        var error = Assert.Throws<ArgumentException>(() => new TemporalWorker(Client, options));
        Assert.Contains($"Duplicate activity named {ModelActivityNames.InvokeModel}", error.Message, StringComparison.Ordinal);
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
    [InlineData(true)]
    [InlineData(false)]
    public void Options_WithBadHostedMcpServerHeaders_AreRejected(bool emptyName)
    {
        var options = new MicrosoftAgentFrameworkPluginOptions
        {
            ChatClients = new Dictionary<string, Func<Microsoft.Extensions.AI.IChatClient>> { ["gpt"] = () => new FakeChatClient(ScriptedResponse.Text("x")) },
            HostedMcpServerHeaders = new Dictionary<string, IReadOnlyDictionary<string, string>>
            {
                [emptyName ? " " : "docs"] = emptyName ? new Dictionary<string, string>() : null!,
            },
        };
        var error = Assert.Throws<ArgumentException>(() => new MicrosoftAgentFrameworkPlugin(options));
        Assert.Contains(emptyName ? "HostedMcpServerHeaders has an empty server name" : "HostedMcpServerHeaders['docs'] has no headers", error.Message, StringComparison.Ordinal);
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
    public void Replayer_ConfiguredByPlugin_HasItsActivities()
    {
        var plugin = TestPlugin.For(new FakeChatClient(ScriptedResponse.Text("unused")));
        var replayer = new WorkflowReplayer(new WorkflowReplayerOptions { Plugins = [plugin] }.AddWorkflow<HelloWorkflow>());
        Assert.Null(replayer.Options.WorkflowFailureExceptionTypes);
    }

    private TemporalClient ClientWithPlugin(MicrosoftAgentFrameworkPlugin plugin) => new(
        Client.Connection,
        new TemporalClientOptions { Namespace = Client.Options.Namespace, Plugins = [plugin] });

    // Two application Activities that share a name: an error the SDK must still report.
    private static class Colliding
    {
        [Activity("Lookup")]
        public static string First(string id) => id;

        [Activity("Lookup")]
        public static string Second(string id) => id.ToUpperInvariant();
    }
}
