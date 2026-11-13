using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using Temporalio.AgentFramework.Testing;

namespace Temporalio.AgentFramework.Tests;

/// <summary>Builds plugins around fakes.</summary>
public static class TestPlugin
{
    public const string Model = "fake";

    public static AgentFrameworkPlugin For(FakeChatClient fake, string model = Model) =>
        For(new Dictionary<string, IChatClient> { [model] = fake });

    public static AgentFrameworkPlugin For(
        IReadOnlyDictionary<string, IChatClient> clients,
        IReadOnlyDictionary<string, IClientTransport>? servers = null,
        TimeSpan? idleTimeout = null) =>
        new(new AgentFrameworkPluginOptions
        {
            ChatClients = clients.ToDictionary(pair => pair.Key, pair => (Func<IChatClient>)(() => pair.Value)),
            McpServers = (servers ?? new Dictionary<string, IClientTransport>()).ToDictionary(pair => pair.Key, pair => (Func<IClientTransport>)(() => pair.Value)),
            McpConnectionIdleTimeout = idleTimeout ?? TimeSpan.FromMinutes(5),
        });

    public static AgentFrameworkPlugin WithMcp(FakeChatClient fake, string server, InMemoryMcpServer mcp, TimeSpan? idleTimeout = null) =>
        For(
            new Dictionary<string, IChatClient> { [Model] = fake },
            new Dictionary<string, IClientTransport> { [server] = mcp.ClientTransport },
            idleTimeout);
}
