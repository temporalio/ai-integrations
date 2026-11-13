using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using Temporalio.MicrosoftAgentFramework.Testing;

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>Builds plugins around fakes.</summary>
public static class TestPlugin
{
    public const string Model = "fake";

    public static MicrosoftAgentFrameworkPlugin For(FakeChatClient fake, string model = Model) =>
        For(new Dictionary<string, IChatClient> { [model] = fake });

    public static MicrosoftAgentFrameworkPlugin For(
        IReadOnlyDictionary<string, IChatClient> clients,
        IReadOnlyDictionary<string, IClientTransport>? servers = null,
        TimeSpan? idleTimeout = null,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? hostedMcpServerHeaders = null,
        IReadOnlyDictionary<string, Func<IReadOnlyDictionary<string, string>>>? hostedMcpServerHeaderProviders = null) =>
        new(new MicrosoftAgentFrameworkPluginOptions
        {
            ChatClients = clients.ToDictionary(pair => pair.Key, pair => (Func<IChatClient>)(() => pair.Value)),
            McpServers = (servers ?? new Dictionary<string, IClientTransport>()).ToDictionary(pair => pair.Key, pair => (Func<IClientTransport>)(() => pair.Value)),
            McpConnectionIdleTimeout = idleTimeout ?? TimeSpan.FromMinutes(5),
            HostedMcpServerHeaders = hostedMcpServerHeaders ?? new Dictionary<string, IReadOnlyDictionary<string, string>>(),
            HostedMcpServerHeaderProviders = hostedMcpServerHeaderProviders ?? new Dictionary<string, Func<IReadOnlyDictionary<string, string>>>(),
        });

    public static MicrosoftAgentFrameworkPlugin WithMcp(FakeChatClient fake, string server, InMemoryMcpServer mcp, TimeSpan? idleTimeout = null) =>
        For(
            new Dictionary<string, IChatClient> { [Model] = fake },
            new Dictionary<string, IClientTransport> { [server] = mcp.ClientTransport },
            idleTimeout);
}
