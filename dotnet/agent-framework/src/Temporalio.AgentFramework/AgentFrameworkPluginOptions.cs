using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;

namespace Temporalio.AgentFramework;

/// <summary>
/// Worker-side registry for <see cref="AgentFrameworkPlugin"/>. Credentials live in these
/// factories; Workflow code passes names only.
/// </summary>
public sealed class AgentFrameworkPluginOptions
{
    /// <summary>
    /// Gets the provider chat clients by model name. Each factory runs at most once per worker,
    /// on the first model call that names it, and the client is reused for the worker's lifetime.
    /// Disable the provider SDK's own retries in the factory; Temporal retries the Activity.
    /// </summary>
    public IReadOnlyDictionary<string, Func<IChatClient>> ChatClients { get; init; } =
        new Dictionary<string, Func<IChatClient>>();

    /// <summary>
    /// Gets the MCP server transports by server name. The plugin connects an
    /// <see cref="McpClient"/> per server on first use, reuses it across Activities, and closes it
    /// after <see cref="McpConnectionIdleTimeout"/> without a call.
    /// </summary>
    public IReadOnlyDictionary<string, Func<IClientTransport>> McpServers { get; init; } =
        new Dictionary<string, Func<IClientTransport>>();

    /// <summary>
    /// Gets how long an MCP connection may sit unused before the worker closes it. The next call
    /// reconnects. Defaults to 5 minutes. Every connection is closed when the worker stops.
    /// </summary>
    public TimeSpan McpConnectionIdleTimeout { get; init; } = TimeSpan.FromMinutes(5);
}
