using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;

namespace Temporalio.Extensions.MicrosoftAgentFramework;

/// <summary>
/// Worker-side registry for <see cref="MicrosoftAgentFrameworkPlugin"/>. Credentials live in these
/// factories; Workflow code passes names only.
/// </summary>
public sealed class MicrosoftAgentFrameworkPluginOptions
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
    /// Gets the HTTP headers the worker adds to a <see cref="HostedMcpServerTool"/> before the model
    /// call, by the tool's <see cref="HostedMcpServerTool.ServerName"/>. The Workflow passes the tool
    /// without headers: a <see cref="HostedMcpServerTool"/> whose <c>Headers</c> are set in the
    /// Workflow fails the run, because the model Activity's input is recorded in Event History and
    /// an Authorization or API key header would be recorded with it.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> HostedMcpServerHeaders { get; init; } =
        new Dictionary<string, IReadOnlyDictionary<string, string>>();

    /// <summary>
    /// Gets header providers for hosted MCP server tools, by the tool's
    /// <see cref="HostedMcpServerTool.ServerName"/>: the worker calls one before every model call
    /// that carries the tool, so a token that expires can be refreshed. A server name goes in this
    /// dictionary or in <see cref="HostedMcpServerHeaders"/>, not both.
    /// </summary>
    public IReadOnlyDictionary<string, Func<IReadOnlyDictionary<string, string>>> HostedMcpServerHeaderProviders { get; init; } =
        new Dictionary<string, Func<IReadOnlyDictionary<string, string>>>();

    /// <summary>
    /// Gets how long an MCP connection may sit unused before the worker closes it. The next call
    /// reconnects. Defaults to 5 minutes. Every connection is closed when the last worker or
    /// replayer running with this plugin instance stops; workers that share one instance share
    /// its connections.
    /// </summary>
    public TimeSpan McpConnectionIdleTimeout { get; init; } = TimeSpan.FromMinutes(5);
}
