using Temporalio.Common;
using Temporalio.MicrosoftAgentFramework.Activities;
using Temporalio.MicrosoftAgentFramework.Internal;
using Temporalio.Worker;

namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// The Temporal plugin for Microsoft Agent Framework. Pass one instance to
/// <c>TemporalClientConnectOptions.Plugins</c> or <c>TemporalWorkerOptions.Plugins</c>. It
/// registers the model and MCP Activities, marks <see cref="MicrosoftAgentFrameworkWorkflowException"/>
/// as a Workflow failure, and closes MCP connections when the worker stops.
/// </summary>
public sealed class MicrosoftAgentFrameworkPlugin : SimplePlugin
{
    /// <summary>Plugin name reported to Temporal.</summary>
    public const string PluginName = "microsoft.AgentFrameworkPlugin";

    /// <summary>
    /// Initializes a new instance of the <see cref="MicrosoftAgentFrameworkPlugin"/> class.
    /// </summary>
    /// <param name="options">Chat client factories, MCP transports and connection settings.</param>
    public MicrosoftAgentFrameworkPlugin(MicrosoftAgentFrameworkPluginOptions options)
        : base(PluginName, BuildOptions(Validate(options)))
    {
    }

    /// <summary>
    /// Adds the plugin's Activities and failure types, then removes Activities that are already
    /// registered under the same name, so a plugin that reaches the worker through both the client
    /// and the worker options does not fail the worker.
    /// </summary>
    /// <param name="options">Worker options to configure.</param>
    public override void ConfigureWorker(TemporalWorkerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        base.ConfigureWorker(options);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < options.Activities.Count; i++)
        {
            var name = options.Activities[i].Name;
            if (name is not null && !seen.Add(name))
            {
                options.Activities.RemoveAt(i);
                i--;
            }
        }

        if (options.WorkflowFailureExceptionTypes is { } types)
        {
            options.WorkflowFailureExceptionTypes = types.Distinct().ToList();
        }
    }

    private static MicrosoftAgentFrameworkPluginOptions Validate(MicrosoftAgentFrameworkPluginOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.ChatClients, nameof(options.ChatClients));
        ArgumentNullException.ThrowIfNull(options.McpServers, nameof(options.McpServers));
        foreach (var (name, factory) in options.ChatClients)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("ChatClients has an empty model name.", nameof(options));
            }

            if (factory is null)
            {
                throw new ArgumentException($"ChatClients['{name}'] has no factory.", nameof(options));
            }
        }

        foreach (var (name, factory) in options.McpServers)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("McpServers has an empty server name.", nameof(options));
            }

            if (factory is null)
            {
                throw new ArgumentException($"McpServers['{name}'] has no factory.", nameof(options));
            }
        }

        ArgumentNullException.ThrowIfNull(options.HostedMcpServerHeaders, nameof(options.HostedMcpServerHeaders));
        foreach (var (name, headers) in options.HostedMcpServerHeaders)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("HostedMcpServerHeaders has an empty server name.", nameof(options));
            }

            if (headers is null)
            {
                throw new ArgumentException($"HostedMcpServerHeaders['{name}'] has no headers.", nameof(options));
            }
        }

        if (options.McpConnectionIdleTimeout <= TimeSpan.Zero)
        {
            throw new ArgumentException("McpConnectionIdleTimeout must be positive.", nameof(options));
        }

        return options;
    }

    private static SimplePluginOptions BuildOptions(MicrosoftAgentFrameworkPluginOptions options)
    {
#pragma warning disable CA2000 // The cache lives as long as the plugin; RunContextAfter closes its connections.
        var connections = new McpConnectionCache(options.McpServers, options.McpConnectionIdleTimeout);
#pragma warning restore CA2000
        var pluginOptions = new SimplePluginOptions
        {
            WorkflowFailureExceptionTypes = [typeof(MicrosoftAgentFrameworkWorkflowException)],
            RunContextAfter = () => connections.DisposeAsync().AsTask(),
        };
        pluginOptions.AddAllActivities(new ModelActivities(options.ChatClients, options.HostedMcpServerHeaders));
        pluginOptions.AddAllActivities(new McpActivities(connections));
        return pluginOptions;
    }
}
