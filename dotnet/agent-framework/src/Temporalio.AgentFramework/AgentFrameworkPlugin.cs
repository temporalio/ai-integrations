using Temporalio.AgentFramework.Activities;
using Temporalio.AgentFramework.Internal;
using Temporalio.Common;
using Temporalio.Worker;

namespace Temporalio.AgentFramework;

/// <summary>
/// The Temporal plugin for Microsoft Agent Framework. Pass one instance to
/// <c>TemporalClientConnectOptions.Plugins</c> or <c>TemporalWorkerOptions.Plugins</c>. It
/// registers the model and MCP Activities, marks <see cref="AgentFrameworkWorkflowException"/>
/// as a Workflow failure, and closes MCP connections when the worker stops.
/// </summary>
public sealed class AgentFrameworkPlugin : SimplePlugin
{
    /// <summary>Plugin name reported to Temporal.</summary>
    public const string PluginName = "microsoft.AgentFrameworkPlugin";

    /// <summary>
    /// Initializes a new instance of the <see cref="AgentFrameworkPlugin"/> class.
    /// </summary>
    /// <param name="options">Chat client factories, MCP transports and connection settings.</param>
    public AgentFrameworkPlugin(AgentFrameworkPluginOptions options)
        : base(PluginName, BuildOptions(options ?? throw new ArgumentNullException(nameof(options))))
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

    private static SimplePluginOptions BuildOptions(AgentFrameworkPluginOptions options)
    {
#pragma warning disable CA2000 // The cache lives as long as the plugin; RunContextAfter closes its connections.
        var connections = new McpConnectionCache(options.McpServers, options.McpConnectionIdleTimeout);
#pragma warning restore CA2000
        var pluginOptions = new SimplePluginOptions
        {
            WorkflowFailureExceptionTypes = [typeof(AgentFrameworkWorkflowException)],
            RunContextAfter = () => connections.DisposeAsync().AsTask(),
        };
        pluginOptions.AddAllActivities(new ModelActivities(options.ChatClients));
        pluginOptions.AddAllActivities(new McpActivities(connections));
        return pluginOptions;
    }
}
