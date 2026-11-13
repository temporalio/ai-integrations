using Temporalio.Common;
using Temporalio.MicrosoftAgentFramework.Activities;
using Temporalio.MicrosoftAgentFramework.Internal;
using Temporalio.Worker;

namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// The Temporal plugin for Microsoft Agent Framework. Pass one instance to
/// <c>TemporalClientConnectOptions.Plugins</c> or <c>TemporalWorkerOptions.Plugins</c>. It
/// registers the model and MCP Activities, marks <see cref="MicrosoftAgentFrameworkWorkflowException"/>
/// as a Workflow failure, and closes MCP connections when the last worker or replayer running with
/// the instance stops. Workers that share one instance share its MCP connections.
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
    /// Adds the plugin's Activities and failure type. When the same instance reaches the worker
    /// twice, through the client options and the worker options, the second pass appends the same
    /// definitions again and this method removes those copies, so the worker starts. Any other
    /// duplicate Activity name, including a second instance of this plugin, is left for the SDK's
    /// own check, which fails the worker with <c>Duplicate activity named ...</c>.
    /// </summary>
    /// <param name="options">Worker options to configure.</param>
    public override void ConfigureWorker(TemporalWorkerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        base.ConfigureWorker(options);
        foreach (var definition in Options.Activities)
        {
            RemoveLaterCopies(options.Activities, definition);
        }

        if (options.WorkflowFailureExceptionTypes is { } types)
        {
            var list = types.ToList();
            RemoveLaterCopies(list, typeof(MicrosoftAgentFrameworkWorkflowException));
            options.WorkflowFailureExceptionTypes = list;
        }
    }

    /// <summary>Removes every occurrence of <paramref name="item"/> after its first, by reference.</summary>
    private static void RemoveLaterCopies<T>(IList<T> list, T item)
        where T : class
    {
        var first = -1;
        for (var i = 0; i < list.Count; i++)
        {
            if (!ReferenceEquals(list[i], item))
            {
                continue;
            }

            if (first < 0)
            {
                first = i;
                continue;
            }

            list.RemoveAt(i);
            i--;
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
#pragma warning disable CA2000 // The cache lives as long as the plugin; the last ReleaseAsync closes its connections.
        var connections = new McpConnectionCache(options.McpServers, options.McpConnectionIdleTimeout);
#pragma warning restore CA2000
        var pluginOptions = new SimplePluginOptions
        {
            WorkflowFailureExceptionTypes = [typeof(MicrosoftAgentFrameworkWorkflowException)],
            RunContextBefore = () =>
            {
                connections.Acquire();
                return Task.CompletedTask;
            },
            RunContextAfter = () => connections.ReleaseAsync().AsTask(),
        };
        pluginOptions.AddAllActivities(new ModelActivities(options.ChatClients, options.HostedMcpServerHeaders));
        pluginOptions.AddAllActivities(new McpActivities(connections));
        return pluginOptions;
    }
}
