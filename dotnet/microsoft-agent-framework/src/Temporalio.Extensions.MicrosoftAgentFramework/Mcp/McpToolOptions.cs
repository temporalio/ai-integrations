using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Mcp;

/// <summary>
/// Options for the tools <see cref="TemporalMcpServer.ListToolsAsync"/> returns.
/// </summary>
public sealed class McpToolOptions
{
    /// <summary>Default <c>StartToCloseTimeout</c> of the list-tools Activity: 30 seconds.</summary>
    public static readonly TimeSpan DefaultListToolsTimeout = TimeSpan.FromSeconds(30);

    /// <summary>Default <c>StartToCloseTimeout</c> of the call-tool Activity: 60 seconds.</summary>
    public static readonly TimeSpan DefaultCallToolTimeout = TimeSpan.FromSeconds(60);

    /// <summary>
    /// Gets or sets the options of the list-tools Activity. Null fields get the defaults: 30-second
    /// start-to-close timeout and the server name as the summary.
    /// </summary>
    public ActivityOptions? ListToolsActivityOptions { get; set; }

    /// <summary>
    /// Gets or sets the options of the call-tool Activity. Null fields get the defaults: 60-second
    /// start-to-close timeout and <c>{server}/{tool}</c> as the summary.
    /// </summary>
    public ActivityOptions? CallToolActivityOptions { get; set; }

    /// <summary>
    /// Gets or sets the server tool names to keep. Null keeps every tool the server lists.
    /// </summary>
    public IReadOnlyCollection<string>? AllowedTools { get; set; }

    /// <summary>
    /// Gets or sets a prefix for the names the model sees, so two servers with the same tool
    /// name do not clash. The name is built the way Agent Framework's MCP tool builds it:
    /// characters outside <c>A-Za-z0-9_.-</c> become <c>-</c> and the prefix is joined with one
    /// <c>_</c>, so <c>gh</c> and <c>gh_</c> both give <c>gh_tool</c>. Calls go to the server
    /// under the original name.
    /// </summary>
    public string? ToolNamePrefix { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the worker may answer the list-tools Activity
    /// from the list it fetched earlier on the same connection. Default false, so a server that
    /// changes its tools is seen on the next list. The Activity runs either way, so Event History
    /// is the same for both values.
    /// </summary>
    public bool CacheTools { get; set; }

    /// <summary>
    /// Returns the list-tools Activity options with defaults applied.
    /// </summary>
    /// <param name="server">Server name, for the summary.</param>
    /// <returns>Options ready to schedule with.</returns>
    internal ActivityOptions ResolveListToolsOptions(string server)
    {
        var options = ListToolsActivityOptions is null ? new ActivityOptions() : (ActivityOptions)ListToolsActivityOptions.Clone();
        if (options.StartToCloseTimeout is null && options.ScheduleToCloseTimeout is null)
        {
            options.StartToCloseTimeout = DefaultListToolsTimeout;
        }

        options.Summary ??= server;
        return options;
    }

    /// <summary>
    /// Returns the call-tool Activity options with defaults applied.
    /// </summary>
    /// <param name="server">Server name, for the summary.</param>
    /// <param name="tool">Tool name, for the summary.</param>
    /// <returns>Options ready to schedule with.</returns>
    internal ActivityOptions ResolveCallToolOptions(string server, string tool)
    {
        var options = CallToolActivityOptions is null ? new ActivityOptions() : (ActivityOptions)CallToolActivityOptions.Clone();
        if (options.StartToCloseTimeout is null && options.ScheduleToCloseTimeout is null)
        {
            options.StartToCloseTimeout = DefaultCallToolTimeout;
        }

        options.Summary ??= $"{server}/{tool}";
        return options;
    }
}
