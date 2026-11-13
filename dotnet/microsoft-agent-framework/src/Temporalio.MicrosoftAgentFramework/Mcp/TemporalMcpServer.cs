using Microsoft.Extensions.AI;
using Temporalio.MicrosoftAgentFramework.Activities;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Mcp;

/// <summary>
/// Workflow-side entry point for MCP servers registered in
/// <see cref="MicrosoftAgentFrameworkPluginOptions.McpServers"/>.
/// </summary>
public static class TemporalMcpServer
{
    /// <summary>
    /// Lists a server's tools with one <c>MicrosoftAgentFramework.McpListTools</c> Activity and returns a
    /// <see cref="TemporalMcpTool"/> per tool, ready for <c>ChatOptions.Tools</c>. Each tool call
    /// is then one <c>MicrosoftAgentFramework.McpCallTool</c> Activity.
    /// </summary>
    /// <param name="server">Server name.</param>
    /// <param name="options">Filtering, naming and Activity options; defaults when null.</param>
    /// <returns>The tools, in the order the server listed them.</returns>
    public static async Task<IReadOnlyList<AIFunction>> ListToolsAsync(string server, McpToolOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(server);
        options ??= new McpToolOptions();
        var input = new McpListToolsInput(server, options.CacheTools);
        var declarations = await Workflow.ExecuteActivityAsync(
            (McpActivities activities) => activities.ListToolsAsync(input),
            options.ResolveListToolsOptions(server)).ConfigureAwait(true);
        var allowed = options.AllowedTools is null ? null : new HashSet<string>(options.AllowedTools, StringComparer.Ordinal);
        return declarations
            .Where(declaration => allowed is null || allowed.Contains(declaration.Name))
            .Select(declaration => (AIFunction)new TemporalMcpTool(server, declaration, options))
            .ToList();
    }
}
