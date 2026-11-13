namespace Temporalio.AgentFramework.Activities;

/// <summary>
/// Input of the <c>AgentFramework.McpListTools</c> Activity.
/// </summary>
/// <param name="Server">Server name registered in <see cref="AgentFrameworkPluginOptions.McpServers"/>.</param>
/// <param name="CacheTools">
/// Whether the worker may answer from the tool list it fetched earlier on the same connection.
/// </param>
public sealed record McpListToolsInput(string Server, bool CacheTools = false);
