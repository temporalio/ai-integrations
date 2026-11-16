namespace Temporalio.Extensions.MicrosoftAgentFramework.Activities;

/// <summary>
/// Input of the <c>MicrosoftAgentFramework.McpListTools</c> Activity.
/// </summary>
/// <param name="Server">Server name registered in <see cref="MicrosoftAgentFrameworkPluginOptions.McpServers"/>.</param>
/// <param name="CacheTools">
/// Whether the worker may answer from the tool list it fetched earlier on the same connection.
/// </param>
public sealed record McpListToolsInput(string Server, bool CacheTools = false);
