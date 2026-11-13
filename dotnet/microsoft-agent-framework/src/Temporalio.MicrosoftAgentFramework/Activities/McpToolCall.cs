namespace Temporalio.MicrosoftAgentFramework.Activities;

/// <summary>
/// Input of the <c>MicrosoftAgentFramework.McpCallTool</c> Activity.
/// </summary>
/// <param name="Server">Server name registered in <see cref="MicrosoftAgentFrameworkPluginOptions.McpServers"/>.</param>
/// <param name="Name">Tool name on the server, without any <see cref="Mcp.McpToolOptions.ToolNamePrefix"/>.</param>
/// <param name="Arguments">Arguments the model supplied.</param>
public sealed record McpToolCall(
    string Server,
    string Name,
    IReadOnlyDictionary<string, object?>? Arguments);
