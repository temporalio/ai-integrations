using System.Text.Json;
using Microsoft.Extensions.AI;
using Temporalio.MicrosoftAgentFramework.Activities;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Mcp;

/// <summary>
/// A tool on an MCP server, callable from a Workflow. Each invocation is one
/// <c>MicrosoftAgentFramework.McpCallTool</c> Activity. <see cref="TemporalMcpServer.ListToolsAsync"/>
/// builds these from the server's catalog; build them yourself to pin a catalog in code.
/// </summary>
public sealed class TemporalMcpTool : AIFunction
{
    private readonly McpToolOptions options;

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalMcpTool"/> class.
    /// </summary>
    /// <param name="server">Server name registered in <see cref="MicrosoftAgentFrameworkPluginOptions.McpServers"/>.</param>
    /// <param name="declaration">The tool as the server lists it.</param>
    /// <param name="options">Activity options and naming; defaults when null.</param>
    public TemporalMcpTool(string server, ToolDeclaration declaration, McpToolOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(server);
        ArgumentNullException.ThrowIfNull(declaration);
        this.options = options ?? new McpToolOptions();
        Server = server;
        ServerToolName = declaration.Name;
        Name = McpToolNames.Build(this.options.ToolNamePrefix, declaration.Name);
        Description = declaration.Description ?? string.Empty;
        JsonSchema = declaration.JsonSchema;
        ReturnJsonSchema = declaration.ReturnJsonSchema;
    }

    /// <summary>Gets the server name.</summary>
    public string Server { get; }

    /// <summary>Gets the tool name on the server, without <see cref="McpToolOptions.ToolNamePrefix"/>.</summary>
    public string ServerToolName { get; }

    /// <summary>Gets the name the model sees.</summary>
    public override string Name { get; }

    /// <inheritdoc />
    public override string Description { get; }

    /// <summary>Gets the input schema the server published.</summary>
    public override JsonElement JsonSchema { get; }

    /// <inheritdoc />
    public override JsonElement? ReturnJsonSchema { get; }

    /// <summary>
    /// Runs the call-tool Activity. The result is the MCP call result as JSON.
    /// </summary>
    /// <param name="arguments">Arguments the model supplied.</param>
    /// <param name="cancellationToken">Cancels the Activity.</param>
    /// <returns>The call result as a <see cref="JsonElement"/>.</returns>
    protected override async ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        var call = new McpToolCall(Server, ServerToolName, new Dictionary<string, object?>(arguments, StringComparer.Ordinal));
        var activityOptions = options.ResolveCallToolOptions(Server, ServerToolName);
        if (cancellationToken.CanBeCanceled && activityOptions.CancellationToken is null)
        {
            activityOptions.CancellationToken = cancellationToken;
        }

        return await Workflow.ExecuteActivityAsync(
            (McpActivities activities) => activities.CallToolAsync(call), activityOptions).ConfigureAwait(true);
    }
}
