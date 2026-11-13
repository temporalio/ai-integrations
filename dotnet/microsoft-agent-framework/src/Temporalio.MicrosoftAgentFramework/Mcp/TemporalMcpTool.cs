using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using Temporalio.MicrosoftAgentFramework.Activities;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Mcp;

/// <summary>
/// A tool on an MCP server, callable from a Workflow. Each invocation is one
/// <c>MicrosoftAgentFramework.McpCallTool</c> Activity. <see cref="TemporalMcpServer.ListToolsAsync"/>
/// builds these from the server's catalog; build them yourself to pin a catalog in code. The
/// result reaches the model the way the MCP C# SDK's <c>McpClientTool</c> delivers it: plain
/// content blocks as <see cref="AIContent"/>, anything with an error flag, structured content or
/// application metadata as the whole result.
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
    /// The mapping <c>McpClientTool.InvokeCoreAsync</c> applies: when the result carries no error
    /// flag, no structured content and no application metadata, one content block becomes one
    /// <see cref="AIContent"/> and several become a list; otherwise the whole result is returned,
    /// so the model reads the server's error text and can correct the call.
    /// </summary>
    /// <param name="element">The call result the Activity returned.</param>
    /// <returns>The function result.</returns>
    internal static object ToFunctionResult(JsonElement element)
    {
        if (element.Deserialize(McpJsonUtilities.DefaultOptions.GetTypeInfo(typeof(CallToolResult))) is not CallToolResult result)
        {
            return element;
        }

        if (result.IsError is not true && result.StructuredContent is null && !HasApplicationResultMetadata(result.Meta))
        {
            switch (result.Content.Count)
            {
                case 1 when result.Content[0].ToAIContent(McpJsonUtilities.DefaultOptions) is { } content:
                    return content;
                case > 1 when result.Content.Select(block => block.ToAIContent(McpJsonUtilities.DefaultOptions)).ToArray() is { } contents
                              && Array.TrueForAll(contents, content => content is not null):
                    return contents;
                default:
                    break;
            }
        }

        return element;
    }

    /// <summary>
    /// Runs the call-tool Activity and maps its result as <c>McpClientTool</c> does.
    /// </summary>
    /// <param name="arguments">Arguments the model supplied.</param>
    /// <param name="cancellationToken">Cancels the Activity.</param>
    /// <returns>An <see cref="AIContent"/>, a list of them, or the whole call result as a <see cref="JsonElement"/>.</returns>
    protected override async ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        var call = new McpToolCall(Server, ServerToolName, new Dictionary<string, object?>(arguments, StringComparer.Ordinal));
        var activityOptions = options.ResolveCallToolOptions(Server, ServerToolName);
        if (cancellationToken.CanBeCanceled && activityOptions.CancellationToken is null)
        {
            activityOptions.CancellationToken = cancellationToken;
        }

        var element = await Workflow.ExecuteActivityAsync(
            (McpActivities activities) => activities.CallToolAsync(call), activityOptions).ConfigureAwait(true);
        return ToFunctionResult(element);
    }

    // Server identity is protocol metadata; anything else in _meta is application data the model should see.
    private static bool HasApplicationResultMetadata(JsonObject? meta) =>
        meta is not null && meta.Any(property => !string.Equals(property.Key, MetaKeys.ServerInfo, StringComparison.Ordinal));
}
