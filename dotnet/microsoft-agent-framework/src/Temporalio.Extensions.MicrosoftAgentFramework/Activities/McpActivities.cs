using System.Text.Json;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using Temporalio.Activities;
using Temporalio.Exceptions;
using Temporalio.Extensions.MicrosoftAgentFramework.Internal;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Activities;

/// <summary>
/// The MCP Activities. Registered by <see cref="MicrosoftAgentFrameworkPlugin"/>; connections come from
/// the transports registered under the input's server name and are shared across calls.
/// </summary>
public sealed class McpActivities
{
    /// <summary>Name of the list-tools Activity.</summary>
    public const string ListToolsActivityName = "MicrosoftAgentFramework.McpListTools";

    /// <summary>Name of the call-tool Activity.</summary>
    public const string CallToolActivityName = "MicrosoftAgentFramework.McpCallTool";

    private readonly McpConnectionCache connections;

    /// <summary>
    /// Initializes a new instance of the <see cref="McpActivities"/> class.
    /// </summary>
    /// <param name="connections">Connection cache shared by every worker running with the plugin instance.</param>
    internal McpActivities(McpConnectionCache connections) => this.connections = connections;

    /// <summary>
    /// Lists the tools of a server with their full input schemas.
    /// </summary>
    /// <param name="input">Server name and cache flag.</param>
    /// <returns>One declaration per tool.</returns>
    [Activity(ListToolsActivityName)]
    public Task<IReadOnlyList<ToolDeclaration>> ListToolsAsync(McpListToolsInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        return connections.ListToolsAsync(input.Server, input.CacheTools, ActivityExecutionContext.Current.CancellationToken);
    }

    /// <summary>
    /// Calls one tool and returns the server's result as it came, error flag included, so the
    /// Workflow-side tool can hand the model the server's text the way the MCP C# SDK's
    /// <c>McpClientTool</c> does. A protocol error for an unknown tool or bad arguments fails with
    /// <see cref="MicrosoftAgentFrameworkErrorTypes.McpError"/> and no retry, as <c>McpClientTool</c>
    /// throws <c>McpException</c>.
    /// </summary>
    /// <param name="input">Server, tool and arguments.</param>
    /// <returns>The MCP call result as JSON: its content blocks, structured content and error flag.</returns>
    [Activity(CallToolActivityName)]
    public async Task<JsonElement> CallToolAsync(McpToolCall input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var context = ActivityExecutionContext.Current;
        CallToolResult result;
        try
        {
            result = await connections.UseAsync(
                input.Server,
                (client, token) => client.CallToolAsync(input.Name, input.Arguments, cancellationToken: token).AsTask(),
                context.CancellationToken);
        }
        catch (McpProtocolException exception) when (exception.ErrorCode is McpErrorCode.InvalidParams or McpErrorCode.MethodNotFound or McpErrorCode.InvalidRequest)
        {
            throw new ApplicationFailureException(
                $"MCP tool '{input.Name}' on server '{input.Server}' rejected the call: {exception.Message}",
                exception,
                errorType: MicrosoftAgentFrameworkErrorTypes.McpError,
                nonRetryable: true);
        }

        return JsonSerializer.SerializeToElement(result, McpJsonUtilities.DefaultOptions.GetTypeInfo(typeof(CallToolResult)));
    }
}
