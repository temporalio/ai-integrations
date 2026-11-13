using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.AgentFramework.Activities;

namespace Temporalio.AgentFramework.Internal;

/// <summary>
/// Moves <see cref="ChatOptions"/> across the Activity boundary. Function tools travel as
/// <see cref="ToolDeclaration"/>s and provider-executed tools as <see cref="HostedToolDeclaration"/>s;
/// <c>RawRepresentationFactory</c> and <c>ContinuationToken</c> are dropped with a warning. The
/// drop list follows Cecil Phillip's draft at https://github.com/temporal-community/temporal-dotnet-agents.
/// </summary>
internal static class ChatOptionsTransport
{
    /// <summary>
    /// Splits <paramref name="options"/> into the serializable options and the tool declarations.
    /// Runs on the Workflow side.
    /// </summary>
    /// <param name="options">Options the caller passed, or null.</param>
    /// <param name="logger">Logger for the drop warnings.</param>
    /// <returns>The transport options and the declarations, each null when absent.</returns>
    public static (ChatOptions? Options, IReadOnlyList<ToolDeclaration>? Tools, IReadOnlyList<HostedToolDeclaration>? HostedTools) Prepare(
        ChatOptions? options, ILogger logger)
    {
        if (options is null)
        {
            return (null, null, null);
        }

        var transport = options.Clone();
        if (transport.RawRepresentationFactory is not null)
        {
            logger.LogWarning("ChatOptions.RawRepresentationFactory is not serializable and is dropped before the model Activity. Configure the provider request in the chat client factory instead.");
            transport.RawRepresentationFactory = null;
        }

#pragma warning disable MEAI001 // ContinuationToken is experimental; the plugin only clears it.
        if (transport.ContinuationToken is not null)
        {
            logger.LogWarning("ChatOptions.ContinuationToken is dropped before the model Activity. Background responses are not supported.");
            transport.ContinuationToken = null;
        }
#pragma warning restore MEAI001

        List<ToolDeclaration>? tools = null;
        List<HostedToolDeclaration>? hosted = null;
        if (transport.Tools is { Count: > 0 })
        {
            foreach (var tool in transport.Tools)
            {
                switch (tool)
                {
                    case AIFunctionDeclaration function:
                        (tools ??= []).Add(new ToolDeclaration(function.Name, function.Description, function.JsonSchema, function.ReturnJsonSchema));
                        break;
                    case HostedWebSearchTool or HostedCodeInterpreterTool or HostedFileSearchTool or HostedMcpServerTool:
                        (hosted ??= []).Add(ToDeclaration(tool));
                        break;
                    default:
                        throw new AgentFrameworkWorkflowException(
                            $"ChatOptions.Tools contains a {tool.GetType().Name}, which the plugin cannot send to the model Activity. Use AIFunction tools, ActivityAsTool, TemporalMcpTool or the hosted tool types.");
                }
            }
        }

        transport.Tools = null;
        return (transport, tools, hosted);
    }

    /// <summary>
    /// Rebuilds the options the provider client receives. Runs on the worker.
    /// </summary>
    /// <param name="input">The Activity input.</param>
    /// <returns>Options with declaration-only function tools and reconstructed hosted tools, or null.</returns>
    public static ChatOptions? Rebuild(ModelCallInput input)
    {
        var options = input.Options?.Clone();
        if (options?.AdditionalProperties is { Count: > 0 } properties)
        {
            // The converter delivers object-typed values as JsonElement; provider clients read these
            // entries with typed checks (bool, string, numbers), so JSON primitives go back to CLR values.
            foreach (var key in properties.Keys.ToList())
            {
                if (properties[key] is JsonElement element)
                {
                    properties[key] = FromJson(element);
                }
            }
        }

        List<AITool>? tools = null;
        foreach (var declaration in input.Tools ?? [])
        {
            (tools ??= []).Add(AIFunctionFactory.CreateDeclaration(
                declaration.Name, declaration.Description, declaration.JsonSchema, declaration.ReturnJsonSchema));
        }

        foreach (var hosted in input.HostedTools ?? [])
        {
            (tools ??= []).Add(FromDeclaration(hosted));
        }

        if (tools is not null)
        {
            options ??= new ChatOptions();
            options.Tools = tools;
        }

        return options;
    }

    private static object? FromJson(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number when element.TryGetInt64(out var integer) => integer,
        JsonValueKind.Number => element.GetDouble(),
        _ => element,
    };

    private static HostedToolDeclaration ToDeclaration(AITool tool)
    {
        var additional = tool.AdditionalProperties.Count == 0
            ? (JsonElement?)null
            : JsonSerializer.SerializeToElement(tool.AdditionalProperties, AIJsonUtilities.DefaultOptions);
        return tool switch
        {
            HostedWebSearchTool => new HostedToolDeclaration { Kind = HostedToolDeclaration.WebSearch, AdditionalProperties = additional },
            HostedCodeInterpreterTool code => new HostedToolDeclaration
            {
                Kind = HostedToolDeclaration.CodeInterpreter,
                AdditionalProperties = additional,
                Inputs = code.Inputs?.ToList(),
            },
            HostedFileSearchTool files => new HostedToolDeclaration
            {
                Kind = HostedToolDeclaration.FileSearch,
                AdditionalProperties = additional,
                Inputs = files.Inputs?.ToList(),
                MaximumResultCount = files.MaximumResultCount,
            },
            HostedMcpServerTool mcp => new HostedToolDeclaration
            {
                Kind = HostedToolDeclaration.McpServer,
                AdditionalProperties = additional,
                ServerName = mcp.ServerName,
                ServerAddress = mcp.ServerAddress,
                ServerDescription = mcp.ServerDescription,
                AllowedTools = mcp.AllowedTools?.ToList(),
                Headers = mcp.Headers?.ToDictionary(pair => pair.Key, pair => pair.Value),
                ApprovalMode = mcp.ApprovalMode switch
                {
                    HostedMcpServerToolAlwaysRequireApprovalMode => "always",
                    HostedMcpServerToolNeverRequireApprovalMode => "never",
                    HostedMcpServerToolRequireSpecificApprovalMode => "specific",
                    _ => null,
                },
                AlwaysRequireApprovalToolNames = (mcp.ApprovalMode as HostedMcpServerToolRequireSpecificApprovalMode)?.AlwaysRequireApprovalToolNames?.ToList(),
                NeverRequireApprovalToolNames = (mcp.ApprovalMode as HostedMcpServerToolRequireSpecificApprovalMode)?.NeverRequireApprovalToolNames?.ToList(),
            },
            _ => throw new ArgumentException($"Unsupported hosted tool {tool.GetType().Name}", nameof(tool)),
        };
    }

    private static AITool FromDeclaration(HostedToolDeclaration declaration)
    {
        var additional = declaration.AdditionalProperties is { } element
            ? element.Deserialize<Dictionary<string, object?>>(AIJsonUtilities.DefaultOptions)
            : null;
        switch (declaration.Kind)
        {
            case HostedToolDeclaration.WebSearch:
                return new HostedWebSearchTool(additional);
            case HostedToolDeclaration.CodeInterpreter:
                return new HostedCodeInterpreterTool(additional) { Inputs = declaration.Inputs?.ToList() };
            case HostedToolDeclaration.FileSearch:
                return new HostedFileSearchTool(additional)
                {
                    Inputs = declaration.Inputs?.ToList(),
                    MaximumResultCount = declaration.MaximumResultCount,
                };
            case HostedToolDeclaration.McpServer:
                var mcp = new HostedMcpServerTool(declaration.ServerName!, declaration.ServerAddress!, additional)
                {
                    ServerDescription = declaration.ServerDescription,
                    AllowedTools = declaration.AllowedTools?.ToList(),
                    Headers = declaration.Headers?.ToDictionary(pair => pair.Key, pair => pair.Value),
                    ApprovalMode = declaration.ApprovalMode switch
                    {
                        "always" => HostedMcpServerToolApprovalMode.AlwaysRequire,
                        "never" => HostedMcpServerToolApprovalMode.NeverRequire,
                        "specific" => HostedMcpServerToolApprovalMode.RequireSpecific(
                            declaration.AlwaysRequireApprovalToolNames?.ToList(),
                            declaration.NeverRequireApprovalToolNames?.ToList()),
                        _ => null,
                    },
                };
                return mcp;
            default:
                throw new ArgumentException($"Unknown hosted tool kind '{declaration.Kind}'", nameof(declaration));
        }
    }
}
