using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.Extensions.MicrosoftAgentFramework.Activities;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Internal;

/// <summary>
/// Moves <see cref="ChatOptions"/> across the Activity boundary. Function tools travel as
/// <see cref="ToolDeclaration"/>s, with their additional properties, and provider-executed tools as
/// <see cref="HostedToolDeclaration"/>s. <c>RawRepresentationFactory</c> is dropped with a warning;
/// <c>AllowBackgroundResponses</c>, <c>ConversationId</c> and <c>ContinuationToken</c> fail the run,
/// as they do on <c>AgentRunOptions</c>, because the Workflow holds the conversation and waits for
/// each complete response. The drop list follows Cecil Phillip's draft at
/// https://github.com/temporal-community/temporal-dotnet-agents.
/// The HTTP headers of a <see cref="HostedMcpServerTool"/> never cross: a tool that carries them on
/// the Workflow side fails the run, and the worker adds the headers registered under
/// <see cref="MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaders"/> when it rebuilds the tool.
/// An <c>AdditionalProperties</c> entry named like a credential header fails the run the same way.
/// </summary>
internal static class ChatOptionsTransport
{
    private static readonly HashSet<string> SecretKeys = new(["authorization", "x-api-key", "api-key"], StringComparer.OrdinalIgnoreCase);

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
        RejectSecrets(transport.AdditionalProperties, "ChatOptions.AdditionalProperties");
        if (transport.AllowBackgroundResponses is true)
        {
            throw new MicrosoftAgentFrameworkWorkflowException("ChatOptions.AllowBackgroundResponses is not supported; the model Activity waits for the complete response.");
        }

        if (transport.ConversationId is not null)
        {
            throw new MicrosoftAgentFrameworkWorkflowException(
                "ChatOptions.ConversationId is not supported: the Workflow holds the conversation and sends it on every call, so a service-managed conversation would make the model see every turn twice. Leave ConversationId unset.");
        }

#pragma warning disable MEAI001 // ContinuationToken is experimental; the plugin only rejects it.
        if (transport.ContinuationToken is not null)
        {
            throw new MicrosoftAgentFrameworkWorkflowException("ChatOptions.ContinuationToken is not supported; background responses do not cross the Activity boundary.");
        }
#pragma warning restore MEAI001

        if (transport.RawRepresentationFactory is not null)
        {
            logger.LogWarning("ChatOptions.RawRepresentationFactory is not serializable and is dropped before the model Activity. Configure the provider request in the chat client factory instead.");
            transport.RawRepresentationFactory = null;
        }

        List<ToolDeclaration>? tools = null;
        List<HostedToolDeclaration>? hosted = null;
        if (transport.Tools is { Count: > 0 })
        {
            foreach (var tool in transport.Tools)
            {
                switch (tool)
                {
                    case AIFunctionDeclaration function:
                        (tools ??= []).Add(new ToolDeclaration(
                            function.Name,
                            function.Description,
                            function.JsonSchema,
                            function.ReturnJsonSchema,
                            function.AdditionalProperties.Count == 0 ? null : JsonSerializer.SerializeToElement(function.AdditionalProperties, AIJsonUtilities.DefaultOptions)));
                        break;
                    case HostedMcpServerTool { Headers: { Count: > 0 } headers } mcp:
                        throw new MicrosoftAgentFrameworkWorkflowException(
                            $"HostedMcpServerTool '{mcp.ServerName}' has {headers.Count} HTTP header(s) set in the Workflow. " +
                            "The model Activity's input is recorded in Event History, so the plugin sends no hosted MCP server headers from the Workflow. " +
                            $"Register them on the worker under MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaders[\"{mcp.ServerName}\"] and leave Headers unset here.");
                    case HostedWebSearchTool or HostedCodeInterpreterTool or HostedFileSearchTool or HostedMcpServerTool:
                        RejectSecrets(tool.AdditionalProperties, $"{tool.GetType().Name}.AdditionalProperties");
                        (hosted ??= []).Add(ToDeclaration(tool));
                        break;
                    default:
                        throw new MicrosoftAgentFrameworkWorkflowException(
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
    /// <param name="hostedMcpServerHeaders">Returns the HTTP headers for a hosted MCP server tool, by server name, or null for none.</param>
    /// <returns>Options with declaration-only function tools and reconstructed hosted tools, or null.</returns>
    public static ChatOptions? Rebuild(ModelCallInput input, Func<string, IReadOnlyDictionary<string, string>?> hostedMcpServerHeaders)
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
            (tools ??= []).Add(new TransportedFunctionDeclaration(declaration));
        }

        foreach (var hosted in input.HostedTools ?? [])
        {
            (tools ??= []).Add(FromDeclaration(hosted, hostedMcpServerHeaders));
        }

        if (tools is not null)
        {
            options ??= new ChatOptions();
            options.Tools = tools;
        }

        return options;
    }

    /// <summary>
    /// Turns the tool results the Workflow serialized with <c>AIJsonUtilities</c> back into
    /// <see cref="AIContent"/> objects, so the provider client applies its own handling of text,
    /// image and file results. Runs on the worker; anything else stays as it arrived.
    /// </summary>
    /// <param name="messages">The Activity input's messages; changed in place.</param>
    public static void RehydrateToolResults(IEnumerable<ChatMessage> messages)
    {
        foreach (var result in messages.SelectMany(message => message.Contents).OfType<FunctionResultContent>())
        {
            try
            {
                result.Result = result.Result switch
                {
                    JsonElement { ValueKind: JsonValueKind.Object } element when element.TryGetProperty("$type", out _) =>
                        element.Deserialize(AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(AIContent))) ?? result.Result,
                    JsonElement { ValueKind: JsonValueKind.Array } element when element.GetArrayLength() > 0 &&
                        element.EnumerateArray().All(item => item.ValueKind == JsonValueKind.Object && item.TryGetProperty("$type", out _)) =>
                        element.Deserialize(AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(IList<AIContent>))) ?? result.Result,
                    _ => result.Result,
                };
            }
            catch (JsonException)
            {
                // A $type this version of Microsoft.Extensions.AI does not know stays JSON; the model reads the text either way.
            }
        }
    }

    /// <summary>Turns a JSON primitive back into the CLR value a provider client checks for; other kinds stay <see cref="JsonElement"/>.</summary>
    /// <param name="element">The element.</param>
    /// <returns>The value.</returns>
    internal static object? FromJson(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number when element.TryGetInt64(out var integer) => integer,
        JsonValueKind.Number => element.GetDouble(),
        _ => element,
    };

    // Credentials belong in the worker factory: everything here is recorded in Event History with the
    // model Activity's input. The check covers the top level and one nested headers dictionary, the
    // two places a provider client reads such entries from.
    private static void RejectSecrets(IEnumerable<KeyValuePair<string, object?>>? properties, string where)
    {
        if (properties is null)
        {
            return;
        }

        foreach (var (key, value) in properties)
        {
            if (SecretKeys.Contains(key))
            {
                throw new MicrosoftAgentFrameworkWorkflowException(
                    $"{where} contains an entry named '{key}'. The model Activity's input is recorded in Event History, so a credential cannot travel in it; " +
                    "set credentials in the chat client factory on the worker.");
            }

            switch (value)
            {
                case IEnumerable<KeyValuePair<string, object?>> nested:
                    RejectSecrets(nested, $"{where}['{key}']");
                    break;
                case IEnumerable<KeyValuePair<string, string>> nestedStrings:
                    RejectSecrets(nestedStrings.Select(pair => new KeyValuePair<string, object?>(pair.Key, pair.Value)), $"{where}['{key}']");
                    break;
                default:
                    break;
            }
        }
    }

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

    private static AITool FromDeclaration(HostedToolDeclaration declaration, Func<string, IReadOnlyDictionary<string, string>?> hostedMcpServerHeaders)
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
                    Headers = hostedMcpServerHeaders(declaration.ServerName!) is { } headers ? new Dictionary<string, string>(headers) : null,
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
