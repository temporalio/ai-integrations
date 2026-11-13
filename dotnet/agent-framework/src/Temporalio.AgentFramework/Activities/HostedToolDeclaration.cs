using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Temporalio.AgentFramework.Activities;

/// <summary>
/// A provider-executed tool (<see cref="HostedWebSearchTool"/>, <see cref="HostedCodeInterpreterTool"/>,
/// <see cref="HostedFileSearchTool"/> or <see cref="HostedMcpServerTool"/>) carried to the model
/// Activity by kind and options. The provider runs these inside the model call; the Workflow never
/// invokes them.
/// </summary>
public sealed record HostedToolDeclaration
{
    /// <summary>Kind value for <see cref="HostedWebSearchTool"/>.</summary>
    public const string WebSearch = "web_search";

    /// <summary>Kind value for <see cref="HostedCodeInterpreterTool"/>.</summary>
    public const string CodeInterpreter = "code_interpreter";

    /// <summary>Kind value for <see cref="HostedFileSearchTool"/>.</summary>
    public const string FileSearch = "file_search";

    /// <summary>Kind value for <see cref="HostedMcpServerTool"/>.</summary>
    public const string McpServer = "mcp_server";

    /// <summary>Gets the tool kind: one of the constants on this type.</summary>
    public required string Kind { get; init; }

    /// <summary>Gets the tool's additional properties, serialized as a JSON object.</summary>
    public JsonElement? AdditionalProperties { get; init; }

    /// <summary>Gets the inputs of a code interpreter or file search tool.</summary>
    public IReadOnlyList<AIContent>? Inputs { get; init; }

    /// <summary>Gets the maximum result count of a file search tool.</summary>
    public int? MaximumResultCount { get; init; }

    /// <summary>Gets the server name of a hosted MCP server tool.</summary>
    public string? ServerName { get; init; }

    /// <summary>Gets the server address of a hosted MCP server tool, when it has one.</summary>
    public string? ServerAddress { get; init; }

    /// <summary>Gets the server description of a hosted MCP server tool.</summary>
    public string? ServerDescription { get; init; }

    /// <summary>Gets the allowed tool names of a hosted MCP server tool.</summary>
    public IReadOnlyList<string>? AllowedTools { get; init; }

    /// <summary>Gets the HTTP headers of a hosted MCP server tool.</summary>
    public IReadOnlyDictionary<string, string>? Headers { get; init; }

    /// <summary>
    /// Gets the approval mode of a hosted MCP server tool: <c>always</c>, <c>never</c>, or null for
    /// the provider default. Specific-tool modes carry their lists in
    /// <see cref="AlwaysRequireApprovalToolNames"/> and <see cref="NeverRequireApprovalToolNames"/>.
    /// </summary>
    public string? ApprovalMode { get; init; }

    /// <summary>Gets the tool names that always require approval, for a specific approval mode.</summary>
    public IReadOnlyList<string>? AlwaysRequireApprovalToolNames { get; init; }

    /// <summary>Gets the tool names that never require approval, for a specific approval mode.</summary>
    public IReadOnlyList<string>? NeverRequireApprovalToolNames { get; init; }
}
