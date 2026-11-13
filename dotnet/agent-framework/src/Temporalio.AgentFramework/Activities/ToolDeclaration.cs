using System.Text.Json;

namespace Temporalio.AgentFramework.Activities;

/// <summary>
/// A function tool as the model sees it: name, description and the full JSON schema of its
/// parameters. The Workflow sends declarations to the model Activity because
/// <c>ChatOptions.Tools</c> is not serializable; the Activity rebuilds declaration-only tools from
/// them. MCP servers return the same shape from the list-tools Activity.
/// </summary>
/// <param name="Name">Tool name as advertised to the model.</param>
/// <param name="Description">Tool description, or null when the tool has none.</param>
/// <param name="JsonSchema">JSON schema of the tool's parameters.</param>
/// <param name="ReturnJsonSchema">JSON schema of the tool's return value, when the tool declares one.</param>
public sealed record ToolDeclaration(
    string Name,
    string? Description,
    JsonElement JsonSchema,
    JsonElement? ReturnJsonSchema = null);
