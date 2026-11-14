using System.Text.Json;
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework.Activities;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Internal;

/// <summary>
/// The declaration-only tool the model Activity rebuilds from a <see cref="ToolDeclaration"/>.
/// <c>AIFunctionFactory.CreateDeclaration</c> takes no additional properties, and provider clients
/// read settings such as <c>strict</c> from them, so this type carries them across.
/// </summary>
internal sealed class TransportedFunctionDeclaration : AIFunctionDeclaration
{
    private static readonly IReadOnlyDictionary<string, object?> NoProperties = new Dictionary<string, object?>();

    /// <summary>
    /// Initializes a new instance of the <see cref="TransportedFunctionDeclaration"/> class.
    /// </summary>
    /// <param name="declaration">The declaration the Workflow sent.</param>
    public TransportedFunctionDeclaration(ToolDeclaration declaration)
    {
        ArgumentNullException.ThrowIfNull(declaration);
        Name = declaration.Name;
        Description = declaration.Description ?? string.Empty;
        JsonSchema = declaration.JsonSchema;
        ReturnJsonSchema = declaration.ReturnJsonSchema;
        AdditionalProperties = declaration.AdditionalProperties is { ValueKind: JsonValueKind.Object } element
            ? element.EnumerateObject().ToDictionary(property => property.Name, property => ChatOptionsTransport.FromJson(property.Value), StringComparer.Ordinal)
            : NoProperties;
    }

    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override string Description { get; }

    /// <inheritdoc />
    public override JsonElement JsonSchema { get; }

    /// <inheritdoc />
    public override JsonElement? ReturnJsonSchema { get; }

    /// <inheritdoc />
    public override IReadOnlyDictionary<string, object?> AdditionalProperties { get; }
}
