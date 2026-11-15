using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Internal;

/// <summary>
/// Wraps non-object response schemas in <c>{"data": ...}</c> the way <c>AIAgent.RunAsync&lt;T&gt;</c>
/// does, so <c>AgentResponse&lt;T&gt;.Result</c> unwraps them.
/// </summary>
internal static class StructuredOutput
{
    private const string DataPropertyName = "data";

    /// <summary>
    /// Builds the response format for <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">Result type.</typeparam>
    /// <param name="serializerOptions">Serializer options the schema is derived from.</param>
    /// <returns>The format and whether the schema was wrapped in an object.</returns>
    public static (ChatResponseFormatJson Format, bool IsWrappedInObject) FormatFor<T>(JsonSerializerOptions serializerOptions)
    {
        var format = ChatResponseFormat.ForJsonSchema<T>(serializerOptions);
        if (format.Schema is not { } schema)
        {
            throw new InvalidOperationException("The response format must have a JSON schema.");
        }

        if (RepresentsObject(schema))
        {
            return (format, false);
        }

        var wrapped = JsonSerializer.SerializeToElement(
            new JsonObject
            {
                { "$schema", "https://json-schema.org/draft/2020-12/schema" },
                { "type", "object" },
                { "properties", new JsonObject { { DataPropertyName, ToNode(schema) } } },
                { "additionalProperties", false },
                { "required", new JsonArray(DataPropertyName) },
            },
            AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(JsonObject)));
        return (ChatResponseFormat.ForJsonSchema(wrapped, format.SchemaName, format.SchemaDescription), true);
    }

    private static bool RepresentsObject(JsonElement schema)
    {
        if (schema.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        foreach (var property in schema.EnumerateObject())
        {
            if (property.NameEquals("type"u8))
            {
                return property.Value.ValueKind == JsonValueKind.String && property.Value.ValueEquals("object"u8);
            }
        }

        return false;
    }

    private static JsonNode? ToNode(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Null => null,
        JsonValueKind.Array => JsonArray.Create(element),
        JsonValueKind.Object => JsonObject.Create(element),
        _ => JsonValue.Create(element),
    };
}
