using System.Text.Json;
using Microsoft.Agents.AI;

namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// The session a <see cref="TemporalAgent"/> runs in. The state bag holds the chat history and
/// provider state; <c>agent.SerializeSessionAsync(session)</c> turns it into a
/// <see cref="JsonElement"/> you pass through continue-as-new and restore with
/// <c>agent.DeserializeSessionAsync(element)</c>.
/// </summary>
public sealed class TemporalAgentSession : AgentSession
{
    private const string StateBagProperty = "stateBag";

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalAgentSession"/> class with an empty state bag.
    /// </summary>
    public TemporalAgentSession()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalAgentSession"/> class around an existing state bag.
    /// </summary>
    /// <param name="stateBag">State to continue from.</param>
    public TemporalAgentSession(AgentSessionStateBag stateBag)
        : base(stateBag)
    {
    }

    /// <summary>
    /// Restores a session written by <see cref="Serialize"/>.
    /// </summary>
    /// <param name="serialized">The serialized session.</param>
    /// <returns>The session.</returns>
    public static TemporalAgentSession Deserialize(JsonElement serialized)
    {
        if (serialized.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException("A serialized TemporalAgentSession is a JSON object.", nameof(serialized));
        }

        return serialized.TryGetProperty(StateBagProperty, out var bag)
            ? new TemporalAgentSession(AgentSessionStateBag.Deserialize(bag))
            : new TemporalAgentSession();
    }

    /// <summary>
    /// Serializes the session as <c>{"stateBag": ...}</c>, the shape <c>ChatClientAgent</c> uses.
    /// </summary>
    /// <returns>The serialized session.</returns>
    public JsonElement Serialize()
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName(StateBagProperty);
            StateBag.Serialize().WriteTo(writer);
            writer.WriteEndObject();
        }

        using var document = JsonDocument.Parse(buffer.ToArray());
        return document.RootElement.Clone();
    }
}
