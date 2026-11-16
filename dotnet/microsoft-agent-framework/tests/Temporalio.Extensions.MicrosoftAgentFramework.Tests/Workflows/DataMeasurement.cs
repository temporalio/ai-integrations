using System.Text.Json.Serialization;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed class DataMeasurement : Measurement
{
    // Serialized as "uri", the property DataContent would read, so the collision is the one a tool could hit.
    [JsonPropertyName("uri")]
    public string? Location { get; set; }

    public double Value { get; set; }
}
