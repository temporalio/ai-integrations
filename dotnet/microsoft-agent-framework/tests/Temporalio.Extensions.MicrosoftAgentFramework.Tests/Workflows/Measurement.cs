using System.Text.Json.Serialization;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

// A polymorphic result: System.Text.Json writes its $type discriminator, and the names collide with
// Microsoft.Extensions.AI content types on purpose.
[JsonPolymorphic]
[JsonDerivedType(typeof(DataMeasurement), "data")]
[JsonDerivedType(typeof(ErrorMeasurement), "error")]
public abstract class Measurement
{
    public string? Sensor { get; set; }
}
