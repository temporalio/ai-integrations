namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed class ErrorMeasurement : Measurement
{
    public string? Message { get; set; }

    public int Code { get; set; }
}
