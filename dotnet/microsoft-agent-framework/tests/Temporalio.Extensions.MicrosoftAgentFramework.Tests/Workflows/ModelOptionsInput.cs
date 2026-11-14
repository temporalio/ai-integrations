namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed record ModelOptionsInput(string Prompt, double? StartToCloseSeconds = null, string? Summary = null, double? HeartbeatSeconds = null);
