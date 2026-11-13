using Temporalio.Extensions.WorkflowStreams;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record StreamingInput(
    string Prompt,
    string? Topic = null,
    bool WithTool = false,
    WorkflowStreamState? StreamState = null,
    TimeSpan? BatchInterval = null,
    bool HostStream = true);
