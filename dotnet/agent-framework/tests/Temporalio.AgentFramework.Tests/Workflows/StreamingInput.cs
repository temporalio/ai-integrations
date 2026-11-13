using Temporalio.Extensions.WorkflowStreams;

namespace Temporalio.AgentFramework.Tests.Workflows;

public sealed record StreamingInput(string Prompt, string? Topic = null, bool WithTool = false, WorkflowStreamState? StreamState = null);
