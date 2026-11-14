namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed record ToolsInput(
    string Prompt,
    string ToolKind,
    bool Concurrent = false,
    string Model = TestPlugin.Model,
    string? AgentId = null,
    string? AgentName = null);
