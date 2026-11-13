namespace Temporalio.AgentFramework.Tests.Workflows;

public sealed record ApprovalInput(string Prompt, string ToolKind, bool WithPlainTool = false);
