namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record ApprovalInput(string Prompt, string ToolKind, bool WithPlainTool = false);
