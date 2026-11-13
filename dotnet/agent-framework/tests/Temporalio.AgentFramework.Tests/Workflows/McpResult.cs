namespace Temporalio.AgentFramework.Tests.Workflows;

public sealed record McpResult(string Text, IList<string> ToolNames, string AddSchema);
