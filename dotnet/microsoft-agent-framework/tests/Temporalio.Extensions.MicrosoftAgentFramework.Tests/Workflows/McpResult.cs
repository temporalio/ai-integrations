namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed record McpResult(string Text, IList<string> ToolNames, string AddSchema);
