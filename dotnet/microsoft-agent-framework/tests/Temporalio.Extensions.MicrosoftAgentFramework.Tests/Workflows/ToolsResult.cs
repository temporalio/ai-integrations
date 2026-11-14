namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed record ToolsResult(string Text, string AgentId, int MessageCount, long? TotalTokens, IList<string> AuthorNames);
