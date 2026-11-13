namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record StreamingResult(IList<string> Texts, int UpdateCount, string FinalText, int ToolUpdates, long? TotalTokens);
