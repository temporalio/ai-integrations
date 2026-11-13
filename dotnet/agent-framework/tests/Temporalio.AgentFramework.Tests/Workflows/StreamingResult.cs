namespace Temporalio.AgentFramework.Tests.Workflows;

public sealed record StreamingResult(IList<string> Texts, int UpdateCount, string FinalText, int ToolUpdates);
