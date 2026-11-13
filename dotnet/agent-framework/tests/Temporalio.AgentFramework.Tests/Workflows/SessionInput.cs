using System.Text.Json;

namespace Temporalio.AgentFramework.Tests.Workflows;

public sealed record SessionInput(
    IList<string> Prompts,
    int Index = 0,
    JsonElement? Session = null,
    IList<string>? Replies = null,
    bool ContinueAsNew = false,
    string HistoryProvider = "memory",
    bool WithContextProvider = false);
