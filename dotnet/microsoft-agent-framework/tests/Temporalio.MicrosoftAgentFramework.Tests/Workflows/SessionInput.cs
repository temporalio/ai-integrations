using System.Text.Json;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record SessionInput(
    IList<string> Prompts,
    int Index = 0,
    JsonElement? Session = null,
    IList<string>? Replies = null,
    bool ContinueAsNew = false,
    string HistoryProvider = "memory",
    string? ContextProvider = null,
    string ProviderWait = ProviderWaits.TaskDelay);
