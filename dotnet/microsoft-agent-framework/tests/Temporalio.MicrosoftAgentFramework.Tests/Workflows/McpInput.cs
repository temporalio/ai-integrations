namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record McpInput(
    string Server,
    string Prompt,
    IList<string>? AllowedTools = null,
    string? ToolNamePrefix = null,
    bool CacheTools = false,
    bool ListTwice = false,
    bool PinnedCatalog = false,
    int MaxAttempts = 0,
    string PinnedTool = "add");
