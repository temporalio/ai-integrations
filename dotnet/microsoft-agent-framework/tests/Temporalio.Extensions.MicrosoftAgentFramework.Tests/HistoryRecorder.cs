using Temporalio.Client;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>
/// Writes a run's history under TestResults/recorded-histories when
/// MICROSOFT_AGENT_FRAMEWORK_RECORD_HISTORIES is set, so the files under Histories/ can be refreshed from a
/// real run. Tests behave the same with or without the variable.
/// </summary>
public static class HistoryRecorder
{
    public static async Task SaveAsync(WorkflowHandle handle, string name)
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("MICROSOFT_AGENT_FRAMEWORK_RECORD_HISTORIES")))
        {
            return;
        }

        var history = await handle.FetchHistoryAsync();
        var directory = Path.Combine(PluginMeta.Root, "TestResults", "recorded-histories");
        Directory.CreateDirectory(directory);
        await File.WriteAllTextAsync(Path.Combine(directory, name + ".json"), history.ToJson());
    }
}
