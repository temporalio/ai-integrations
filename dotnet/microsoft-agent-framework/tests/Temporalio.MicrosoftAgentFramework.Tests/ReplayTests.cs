using Temporalio.Common;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Worker;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class ReplayTests
{
    [Fact]
    public async Task Histories_ReplayThroughPlugin()
    {
        var directory = Path.Combine(AppContext.BaseDirectory, "Histories");
        var files = Directory.Exists(directory) ? Directory.GetFiles(directory, "*.json").OrderBy(f => f, StringComparer.Ordinal).ToList() : [];
        Assert.True(files.Count >= 11, $"expected at least eleven recorded histories under {directory}, found {files.Count}");

        var plugin = TestPlugin.For(new FakeChatClient(ScriptedResponse.Text("unused")));
        var options = new WorkflowReplayerOptions { Plugins = [plugin] };
        foreach (var workflow in TestWorkflows.All)
        {
            options.AddWorkflow(workflow);
        }

        var replayer = new WorkflowReplayer(options);
        foreach (var file in files)
        {
            var history = WorkflowHistory.FromJson(Path.GetFileNameWithoutExtension(file), await File.ReadAllTextAsync(file, TestContext.Current.CancellationToken));
            var result = await replayer.ReplayWorkflowAsync(history, throwOnReplayFailure: false, TestContext.Current.CancellationToken);
            Assert.True(result.ReplayFailure is null, $"{Path.GetFileName(file)}: {result.ReplayFailure}");
        }
    }
}
