using Temporalio.Workflows;

namespace Temporalio.AgentFramework.Tests.Workflows;

[Workflow]
public class TwoAgentsWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var alpha = new TemporalAgent("alpha", new TemporalAgentOptions { Name = "alpha" });
        var beta = new TemporalAgent("beta", new TemporalAgentOptions { Name = "beta" });
        var first = await alpha.RunAsync(prompt);
        var second = await beta.RunAsync(prompt);
        return $"{first.Text}|{second.Text}";
    }
}
