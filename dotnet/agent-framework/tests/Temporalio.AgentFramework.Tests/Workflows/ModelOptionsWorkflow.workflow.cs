using Temporalio.Workflows;

namespace Temporalio.AgentFramework.Tests.Workflows;

[Workflow]
public class ModelOptionsWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(ModelOptionsInput input)
    {
        ActivityOptions? modelOptions = null;
        if (input.StartToCloseSeconds is not null || input.Summary is not null || input.HeartbeatSeconds is not null)
        {
            modelOptions = new ActivityOptions
            {
                StartToCloseTimeout = input.StartToCloseSeconds is { } seconds ? TimeSpan.FromSeconds(seconds) : null,
                HeartbeatTimeout = input.HeartbeatSeconds is { } heartbeat ? TimeSpan.FromSeconds(heartbeat) : null,
                Summary = input.Summary,
            };
        }

        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions { ModelActivityOptions = modelOptions });
        return (await agent.RunAsync(input.Prompt)).Text;
    }
}
