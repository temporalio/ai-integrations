using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

// Both Activities wait for the cancellation to be confirmed, so history records ActivityTaskCanceled
// before WorkflowExecutionCanceled; with the default TryCancel the Workflow closes first.
[Workflow]
public class CancellationWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var toolOptions = new ActivityOptions
        {
            StartToCloseTimeout = TimeSpan.FromSeconds(60),
            HeartbeatTimeout = TimeSpan.FromSeconds(2),
            CancellationType = ActivityCancellationType.WaitCancellationCompleted,
        };
        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions
        {
            Name = "assistant",
            ChatOptions = new ChatOptions
            {
                Instructions = "Be brief.",
                Tools = [ActivityAsTool.Create((WeatherActivities a) => a.SlowWeatherAsync(default!), toolOptions, "get_weather")],
            },
            ModelActivityOptions = new ActivityOptions
            {
                StartToCloseTimeout = TimeSpan.FromSeconds(60),
                CancellationType = ActivityCancellationType.WaitCancellationCompleted,
            },
        });
        var response = await agent.RunAsync(prompt);
        return response.Text;
    }
}
