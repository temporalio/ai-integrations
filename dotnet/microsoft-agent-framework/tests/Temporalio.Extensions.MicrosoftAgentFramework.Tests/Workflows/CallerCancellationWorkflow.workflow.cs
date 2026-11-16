using Microsoft.Extensions.AI;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

// Passes its own token to RunAsync and cancels it from a Signal, so the cancellation reaches the
// running tool Activity through the plugin's token wiring, not through the Workflow's own token.
[Workflow]
#pragma warning disable CA1001 // A Workflow instance is never disposed; the token source lives as long as the run.
public class CallerCancellationWorkflow
#pragma warning restore CA1001
{
    private readonly CancellationTokenSource cancellation = new();

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
        });
        try
        {
            var response = await agent.RunAsync(prompt, cancellationToken: cancellation.Token);
            return response.Text;
        }
#pragma warning disable CA1031 // The filter keeps it to cancellation; the test reads where it came from.
        catch (Exception exception) when (exception is OperationCanceledException || TemporalException.IsCanceledException(exception))
#pragma warning restore CA1031
        {
            // The tool Activity's own cancellation names the Activity; a cancellation raised elsewhere, such as the
            // next model call seeing the cancelled token, names the exception type instead.
            return $"cancelled: {(exception is ActivityFailureException activity ? activity.ActivityType : exception.GetType().Name)}";
        }
    }

    [WorkflowSignal]
    public Task CancelRunAsync()
    {
#pragma warning disable CA1849 // CancelAsync runs the registered callbacks on the thread pool; Workflow code must stay on its scheduler.
        cancellation.Cancel();
#pragma warning restore CA1849
        return Task.CompletedTask;
    }
}
