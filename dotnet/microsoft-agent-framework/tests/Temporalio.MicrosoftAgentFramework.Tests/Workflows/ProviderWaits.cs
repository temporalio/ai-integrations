using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>The waits a provider must not make inside the Workflow, by name.</summary>
public static class ProviderWaits
{
    /// <summary>Awaits <c>Task.Delay</c>: a .NET timer, off the Workflow scheduler.</summary>
    public const string TaskDelay = "task";

    /// <summary>Awaits <c>Workflow.DelayAsync</c>: a Workflow timer the provider cannot wait for.</summary>
    public const string WorkflowTimer = "timer";

    /// <summary>Awaits an Activity the provider cannot wait for.</summary>
    public const string WorkflowActivity = "activity";

    /// <summary>Blocks on <c>Task.Run(...).Result</c> and returns a completed task: sync-over-async off the scheduler.</summary>
    public const string ThreadHop = "thread";

    public static Task WaitAsync(string kind, CancellationToken cancellationToken)
    {
        switch (kind)
        {
            case WorkflowTimer:
                return Workflow.DelayAsync(TimeSpan.FromMilliseconds(10));
            case WorkflowActivity:
                return Workflow.ExecuteActivityAsync(
                    () => StaticActivities.LookupTimeZone("Kingston"),
                    new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(10) });
            case ThreadHop:
#pragma warning disable CA1849 // The blocking wait on thread-pool work is the point of this case.
                _ = Task.Run(() => Environment.TickCount64).Result;
#pragma warning restore CA1849
                return Task.CompletedTask;
            default:
                return Task.Delay(50, cancellationToken);
        }
    }
}
