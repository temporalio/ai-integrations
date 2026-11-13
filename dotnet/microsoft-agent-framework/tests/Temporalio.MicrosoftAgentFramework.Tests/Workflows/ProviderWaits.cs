using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>The waits a provider must not make inside the Workflow, by name.</summary>
public static class ProviderWaits
{
    public const string TaskDelay = "task";

    public const string WorkflowTimer = "timer";

    public const string WorkflowActivity = "activity";

    public static Task WaitAsync(string kind, CancellationToken cancellationToken) => kind switch
    {
        WorkflowTimer => Workflow.DelayAsync(TimeSpan.FromMilliseconds(10)),
        WorkflowActivity => Workflow.ExecuteActivityAsync(
            () => StaticActivities.LookupTimeZone("Kingston"),
            new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(10) }),
        _ => Task.Delay(50, cancellationToken),
    };
}
