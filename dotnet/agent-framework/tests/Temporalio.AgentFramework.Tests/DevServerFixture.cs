using Temporalio.AgentFramework.Tests.Workflows;
using Temporalio.Client;
using Temporalio.Testing;
using Temporalio.Worker;

namespace Temporalio.AgentFramework.Tests;

/// <summary>
/// One local dev server for the whole run. Startup follows the repository policy: three
/// attempts, one second apart, only for the SDK's startup timeout, download and archive errors.
/// </summary>
public sealed class DevServerFixture : IAsyncLifetime
{
    private static readonly string[] RetryableMarkers = ["download", "timed out", "timeout", "extract", "archive"];

    public WorkflowEnvironment Env { get; private set; } = null!;

    public ITemporalClient Client => Env.Client;

    public async ValueTask InitializeAsync()
    {
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                Env = await WorkflowEnvironment.StartLocalAsync();
                return;
            }
            catch (InvalidOperationException exception) when (attempt < 3 && IsRetryable(exception))
            {
                await Task.Delay(TimeSpan.FromSeconds(1));
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Env is not null)
        {
            await Env.ShutdownAsync();
        }
    }

    // Creates a worker on a fresh task queue with every test Workflow, the test Activities and the given plugin.
    public TemporalWorker NewWorker(AgentFrameworkPlugin? plugin, Action<TemporalWorkerOptions>? configure = null, ITemporalClient? client = null)
    {
        var options = new TemporalWorkerOptions($"agent-framework-{Guid.NewGuid():N}")
            .AddAllActivities(new WeatherActivities())
            .AddActivity(StaticActivities.LookupTimeZone);
        foreach (var workflow in TestWorkflows.All)
        {
            options.AddWorkflow(workflow);
        }

        if (plugin is not null)
        {
            options.Plugins = [plugin];
        }

        configure?.Invoke(options);
        return new TemporalWorker(client ?? Client, options);
    }

    // Options with a run timeout, so a stalled Workflow fails the test instead of hanging it. The server
    // caps Activity timeouts at the run timeout, so tests that assert the 5-minute model default pass a longer one.
    public WorkflowOptions WorkflowOptions(TemporalWorker worker, string prefix, TimeSpan? runTimeout = null) =>
        new(id: $"{prefix}-{Guid.NewGuid():N}", taskQueue: worker.Options.TaskQueue!)
        {
            RunTimeout = runTimeout ?? TimeSpan.FromMinutes(2),
        };

    private static bool IsRetryable(Exception exception) =>
        RetryableMarkers.Any(marker => exception.Message.Contains(marker, StringComparison.OrdinalIgnoreCase));
}
