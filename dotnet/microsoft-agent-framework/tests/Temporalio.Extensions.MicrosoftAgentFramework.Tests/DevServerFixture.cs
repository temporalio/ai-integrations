using Temporalio.Client;
using Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Worker;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>
/// Connect to the server shared by the integration tests.
/// </summary>
public sealed class DevServerFixture : IAsyncLifetime
{
    private ITemporalClient? client;

    public ITemporalClient Client => client ?? throw new InvalidOperationException(
        "Run live Temporal tests with `make test` or set TEMPORAL_TEST_ADDRESS to a running server.");

    public async ValueTask InitializeAsync()
    {
        if (Environment.GetEnvironmentVariable("TEMPORAL_TEST_ADDRESS") is { Length: > 0 } address)
        {
            client = await TemporalClient.ConnectAsync(new(address));
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    // Creates a worker on a fresh task queue with every test Workflow, the test Activities and the given plugin.
    public TemporalWorker NewWorker(MicrosoftAgentFrameworkPlugin? plugin, Action<TemporalWorkerOptions>? configure = null, ITemporalClient? client = null)
    {
        var options = new TemporalWorkerOptions($"microsoft-agent-framework-{Guid.NewGuid():N}")
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
}
