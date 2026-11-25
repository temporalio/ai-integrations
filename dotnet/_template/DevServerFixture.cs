// Copy this fixture into a new plugin's test project. The test launcher supplies
// the shared dev server address. Replace the namespace and worker registrations.
//
// [assembly: AssemblyFixture(typeof(DevServerFixture))]
using Temporalio.Client;
using Temporalio.Worker;
using Xunit;

namespace Temporalio.Template.Tests;

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

    // Creates a worker on a fresh task queue; register the plugin's Workflows and Activities here.
    public TemporalWorker NewWorker(Action<TemporalWorkerOptions>? configure = null)
    {
        var options = new TemporalWorkerOptions($"template-{Guid.NewGuid():N}");
        configure?.Invoke(options);
        return new TemporalWorker(Client, options);
    }

    // Options with a run timeout, so a stalled Workflow fails the test instead of hanging it.
    public WorkflowOptions WorkflowOptions(TemporalWorker worker, string prefix, TimeSpan? runTimeout = null) =>
        new(id: $"{prefix}-{Guid.NewGuid():N}", taskQueue: worker.Options.TaskQueue!)
        {
            RunTimeout = runTimeout ?? TimeSpan.FromMinutes(2),
        };
}
