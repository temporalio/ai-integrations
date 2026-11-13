// Copy this file into a new plugin's test project (and add the AssemblyFixture attribute shown
// below) to get the repository's dev server startup policy: three attempts, one second apart,
// only for the Temporal SDK's own startup timeout, download and archive failures, matched by an
// anchored pattern. Replace the namespace and the worker registration to fit the plugin.
//
// [assembly: AssemblyFixture(typeof(DevServerFixture))]
using System.Text.RegularExpressions;
using Temporalio.Client;
using Temporalio.Testing;
using Temporalio.Worker;
using Xunit;

namespace Temporalio.Template.Tests;

/// <summary>One local dev server for the whole run, started under the repository's retry policy.</summary>
public sealed partial class DevServerFixture : IAsyncLifetime
{
    // The three startup failures as Temporal's core words them, anchored to the start of the
    // message after an optional "<context>: " prefix. A client error that merely mentions a timeout
    // is not one of them.
    [GeneratedRegex(@"^(?:[^:]*: )?(?:ephemeral server at \S+ did not start within |failed to download ephemeral server executable:|invalid ephemeral server download: could not (?:read|extract) )", RegexOptions.CultureInvariant)]
    private static partial Regex RetryableStartupError();

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
            catch (InvalidOperationException exception) when (attempt < 3 && IsRetryable(exception.Message))
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

    internal static bool IsRetryable(string message) => RetryableStartupError().IsMatch(message);
}
