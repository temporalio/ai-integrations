using System.Text.RegularExpressions;
using Temporalio.Client;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Testing;
using Temporalio.Worker;

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>
/// One local dev server for the whole run. Startup follows the repository policy: three
/// attempts, one second apart, only for the SDK's startup timeout, download and archive errors.
/// </summary>
public sealed partial class DevServerFixture : IAsyncLifetime
{
    // The three startup failures as Temporal's core words them, anchored to the start of the
    // message after an optional "<context>: " prefix. A client error that merely mentions a timeout
    // is not one of them. dotnet/_template/DevServerFixture.cs carries the same pattern.
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

    internal static bool IsRetryable(string message) => RetryableStartupError().IsMatch(message);
}
