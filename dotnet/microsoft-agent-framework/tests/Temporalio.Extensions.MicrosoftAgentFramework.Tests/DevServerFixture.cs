using System.Text.RegularExpressions;
using Temporalio.Client;
using Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Testing;
using Temporalio.Worker;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>
/// Use the test launcher's server, or start one for a direct dotnet test run.
/// </summary>
public sealed partial class DevServerFixture : IAsyncLifetime
{
    private WorkflowEnvironment? localEnvironment;

    public ITemporalClient Client { get; private set; } = null!;

    public async ValueTask InitializeAsync()
    {
        if (Environment.GetEnvironmentVariable("TEMPORAL_TEST_ADDRESS") is { Length: > 0 } address)
        {
            Client = await TemporalClient.ConnectAsync(new(address));
            return;
        }

        for (var attempt = 1; ; attempt++)
        {
            try
            {
                localEnvironment = await WorkflowEnvironment.StartLocalAsync();
                Client = localEnvironment.Client;
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
        if (localEnvironment is not null)
        {
            await localEnvironment.ShutdownAsync();
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

    // The three startup failures as Temporal's core words them, anchored to the start of the
    // message after an optional "<context>: " prefix. A client error that merely mentions a timeout
    // is not one of them. dotnet/_template/DevServerFixture.cs carries the same pattern.
    [GeneratedRegex(@"^(?:[^:]*: )?(?:ephemeral server at \S+ did not start within |failed to download ephemeral server executable:|invalid ephemeral server download: could not (?:read|extract) )", RegexOptions.CultureInvariant)]
    private static partial Regex RetryableStartupError();
}
