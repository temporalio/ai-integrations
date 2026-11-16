using System.Linq.Expressions;
using Temporalio.Client;
using Temporalio.Exceptions;
using Temporalio.Worker;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

public abstract class WorkflowTestBase
{
    protected WorkflowTestBase(DevServerFixture fixture) => Fixture = fixture;

    protected DevServerFixture Fixture { get; }

    protected ITemporalClient Client => Fixture.Client;

    protected static ApplicationFailureException? AppFailure(Exception? exception)
    {
        for (var current = exception; current is not null; current = current.InnerException)
        {
            if (current is ApplicationFailureException failure)
            {
                return failure;
            }
        }

        return null;
    }

    // Longest a single test may keep a worker running before it is stopped and the test fails.
    protected static readonly TimeSpan WorkerBudget = TimeSpan.FromMinutes(4);

    // Runs the worker until the body completes, stopping it when the test is cancelled or WorkerBudget runs out.
    protected static async Task WithWorkerAsync(TemporalWorker worker, Func<Task> body)
    {
        using var budget = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        budget.CancelAfter(WorkerBudget);
        await worker.ExecuteAsync(body, budget.Token);
    }

    protected async Task<(TResult Result, WorkflowHandle Handle)> RunAsync<TWorkflow, TResult>(
        MicrosoftAgentFrameworkPlugin? plugin,
        Expression<Func<TWorkflow, Task<TResult>>> call,
        Action<TemporalWorkerOptions>? configure = null,
        ITemporalClient? client = null,
        TimeSpan? runTimeout = null)
    {
        using var worker = Fixture.NewWorker(plugin, configure, client);
        (TResult Result, WorkflowHandle Handle) outcome = default;
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await (client ?? Client).StartWorkflowAsync(call, Fixture.WorkflowOptions(worker, typeof(TWorkflow).Name, runTimeout));
            var result = await handle.GetResultAsync();
            outcome = (result, handle);
        });
        return outcome;
    }

    protected async Task<(WorkflowFailedException Error, WorkflowHandle Handle)> RunFailingAsync<TWorkflow, TResult>(
        MicrosoftAgentFrameworkPlugin? plugin,
        Expression<Func<TWorkflow, Task<TResult>>> call,
        Action<TemporalWorkerOptions>? configure = null)
    {
        using var worker = Fixture.NewWorker(plugin, configure);
        (WorkflowFailedException Error, WorkflowHandle Handle) outcome = default;
        await WithWorkerAsync(worker, async () =>
        {
            var handle = await Client.StartWorkflowAsync(call, Fixture.WorkflowOptions(worker, typeof(TWorkflow).Name));
            var error = await Assert.ThrowsAsync<WorkflowFailedException>(() => handle.GetResultAsync());
            outcome = (error, handle);
        });
        return outcome;
    }
}
