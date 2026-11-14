using Microsoft.Agents.AI;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>A context provider that waits in a way a Workflow cannot allow: a timer, a Workflow timer or an Activity.</summary>
public sealed class DelayingContextProvider : AIContextProvider
{
    private readonly string wait;

    public DelayingContextProvider(string wait = ProviderWaits.TaskDelay) => this.wait = wait;

    protected override async ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default)
    {
        await ProviderWaits.WaitAsync(wait, cancellationToken);
        return new AIContext();
    }
}
