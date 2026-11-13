using Microsoft.Agents.AI;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>A context provider that awaits a timer, which a Workflow cannot allow.</summary>
public sealed class DelayingContextProvider : AIContextProvider
{
    protected override async ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default)
    {
        await Task.Delay(50, cancellationToken);
        return new AIContext();
    }
}
