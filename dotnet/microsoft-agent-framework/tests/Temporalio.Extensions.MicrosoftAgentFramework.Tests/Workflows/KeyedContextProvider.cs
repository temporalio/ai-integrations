using Microsoft.Agents.AI;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>A context provider that declares the session state key it is given.</summary>
public sealed class KeyedContextProvider : AIContextProvider
{
    private readonly string key;

    public KeyedContextProvider(string key) => this.key = key;

    public override IReadOnlyList<string> StateKeys => [key];

    protected override ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default) =>
        new(new AIContext());
}
