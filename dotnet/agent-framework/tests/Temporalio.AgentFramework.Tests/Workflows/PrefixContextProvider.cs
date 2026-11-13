using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace Temporalio.AgentFramework.Tests.Workflows;

/// <summary>Adds an instruction and a context message to every run.</summary>
public sealed class PrefixContextProvider : AIContextProvider
{
    public const string Instruction = "Always answer in French.";

    public const string ContextMessage = "Context: the user is in Kingston.";

    protected override ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default) =>
        new(new AIContext
        {
            Instructions = Instruction,
            Messages = [new ChatMessage(ChatRole.User, ContextMessage)],
        });
}
