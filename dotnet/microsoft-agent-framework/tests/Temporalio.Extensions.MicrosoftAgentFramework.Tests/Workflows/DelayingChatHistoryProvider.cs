using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>A history provider that waits before answering: a timer, a Workflow timer or an Activity.</summary>
public sealed class DelayingChatHistoryProvider : ChatHistoryProvider
{
    private readonly string wait;

    public DelayingChatHistoryProvider(string wait = ProviderWaits.TaskDelay) => this.wait = wait;

    protected override async ValueTask<IEnumerable<ChatMessage>> ProvideChatHistoryAsync(InvokingContext context, CancellationToken cancellationToken = default)
    {
        await ProviderWaits.WaitAsync(wait, cancellationToken);
        return [];
    }
}
