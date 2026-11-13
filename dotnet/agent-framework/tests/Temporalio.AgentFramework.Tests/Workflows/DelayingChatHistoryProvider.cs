using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace Temporalio.AgentFramework.Tests.Workflows;

/// <summary>A history provider that awaits a timer before answering.</summary>
public sealed class DelayingChatHistoryProvider : ChatHistoryProvider
{
    protected override async ValueTask<IEnumerable<ChatMessage>> ProvideChatHistoryAsync(InvokingContext context, CancellationToken cancellationToken = default)
    {
        await Task.Delay(50, cancellationToken);
        return [];
    }
}
