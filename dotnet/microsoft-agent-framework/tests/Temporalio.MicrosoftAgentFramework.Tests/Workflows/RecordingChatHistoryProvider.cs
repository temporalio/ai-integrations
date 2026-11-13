using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>A history provider that is not the in-memory default; it keeps messages on the instance.</summary>
public sealed class RecordingChatHistoryProvider : ChatHistoryProvider
{
    private readonly List<ChatMessage> messages = [];

    protected override ValueTask<IEnumerable<ChatMessage>> ProvideChatHistoryAsync(InvokingContext context, CancellationToken cancellationToken = default) =>
        new(messages.ToList());

    protected override ValueTask StoreChatHistoryAsync(InvokedContext context, CancellationToken cancellationToken = default)
    {
        messages.AddRange(context.RequestMessages);
        messages.AddRange(context.ResponseMessages ?? []);
        return default;
    }
}
