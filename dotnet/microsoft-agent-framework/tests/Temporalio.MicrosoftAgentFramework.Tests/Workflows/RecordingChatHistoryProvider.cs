using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>
/// A history provider that is not the in-memory default; it keeps messages on the instance and
/// records the failures the agent reports to it.
/// </summary>
public sealed class RecordingChatHistoryProvider : ChatHistoryProvider
{
    private readonly List<ChatMessage> messages = [];
    private readonly List<string> failures = [];

    /// <summary>Gets the type names of the exceptions <c>InvokedAsync</c> reported, oldest first.</summary>
    public IReadOnlyList<string> Failures => failures;

    /// <summary>Gets the number of messages stored so far.</summary>
    public int MessageCount => messages.Count;

    protected override ValueTask<IEnumerable<ChatMessage>> ProvideChatHistoryAsync(InvokingContext context, CancellationToken cancellationToken = default) =>
        new(messages.ToList());

    protected override ValueTask InvokedCoreAsync(InvokedContext context, CancellationToken cancellationToken = default)
    {
        if (context.InvokeException is { } exception)
        {
            failures.Add(exception.GetType().Name);
        }

        return base.InvokedCoreAsync(context, cancellationToken);
    }

    protected override ValueTask StoreChatHistoryAsync(InvokedContext context, CancellationToken cancellationToken = default)
    {
        messages.AddRange(context.RequestMessages);
        messages.AddRange(context.ResponseMessages ?? []);
        return default;
    }
}
