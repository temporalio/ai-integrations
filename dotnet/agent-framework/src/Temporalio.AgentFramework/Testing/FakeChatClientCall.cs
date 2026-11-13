using Microsoft.Extensions.AI;

namespace Temporalio.AgentFramework.Testing;

/// <summary>
/// One call a <see cref="FakeChatClient"/> received.
/// </summary>
/// <param name="Messages">The messages, copied at call time.</param>
/// <param name="Options">The options, cloned at call time; tools are declaration-only on the worker.</param>
public sealed record FakeChatClientCall(IReadOnlyList<ChatMessage> Messages, ChatOptions? Options);
