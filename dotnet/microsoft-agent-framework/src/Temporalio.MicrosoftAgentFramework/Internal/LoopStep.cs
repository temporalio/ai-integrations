using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// One step of <see cref="AgentLoop"/>: either a model response or messages the loop produced
/// itself (tool results, reconstructed tool calls).
/// </summary>
/// <param name="Response">The model response, or null for loop-produced messages.</param>
/// <param name="Messages">The messages of this step.</param>
internal sealed record LoopStep(ChatResponse? Response, IReadOnlyList<ChatMessage> Messages);
