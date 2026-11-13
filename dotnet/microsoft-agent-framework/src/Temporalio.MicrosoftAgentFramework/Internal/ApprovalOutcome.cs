using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// Result of <see cref="ApprovalProcessor.Process"/>.
/// </summary>
/// <param name="ReconstructedMessages">
/// Messages inserted into the provider view and returned to the caller: the tool-call messages
/// rebuilt from the answered requests and the rejection results.
/// </param>
/// <param name="ApprovedCalls">Approved calls that have not run yet.</param>
/// <param name="ResultInsertIndex">Where the results of <paramref name="ApprovedCalls"/> go in the provider view.</param>
internal sealed record ApprovalOutcome(
    IReadOnlyList<ChatMessage> ReconstructedMessages,
    IReadOnlyList<FunctionCallContent> ApprovedCalls,
    int ResultInsertIndex)
{
    /// <summary>Gets the outcome for a run without approval contents.</summary>
    public static ApprovalOutcome Empty { get; } = new([], [], 0);
}
