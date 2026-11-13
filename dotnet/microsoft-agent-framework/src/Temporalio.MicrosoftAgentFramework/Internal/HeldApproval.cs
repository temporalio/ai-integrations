using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// A request the previous run held back in the session because the same response also carried
/// calls that needed a human, with the decision this run makes for it.
/// </summary>
/// <param name="Request">The held request; the call that runs is a copy of its tool call.</param>
/// <param name="Approved">True to run the call; false when the tool is gone or now needs approval.</param>
internal sealed record HeldApproval(ToolApprovalRequestContent Request, bool Approved);
