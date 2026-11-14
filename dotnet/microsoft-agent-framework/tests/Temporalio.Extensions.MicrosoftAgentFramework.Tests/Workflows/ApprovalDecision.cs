namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>The Update input of <see cref="ApprovalUpdateWorkflow"/>: a decision the client makes about one request.</summary>
public sealed record ApprovalDecision(string RequestId, bool Approved, string? Reason = null);
