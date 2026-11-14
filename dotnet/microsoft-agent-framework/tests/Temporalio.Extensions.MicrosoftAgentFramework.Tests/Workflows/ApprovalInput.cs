using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed record ApprovalInput(
    string Prompt,
    string ToolKind,
    bool WithPlainTool = false,
    bool RetryAfterModelFailure = false,
    bool ContinueAsNew = false,
    JsonElement? Session = null,
    ToolApprovalRequestContent? PendingRequest = null,
    bool TwoApprovalTools = false,
    bool DecisionWithText = false);
