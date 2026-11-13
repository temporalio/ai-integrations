using Microsoft.Extensions.AI;

namespace Temporalio.AgentFramework.Internal;

/// <summary>
/// Applies the <c>FunctionInvokingChatClient</c> approval contract to the messages of a run:
/// answered <see cref="ToolApprovalRequestContent"/>s become the tool calls and tool results
/// the provider expects, rejected ones get a rejection result, and the approval contents never
/// reach the provider.
/// </summary>
internal static class ApprovalProcessor
{
    /// <summary>Prefix of the approval request ids, the one <c>FunctionInvokingChatClient</c> uses.</summary>
    public const string RequestIdPrefix = "ficc_";

    /// <summary>Result text for a rejected call, the one <c>FunctionInvokingChatClient</c> uses.</summary>
    public const string RejectedResultText = "Tool call invocation was rejected by the approver and must not be retried.";

    /// <summary>
    /// Rewrites <paramref name="messages"/> in place.
    /// </summary>
    /// <param name="messages">History plus the run's input, as the provider would see it.</param>
    /// <returns>What was reconstructed and which approved calls still need to run.</returns>
    public static ApprovalOutcome Process(List<ChatMessage> messages)
    {
        var requests = new Dictionary<string, (ChatMessage Message, ToolApprovalRequestContent Request)>(StringComparer.Ordinal);
        var responses = new List<ToolApprovalResponseContent>();
        var unansweredCallIds = new HashSet<string>(StringComparer.Ordinal);
        var executedCallIds = new HashSet<string>(StringComparer.Ordinal);
        var lastApprovalIndex = -1;
        var lastApprovalHasOtherContent = false;

        for (var i = 0; i < messages.Count; i++)
        {
            var hasApproval = false;
            var hasOther = false;
            foreach (var content in messages[i].Contents)
            {
                switch (content)
                {
                    case ToolApprovalRequestContent { ToolCall: FunctionCallContent { InformationalOnly: false } call } request:
                        unansweredCallIds.Add(call.CallId);
                        requests[request.RequestId] = (messages[i], request);
                        hasApproval = true;
                        break;
                    case ToolApprovalResponseContent { ToolCall: FunctionCallContent { InformationalOnly: false } call } response:
                        unansweredCallIds.Remove(call.CallId);
                        responses.Add(response);
                        hasApproval = true;
                        break;
                    case FunctionResultContent result:
                        executedCallIds.Add(result.CallId);
                        hasOther = true;
                        break;
                    default:
                        hasOther = true;
                        break;
                }
            }

            if (hasApproval)
            {
                lastApprovalIndex = i;
                lastApprovalHasOtherContent = hasOther;
            }
        }

        if (responses.Count == 0 && requests.Count == 0)
        {
            return ApprovalOutcome.Empty;
        }

        if (unansweredCallIds.Count > 0)
        {
            throw new AgentFrameworkWorkflowException(
                $"ToolApprovalRequestContent found with FunctionCall.CallId(s) '{string.Join(", ", unansweredCallIds)}' that have no matching ToolApprovalResponseContent. " +
                "Answer every pending request with request.CreateResponse(approved) before the next run.");
        }

        var trailing = lastApprovalIndex < 0 ? 0 : messages.Count - (lastApprovalIndex + 1) + (lastApprovalHasOtherContent ? 1 : 0);
        RemoveApprovalContents(messages);
        var insertIndex = messages.Count - trailing;

        var rejected = new List<ToolApprovalResponseContent>();
        var approved = new List<ToolApprovalResponseContent>();
        foreach (var response in responses)
        {
            if (executedCallIds.Contains(response.ToolCall.CallId))
            {
                continue;
            }

            (response.Approved ? approved : rejected).Add(response);
        }

        var reconstructed = new List<ChatMessage>();
        var callMessages = new Dictionary<ChatMessage, ChatMessage>(ReferenceEqualityComparer.Instance);
        foreach (var response in rejected.Concat(approved))
        {
            var requestMessage = requests.TryGetValue(response.RequestId, out var found) ? found.Message : null;
            ChatMessage? target = null;
            if (requestMessage is not null && callMessages.TryGetValue(requestMessage, out var existing))
            {
                target = existing;
            }

            if (target is null)
            {
                target = requestMessage?.Clone() ?? new ChatMessage { Role = ChatRole.Assistant };
                target.Contents = [];
                reconstructed.Add(target);
                if (requestMessage is not null)
                {
                    callMessages[requestMessage] = target;
                }
            }

            target.Contents.Add(response.ToolCall);
        }

        if (rejected.Count > 0)
        {
            var results = new List<AIContent>(rejected.Count);
            foreach (var response in rejected)
            {
                var call = (FunctionCallContent)response.ToolCall;
                var text = string.IsNullOrWhiteSpace(response.Reason) ? RejectedResultText : $"{RejectedResultText} {response.Reason}";
                call.InformationalOnly = true;
                if (requests.TryGetValue(response.RequestId, out var found) && found.Request.ToolCall is FunctionCallContent requestCall)
                {
                    requestCall.InformationalOnly = true;
                }

                results.Add(new FunctionResultContent(call.CallId, text));
            }

            reconstructed.Add(new ChatMessage(ChatRole.Tool, results));
        }

        messages.InsertRange(insertIndex, reconstructed);
        var approvedCalls = approved.Select(response => (FunctionCallContent)response.ToolCall).ToList();
        foreach (var response in approved)
        {
            if (requests.TryGetValue(response.RequestId, out var found) && found.Request.ToolCall is FunctionCallContent requestCall)
            {
                requestCall.InformationalOnly = true;
            }
        }

        return new ApprovalOutcome(reconstructed, approvedCalls, insertIndex + reconstructed.Count);
    }

    /// <summary>
    /// Replaces the calls to approval-required tools in <paramref name="response"/> with
    /// <see cref="ToolApprovalRequestContent"/>s.
    /// </summary>
    /// <param name="response">The model response; its messages are cloned when changed.</param>
    /// <param name="approvalRequiredNames">Names of the tools that need approval.</param>
    /// <returns>True when at least one call was replaced.</returns>
    public static bool ReplaceWithApprovalRequests(ChatResponse response, HashSet<string> approvalRequiredNames)
    {
        var replaced = false;
        for (var i = 0; i < response.Messages.Count; i++)
        {
            var message = response.Messages[i];
            List<AIContent>? contents = null;
            for (var j = 0; j < message.Contents.Count; j++)
            {
                if (message.Contents[j] is FunctionCallContent { InformationalOnly: false } call && approvalRequiredNames.Contains(call.Name))
                {
                    contents ??= [.. message.Contents];
                    contents[j] = new ToolApprovalRequestContent(RequestIdPrefix + call.CallId, call);
                    replaced = true;
                }
            }

            if (contents is not null)
            {
                var clone = message.Clone();
                clone.Contents = contents;
                response.Messages[i] = clone;
            }
        }

        return replaced;
    }

    private static void RemoveApprovalContents(List<ChatMessage> messages)
    {
        for (var i = messages.Count - 1; i >= 0; i--)
        {
            var message = messages[i];
            if (!message.Contents.Any(IsFunctionApproval))
            {
                continue;
            }

            var kept = message.Contents.Where(content => !IsFunctionApproval(content)).ToList();
            if (kept.Count == 0)
            {
                messages.RemoveAt(i);
            }
            else
            {
                var clone = message.Clone();
                clone.Contents = kept;
                messages[i] = clone;
            }
        }
    }

    private static bool IsFunctionApproval(AIContent content) =>
        content is ToolApprovalRequestContent { ToolCall: FunctionCallContent { InformationalOnly: false } }
            or ToolApprovalResponseContent { ToolCall: FunctionCallContent { InformationalOnly: false } };
}
