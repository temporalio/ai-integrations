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
    /// <summary>Prefix of the approval request ids, the one <c>FunctionInvokingChatClient</c> composes.</summary>
    public const string RequestIdPrefix = "ficc_";

    /// <summary>The plugin's result text for a rejected call; the reason, when given, follows it.</summary>
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

        // Rebuild the assistant tool-call messages: a response whose request message is in the
        // history gets a clone of that message; a response without one (an auto-approved call
        // carried in the session) joins the message being built, as FunctionInvokingChatClient does.
        var reconstructed = new List<ChatMessage>();
        var callMessages = new Dictionary<ChatMessage, ChatMessage>(ReferenceEqualityComparer.Instance);
        ChatMessage? current = null;
        foreach (var response in rejected.Concat(approved))
        {
            var requestMessage = requests.TryGetValue(response.RequestId, out var found) ? found.Message : null;
            if (requestMessage is null)
            {
                if (current is null)
                {
                    current = new ChatMessage { Role = ChatRole.Assistant, Contents = [] };
                    reconstructed.Add(current);
                }
            }
            else if (!callMessages.TryGetValue(requestMessage, out current))
            {
                current = requestMessage.Clone();
                current.Contents = [];
                callMessages[requestMessage] = current;
                reconstructed.Add(current);
            }

            current.Contents.Add(response.ToolCall);
        }

        if (rejected.Count > 0)
        {
            var results = new List<AIContent>(rejected.Count);
            foreach (var response in rejected)
            {
                var call = (FunctionCallContent)response.ToolCall;
                var text = string.IsNullOrWhiteSpace(response.Reason) ? RejectedResultText : $"{RejectedResultText} {response.Reason}";
                call.InformationalOnly = true;
                MarkRequestInformational(requests, response.RequestId);
                results.Add(new FunctionResultContent(call.CallId, text));
            }

            reconstructed.Add(new ChatMessage(ChatRole.Tool, results));
        }

        messages.InsertRange(insertIndex, reconstructed);
        var approvedCalls = approved.Select(response => (FunctionCallContent)response.ToolCall).ToList();
        foreach (var response in approved)
        {
            MarkRequestInformational(requests, response.RequestId);
        }

        return new ApprovalOutcome(reconstructed, approvedCalls, insertIndex + reconstructed.Count);
    }

    /// <summary>
    /// When any call in <paramref name="response"/> targets an approval-required tool, replaces
    /// every pending call: calls that need a decision become <see cref="ToolApprovalRequestContent"/>s
    /// in the response, the others are removed from it and returned in <paramref name="deferred"/>
    /// so the next run can approve them itself. Nothing runs until the human has answered, which is
    /// what <c>FunctionInvokingChatClient</c> with Agent Framework's default middleware does.
    /// </summary>
    /// <param name="response">The model response; changed messages are cloned.</param>
    /// <param name="approvalRequiredNames">Names of the tools that need approval.</param>
    /// <param name="deferred">Receives the requests for the calls that need no decision.</param>
    /// <returns>True when the response now carries approval requests.</returns>
    public static bool ReplaceWithApprovalRequests(ChatResponse response, HashSet<string> approvalRequiredNames, List<ToolApprovalRequestContent> deferred)
    {
        var pending = response.Messages.SelectMany(message => message.Contents).OfType<FunctionCallContent>().Where(call => !call.InformationalOnly).ToList();
        if (!pending.Exists(call => approvalRequiredNames.Contains(call.Name)))
        {
            return false;
        }

        for (var i = response.Messages.Count - 1; i >= 0; i--)
        {
            var message = response.Messages[i];
            if (!message.Contents.OfType<FunctionCallContent>().Any(call => !call.InformationalOnly))
            {
                continue;
            }

            var contents = new List<AIContent>(message.Contents.Count);
            foreach (var content in message.Contents)
            {
                if (content is FunctionCallContent { InformationalOnly: false } call)
                {
                    var request = new ToolApprovalRequestContent(RequestIdPrefix + call.CallId, call);
                    if (approvalRequiredNames.Contains(call.Name))
                    {
                        contents.Add(request);
                    }
                    else
                    {
                        deferred.Add(request);
                    }
                }
                else
                {
                    contents.Add(content);
                }
            }

            if (contents.Count == 0)
            {
                response.Messages.RemoveAt(i);
                continue;
            }

            var clone = message.Clone();
            clone.Contents = contents;
            response.Messages[i] = clone;
        }

        return true;
    }

    private static void MarkRequestInformational(Dictionary<string, (ChatMessage Message, ToolApprovalRequestContent Request)> requests, string requestId)
    {
        if (requests.TryGetValue(requestId, out var found) && found.Request.ToolCall is FunctionCallContent call)
        {
            call.InformationalOnly = true;
        }
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
