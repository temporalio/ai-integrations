using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// Applies the <c>FunctionInvokingChatClient</c> approval contract to the messages of a run:
/// answered <see cref="ToolApprovalRequestContent"/>s become the tool calls and tool results
/// the provider expects, rejected ones get a rejection result, and the approval contents never
/// reach the provider. A response is bound to the request the session surfaced, as Agent
/// Framework's default middleware binds it: the call that runs is a copy of the one recorded
/// in the session under the response's request id, never the one the response carries. Nothing
/// held by the session is changed; a run that fails leaves it as it was.
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
    /// <param name="fromHistory">
    /// The messages the chat history provider supplied. Only a request found in one of them, or
    /// in <paramref name="held"/>, can authorize a call; a request that appears only in the run's
    /// input proves nothing about what a human was asked.
    /// </param>
    /// <param name="held">Requests the previous run held back in the session, each with the decision this run makes for it.</param>
    /// <returns>What was reconstructed and which approved calls still need to run.</returns>
    public static ApprovalOutcome Process(List<ChatMessage> messages, IReadOnlySet<ChatMessage> fromHistory, IReadOnlyList<HeldApproval> held)
    {
        // Requests the session surfaced, by request id: the message they sit in (null for a held one) and the call.
        var surfaced = new Dictionary<string, (ChatMessage? Message, FunctionCallContent Call)>(StringComparer.Ordinal);

        // Every request in the messages, surfaced or not, with its call id: each needs an answer or a result.
        var allRequests = new Dictionary<string, string>(StringComparer.Ordinal);
        var responses = new List<ToolApprovalResponseContent>();
        var executedCallIds = new HashSet<string>(StringComparer.Ordinal);
        var lastApprovalIndex = -1;
        var lastApprovalHasOtherContent = false;

        for (var i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            var hasApproval = false;
            var hasOther = false;
            foreach (var content in message.Contents)
            {
                switch (content)
                {
                    case ToolApprovalRequestContent { ToolCall: FunctionCallContent call } request:
                        allRequests.TryAdd(request.RequestId, call.CallId);
                        if (IsFromHistory(message, fromHistory))
                        {
                            surfaced.TryAdd(request.RequestId, (message, call));
                        }

                        hasApproval = true;
                        break;
                    case ToolApprovalResponseContent { ToolCall: FunctionCallContent } response:
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

        foreach (var entry in held)
        {
            if (entry.Request.ToolCall is FunctionCallContent call)
            {
                allRequests.TryAdd(entry.Request.RequestId, call.CallId);
                surfaced.TryAdd(entry.Request.RequestId, (null, call));
                responses.Add(entry.Request.CreateResponse(entry.Approved));
            }
        }

        if (responses.Count == 0 && allRequests.Count == 0)
        {
            return ApprovalOutcome.Empty;
        }

        // Bind each response to the surfaced request with its id. The response's own ToolCall is
        // read for nothing but its call id, and only to recognize a call that already has a result.
        var answered = new HashSet<string>(StringComparer.Ordinal);
        var unknown = new List<string>();
        var decisions = new List<Decision>();
        foreach (var response in responses)
        {
            if (surfaced.TryGetValue(response.RequestId, out var request))
            {
                if (answered.Add(response.RequestId) && !executedCallIds.Contains(request.Call.CallId))
                {
                    decisions.Add(new Decision(response, Snapshot(request.Call), request.Message));
                }
            }
            else if (response.ToolCall is FunctionCallContent { CallId: var callId } && executedCallIds.Contains(callId))
            {
                // Settled history, for example a session written by ChatClientAgent: the result is there and nothing runs.
                answered.Add(response.RequestId);
            }
            else if (!unknown.Contains(response.RequestId))
            {
                unknown.Add(response.RequestId);
            }
        }

        var unanswered = allRequests
            .Where(pair => !answered.Contains(pair.Key) && !executedCallIds.Contains(pair.Value) && !unknown.Contains(pair.Key))
            .Select(pair => pair.Key)
            .ToList();
        if (unknown.Count > 0 || unanswered.Count > 0)
        {
            throw new MicrosoftAgentFrameworkWorkflowException(Describe(unknown, unanswered));
        }

        var trailing = lastApprovalIndex < 0 ? 0 : messages.Count - (lastApprovalIndex + 1) + (lastApprovalHasOtherContent ? 1 : 0);
        RemoveApprovalContents(messages);
        var insertIndex = messages.Count - trailing;

        var rejected = decisions.Where(decision => !decision.Response.Approved).ToList();
        var approved = decisions.Where(decision => decision.Response.Approved).ToList();

        // Rebuild the assistant tool-call messages: a response whose request message is in the
        // history gets a clone of that message; a held call joins the message being built, as
        // FunctionInvokingChatClient does. Rejections come first, as there.
        var reconstructed = new List<ChatMessage>();
        var callMessages = new Dictionary<ChatMessage, ChatMessage>(ReferenceEqualityComparer.Instance);
        ChatMessage? current = null;
        foreach (var decision in rejected.Concat(approved))
        {
            if (decision.RequestMessage is null)
            {
                if (current is null)
                {
                    current = new ChatMessage { Role = ChatRole.Assistant, Contents = [] };
                    reconstructed.Add(current);
                }
            }
            else if (!callMessages.TryGetValue(decision.RequestMessage, out current))
            {
                current = decision.RequestMessage.Clone();
                current.Contents = [];
                callMessages[decision.RequestMessage] = current;
                reconstructed.Add(current);
            }

            current.Contents.Add(decision.Call);
        }

        if (rejected.Count > 0)
        {
            var results = new List<AIContent>(rejected.Count);
            foreach (var decision in rejected)
            {
                var text = string.IsNullOrWhiteSpace(decision.Response.Reason) ? RejectedResultText : $"{RejectedResultText} {decision.Response.Reason}";
                decision.Call.InformationalOnly = true;
                results.Add(new FunctionResultContent(decision.Call.CallId, text));
            }

            reconstructed.Add(new ChatMessage(ChatRole.Tool, results));
        }

        messages.InsertRange(insertIndex, reconstructed);
        return new ApprovalOutcome(reconstructed, approved.Select(decision => decision.Call).ToList(), insertIndex + reconstructed.Count);
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

    private static bool IsFromHistory(ChatMessage message, IReadOnlySet<ChatMessage> fromHistory) =>
        fromHistory.Contains(message) || message.GetAgentRequestMessageSourceType() == AgentRequestMessageSourceType.ChatHistory;

    private static FunctionCallContent Snapshot(FunctionCallContent call) =>
        new(call.CallId, call.Name, call.Arguments is null ? null : new Dictionary<string, object?>(call.Arguments))
        {
            AdditionalProperties = call.AdditionalProperties is null ? null : new(call.AdditionalProperties),
            RawRepresentation = call.RawRepresentation,
            Annotations = call.Annotations is null ? null : [.. call.Annotations],
        };

    private static string Describe(List<string> unknown, List<string> unanswered)
    {
        var parts = new List<string>(2);
        if (unknown.Count > 0)
        {
            parts.Add(
                $"ToolApprovalResponseContent with request id(s) '{string.Join(", ", unknown)}' matches no ToolApprovalRequestContent this session surfaced. " +
                "A response is bound to the request an earlier run of the same session returned; answer that request with request.CreateResponse(approved).");
        }

        if (unanswered.Count > 0)
        {
            parts.Add(
                $"ToolApprovalRequestContent with request id(s) '{string.Join(", ", unanswered)}' has no matching ToolApprovalResponseContent. " +
                "Answer every pending request with request.CreateResponse(approved) before the next run.");
        }

        return string.Join(" ", parts);
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
        content is ToolApprovalRequestContent { ToolCall: FunctionCallContent } or ToolApprovalResponseContent { ToolCall: FunctionCallContent };

    /// <summary>A bound decision: the response, the copy of the surfaced call it applies to, and the message that call came from.</summary>
    private sealed record Decision(ToolApprovalResponseContent Response, FunctionCallContent Call, ChatMessage? RequestMessage);
}
