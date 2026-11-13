using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// The agent loop <see cref="TemporalAgent"/> runs in the Workflow: history and context
/// providers, approval bookkeeping, one model Activity per round, tool dispatch and the
/// iteration cap, with the same contract as <c>FunctionInvokingChatClient</c>. Every await
/// stays on the Workflow scheduler. The fan-out and merge discipline follows Cecil Phillip's
/// draft at https://github.com/temporal-community/temporal-dotnet-agents.
/// </summary>
internal sealed class AgentLoop
{
    /// <summary>Session state key under which calls approved without a human wait for the next run.</summary>
    internal const string DeferredApprovalsKey = "TemporalAgent.DeferredApprovalRequests";

    private const int MaximumConsecutiveToolErrors = 3;
    private const string ToolFailedText = "Error: Function failed.";
    private const string ToolSucceededText = "Success: Function completed.";

    private readonly TemporalAgent agent;
    private readonly TemporalAgentOptions options;
    private readonly TemporalAgentSession session;
    private readonly IReadOnlyCollection<ChatMessage> input;
    private readonly AgentRunOptions? runOptions;
    private readonly CancellationToken cancellationToken;
    private int consecutiveToolErrors;

    /// <summary>
    /// Initializes a new instance of the <see cref="AgentLoop"/> class.
    /// </summary>
    /// <param name="agent">The agent, for provider contexts and the chat client.</param>
    /// <param name="options">The agent options.</param>
    /// <param name="session">The session this run reads and extends.</param>
    /// <param name="input">The run's new messages.</param>
    /// <param name="runOptions">Per-run options.</param>
    /// <param name="cancellationToken">Cancels Activities.</param>
    public AgentLoop(
        TemporalAgent agent,
        TemporalAgentOptions options,
        TemporalAgentSession session,
        IReadOnlyCollection<ChatMessage> input,
        AgentRunOptions? runOptions,
        CancellationToken cancellationToken)
    {
        this.agent = agent;
        this.options = options;
        this.session = session;
        this.input = input;
        this.runOptions = runOptions;
        this.cancellationToken = cancellationToken;
    }

    /// <summary>Gets the run result once <see cref="RunAsync"/> has been enumerated to the end.</summary>
    public AgentResponse? Result { get; private set; }

    /// <summary>
    /// Runs the loop. Enumerate to the end; <see cref="Result"/> is set when the enumeration
    /// completes and the providers have been told about the new messages.
    /// </summary>
    /// <param name="enumerationToken">Cancellation token supplied by the enumerator.</param>
    /// <returns>The steps.</returns>
    public async IAsyncEnumerable<LoopStep> RunAsync([EnumeratorCancellation] CancellationToken enumerationToken = default)
    {
        if (!Workflow.InWorkflow)
        {
            throw new InvalidOperationException(
                "TemporalAgent runs its model calls as Temporal Activities and must be called inside a Workflow. " +
                "Outside a Workflow, use ChatClientAgent over the provider client your MicrosoftAgentFrameworkPluginOptions.ChatClients factory builds.");
        }

        var token = cancellationToken.CanBeCanceled ? cancellationToken : enumerationToken;
        var chatOptions = MergeChatOptions();
        var messages = LoadHistory();

        // Only a request in a message the history provider supplied can authorize an approved call.
        // The provider's base class stamps those messages; this set covers a provider that does not.
        var inputMessages = new HashSet<ChatMessage>(input, ReferenceEqualityComparer.Instance);
        var fromHistory = new HashSet<ChatMessage>(messages.Where(message => !inputMessages.Contains(message)), ReferenceEqualityComparer.Instance);
        chatOptions = ApplyContextProviders(chatOptions, ref messages);

        // The providers hear about this list at the end of the run, as they do under ChatClientAgent:
        // history, the input and anything the context providers added. Their base classes filter it.
        var requestMessages = messages.ToList();
        var approvalNames = ApprovalRequiredNames(chatOptions?.Tools);
        var held = HeldApprovals(chatOptions?.Tools, approvalNames);

        var responseMessages = new List<ChatMessage>();
        ApprovalOutcome approvals;
        try
        {
            approvals = ApprovalProcessor.Process(messages, fromHistory, held);
        }
        catch (MicrosoftAgentFrameworkWorkflowException exception)
        {
            NotifyFailure(requestMessages, exception);
            throw;
        }

        if (approvals.ReconstructedMessages.Count > 0)
        {
            responseMessages.AddRange(approvals.ReconstructedMessages);
            yield return new LoopStep(null, approvals.ReconstructedMessages);
        }

        if (approvals.ApprovedCalls.Count > 0)
        {
            var approvedResults = await InvokeToolsAsync(approvals.ApprovedCalls, chatOptions?.Tools, requestMessages, token).ConfigureAwait(true);
            messages.Insert(approvals.ResultInsertIndex, approvedResults);
            responseMessages.Add(approvedResults);
            yield return new LoopStep(null, [approvedResults]);
        }

        ChatResponse? last = null;
        UsageDetails? totalUsage = null;
        var deferred = new List<ToolApprovalRequestContent>();
        for (var iteration = 0; ; iteration++)
        {
            var callOptions = chatOptions;
            if (iteration >= options.MaximumIterationsPerRun)
            {
                Workflow.Logger.LogWarning(
                    "TemporalAgent reached MaximumIterationsPerRun ({Maximum}); making one more model call without function tools.",
                    options.MaximumIterationsPerRun);
                callOptions = WithoutFunctionTools(chatOptions);
            }

            ChatResponse response;
            try
            {
                response = await agent.ChatClient.GetResponseAsync(messages, callOptions, token).ConfigureAwait(true);
            }
            catch (Exception exception) when (exception is not ContinueAsNewException)
            {
                NotifyFailure(requestMessages, exception);
                throw;
            }

            foreach (var message in response.Messages)
            {
                message.AuthorName ??= agent.Name;
            }

            MarkServerHandledFunctionCalls(response.Messages);
            var approvalRequested = false;
            if (iteration < options.MaximumIterationsPerRun && approvalNames.Count > 0)
            {
                approvalRequested = ApprovalProcessor.ReplaceWithApprovalRequests(response, approvalNames, deferred);
            }

            last = response;
            responseMessages.AddRange(response.Messages);
            totalUsage = Add(totalUsage, response.Usage);
            yield return new LoopStep(response, [.. response.Messages]);

            if (approvalRequested)
            {
                // Every call of this response waits for the decision; the next run executes them.
                break;
            }

            var calls = iteration < options.MaximumIterationsPerRun ? PendingCalls(response.Messages) : [];
            if (calls.Count == 0 || HasDeclarationOnlyTarget(calls, callOptions?.Tools))
            {
                break;
            }

            messages.AddMessages(response);
            var results = await InvokeToolsAsync(calls, callOptions?.Tools, requestMessages, token).ConfigureAwait(true);
            messages.Add(results);
            responseMessages.Add(results);
            yield return new LoopStep(null, [results]);
        }

        var result = new AgentResponse(last!)
        {
            AgentId = agent.Id,
            Messages = responseMessages,
            Usage = totalUsage,
        };
        NotifySuccess(requestMessages, responseMessages);
        if (held.Count > 0)
        {
            session.StateBag.TryRemoveValue(DeferredApprovalsKey);
        }

        if (deferred.Count > 0)
        {
            session.StateBag.SetValue(DeferredApprovalsKey, deferred, AgentAbstractionsJsonUtilities.DefaultOptions);
        }

        Result = result;
    }

    // The provider runs with the SDK's scheduler listener on, as any Workflow code does: a thread
    // hop or a blocking wait on thread-pool work fails the Workflow Task with the SDK's own error.
    // What the listener cannot see is a provider that awaited an Activity, a Workflow timer or a
    // promise and returned an incomplete ValueTask; that is reported here as a Workflow failure.
    private static T RunSynchronously<T>(Func<ValueTask<T>> start, string what)
    {
        var task = start();
        if (!task.IsCompleted)
        {
            throw new MicrosoftAgentFrameworkWorkflowException(SynchronousRule(what));
        }

        return task.GetAwaiter().GetResult();
    }

    private static void RunSynchronously(Func<ValueTask> start, string what)
    {
        var task = start();
        if (!task.IsCompleted)
        {
            throw new MicrosoftAgentFrameworkWorkflowException(SynchronousRule(what));
        }

        task.GetAwaiter().GetResult();
    }

    private static string SynchronousRule(string what) =>
        $"{what} did not complete synchronously: it awaited an Activity, a Workflow timer or another incomplete task before returning. Chat history and context providers run inside the Workflow and must return a completed ValueTask. Run that work in the Workflow before calling the agent and add its result to the prompt or the provider's state.";

    private static UsageDetails? Add(UsageDetails? total, UsageDetails? usage)
    {
        if (usage is null)
        {
            return total;
        }

        // Sum into a fresh object: the response's own UsageDetails was already handed out in a
        // UsageContent update and must not change afterwards.
        total ??= new UsageDetails();
        total.Add(usage);
        return total;
    }

    private static void MarkServerHandledFunctionCalls(IList<ChatMessage> messages)
    {
        var resultIds = new HashSet<string>(
            messages.SelectMany(message => message.Contents).OfType<FunctionResultContent>().Select(result => result.CallId),
            StringComparer.Ordinal);
        if (resultIds.Count == 0)
        {
            return;
        }

        foreach (var call in messages.SelectMany(message => message.Contents).OfType<FunctionCallContent>())
        {
            if (!call.InformationalOnly && resultIds.Contains(call.CallId))
            {
                call.InformationalOnly = true;
            }
        }
    }

    private static List<FunctionCallContent> PendingCalls(IList<ChatMessage> messages) =>
        messages.SelectMany(message => message.Contents).OfType<FunctionCallContent>().Where(call => !call.InformationalOnly).ToList();

    private static HashSet<string> ApprovalRequiredNames(IList<AITool>? tools) =>
        new((tools ?? []).Where(tool => tool.GetService<ApprovalRequiredAIFunction>() is not null).Select(tool => tool.Name), StringComparer.Ordinal);

    private static AIFunctionDeclaration? FindTool(string name, IList<AITool>? tools)
    {
        if (tools is null)
        {
            return null;
        }

        foreach (var tool in tools)
        {
            if (tool is AIFunctionDeclaration declaration && string.Equals(declaration.Name, name, StringComparison.Ordinal))
            {
                return declaration;
            }
        }

        return null;
    }

    private static bool HasDeclarationOnlyTarget(List<FunctionCallContent> calls, IList<AITool>? tools)
    {
        foreach (var call in calls)
        {
            if (FindTool(call.Name, tools) is { } tool && tool is not AIFunction)
            {
                // A declaration without an implementation: hand the calls back to the caller, as FunctionInvokingChatClient does.
                return true;
            }
        }

        return false;
    }

    private static ChatOptions? WithoutFunctionTools(ChatOptions? chatOptions)
    {
        if (chatOptions?.Tools is not { Count: > 0 })
        {
            return chatOptions;
        }

        var remaining = chatOptions.Tools.Where(tool => tool is not AIFunctionDeclaration).ToList();
        if (remaining.Count == chatOptions.Tools.Count)
        {
            return chatOptions;
        }

        var clone = chatOptions.Clone();
        clone.Tools = remaining.Count == 0 ? null : remaining;
        if (remaining.Count == 0)
        {
            clone.ToolMode = null;
        }

        return clone;
    }

    // FunctionInvokingChatClient keeps the result object and the provider client serializes it with
    // AIJsonUtilities when it builds the request. Here the result crosses the Activity boundary first,
    // in the data converter's shape, so the same serialization happens in the Workflow and the model
    // reads the text it would have read natively: a string as is, an AIContent with its $type, any
    // other object as camel-cased JSON. The caller sees the JSON in FunctionResultContent.Result.
    private static object? ToTransportable(object? result) => result switch
    {
        null => null,
        string text => text,
        JsonElement element => element,
        AIContent content => JsonSerializer.SerializeToElement(content, AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(AIContent))),
        IEnumerable<AIContent> contents => JsonSerializer.SerializeToElement(contents.ToList(), AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(IList<AIContent>))),
        _ => JsonSerializer.SerializeToElement(result, result.GetType(), AIJsonUtilities.DefaultOptions),
    };

    private static bool IsCancellation(Exception exception) =>
        exception is OperationCanceledException || TemporalException.IsCanceledException(exception);

    private static async Task<FunctionResultContent> InvokeToolAsync(FunctionCallContent call, IList<AITool>? tools, CancellationToken token)
    {
        var tool = FindTool(call.Name, tools);
        if (tool is not AIFunction function)
        {
            Workflow.Logger.LogWarning("The model requested tool '{Tool}', which is not in ChatOptions.Tools.", call.Name);
            return new FunctionResultContent(call.CallId, $"Error: Requested function \"{call.Name}\" not found.");
        }

        try
        {
            var arguments = new AIFunctionArguments(call.Arguments);
            var result = await function.InvokeAsync(arguments, token).ConfigureAwait(true);
            return new FunctionResultContent(call.CallId, ToTransportable(result) ?? ToolSucceededText);
        }
#pragma warning disable CA1031 // The model is told that the tool failed; FunctionInvokingChatClient does the same.
        catch (Exception exception) when (!IsCancellation(exception) && exception is not ContinueAsNewException)
#pragma warning restore CA1031
        {
            Workflow.Logger.LogWarning(exception, "Tool '{Tool}' failed.", call.Name);
            return new FunctionResultContent(call.CallId, ToolFailedText) { Exception = exception };
        }
    }

    private ChatOptions? MergeChatOptions()
    {
        if (runOptions is ChatClientAgentRunOptions { ChatClientFactory: not null })
        {
            throw new MicrosoftAgentFrameworkWorkflowException(
                "ChatClientAgentRunOptions.ChatClientFactory is not supported: the provider client is built by the worker-side factory in MicrosoftAgentFrameworkPluginOptions.ChatClients.");
        }

        if (runOptions?.AllowBackgroundResponses is true)
        {
            throw new MicrosoftAgentFrameworkWorkflowException("AgentRunOptions.AllowBackgroundResponses is not supported; the model Activity waits for the complete response.");
        }

#pragma warning disable MEAI001 // ContinuationToken is experimental; the plugin only rejects it.
        if (runOptions?.ContinuationToken is not null)
        {
            throw new MicrosoftAgentFrameworkWorkflowException("AgentRunOptions.ContinuationToken is not supported; background responses do not cross the Activity boundary.");
        }
#pragma warning restore MEAI001

        var request = (runOptions as ChatClientAgentRunOptions)?.ChatOptions?.Clone();
        var agentOptions = options.ChatOptions;
        ChatOptions? merged;
        if (agentOptions is null)
        {
            merged = request;
        }
        else if (request is null)
        {
            merged = agentOptions.Clone();
        }
        else
        {
            merged = request;
            merged.AllowMultipleToolCalls ??= agentOptions.AllowMultipleToolCalls;
            merged.ConversationId ??= agentOptions.ConversationId;
            merged.FrequencyPenalty ??= agentOptions.FrequencyPenalty;
            merged.MaxOutputTokens ??= agentOptions.MaxOutputTokens;
            merged.ModelId ??= agentOptions.ModelId;
            merged.PresencePenalty ??= agentOptions.PresencePenalty;
            merged.ResponseFormat ??= agentOptions.ResponseFormat;
            merged.Reasoning ??= agentOptions.Reasoning;
            merged.Seed ??= agentOptions.Seed;
            merged.Temperature ??= agentOptions.Temperature;
            merged.TopP ??= agentOptions.TopP;
            merged.TopK ??= agentOptions.TopK;
            merged.ToolMode ??= agentOptions.ToolMode;
            merged.Instructions = (string.IsNullOrWhiteSpace(merged.Instructions), string.IsNullOrWhiteSpace(agentOptions.Instructions)) switch
            {
                (false, false) => $"{agentOptions.Instructions}\n{merged.Instructions}",
                (false, true) => merged.Instructions,
                _ => agentOptions.Instructions,
            };
            if (agentOptions.AdditionalProperties is { } agentProperties)
            {
                merged.AdditionalProperties ??= [];
                foreach (var pair in agentProperties)
                {
                    merged.AdditionalProperties.TryAdd(pair.Key, pair.Value);
                }
            }

            if (agentOptions.StopSequences is { Count: > 0 } stops)
            {
                merged.StopSequences = [.. merged.StopSequences ?? [], .. stops];
            }

            if (agentOptions.Tools is { Count: > 0 } agentTools)
            {
                merged.Tools = [.. merged.Tools ?? [], .. agentTools];
            }
        }

        if (runOptions?.ResponseFormat is { } responseFormat)
        {
            merged ??= new ChatOptions();
            merged.ResponseFormat = responseFormat;
        }

        if (runOptions?.AdditionalProperties is { Count: > 0 } runProperties)
        {
            merged ??= new ChatOptions();
            merged.AdditionalProperties ??= [];
            foreach (var pair in runProperties)
            {
                merged.AdditionalProperties[pair.Key] = pair.Value;
            }
        }

        return merged;
    }

    private List<ChatMessage> LoadHistory()
    {
#pragma warning disable MAAI001 // The provider contexts are experimental in Microsoft.Agents.AI 1.26; the package floor is that line.
        var context = new ChatHistoryProvider.InvokingContext(agent, session, input);
#pragma warning restore MAAI001
        var provider = agent.ChatHistoryProvider;
        var merged = RunSynchronously(() => provider.InvokingAsync(context, cancellationToken), $"{provider.GetType().Name}.InvokingAsync");
        return merged.ToList();
    }

    private ChatOptions? ApplyContextProviders(ChatOptions? chatOptions, ref List<ChatMessage> messages)
    {
        if (agent.AIContextProviders.Count == 0)
        {
            return chatOptions;
        }

        var aiContext = new AIContext
        {
            Instructions = chatOptions?.Instructions,
            Messages = messages,
            Tools = chatOptions?.Tools,
        };
        foreach (var provider in agent.AIContextProviders)
        {
#pragma warning disable MAAI001 // See LoadHistory.
            var context = new AIContextProvider.InvokingContext(agent, session, aiContext);
#pragma warning restore MAAI001
            aiContext = RunSynchronously(() => provider.InvokingAsync(context, cancellationToken), $"{provider.GetType().Name}.InvokingAsync");
        }

        messages = aiContext.Messages?.ToList() ?? [];
        var tools = aiContext.Tools?.ToList();
        if (chatOptions?.Tools is { Count: > 0 } || tools is { Count: > 0 })
        {
            chatOptions ??= new ChatOptions();
            chatOptions.Tools = tools;
        }

        if (chatOptions?.Instructions is not null || aiContext.Instructions is not null)
        {
            chatOptions ??= new ChatOptions();
            chatOptions.Instructions = aiContext.Instructions;
        }

        return chatOptions;
    }

    private List<HeldApproval> HeldApprovals(IList<AITool>? tools, HashSet<string> approvalNames)
    {
        if (!session.StateBag.TryGetValue(DeferredApprovalsKey, out List<ToolApprovalRequestContent>? pending, AgentAbstractionsJsonUtilities.DefaultOptions) ||
            pending is not { Count: > 0 })
        {
            return [];
        }

        // The previous response mixed these calls with ones that needed a decision. Approve them
        // now, unless the tool has since gone or started to need approval, as Agent Framework does.
        var held = new List<HeldApproval>(pending.Count);
        foreach (var request in pending)
        {
            var name = (request.ToolCall as FunctionCallContent)?.Name;
            var stillPlain = name is not null && FindTool(name, tools) is AIFunction && !approvalNames.Contains(name);
            if (!stillPlain)
            {
                Workflow.Logger.LogWarning("A call to '{Tool}' was held back on the previous run; the tool now needs approval or is gone, so it is rejected instead of run.", name);
            }

            held.Add(new HeldApproval(request, stillPlain));
        }

        return held;
    }

    private void NotifySuccess(IReadOnlyList<ChatMessage> requestMessages, IReadOnlyList<ChatMessage> responseMessages)
    {
#pragma warning disable MAAI001 // See LoadHistory.
        var history = agent.ChatHistoryProvider;
        var historyContext = new ChatHistoryProvider.InvokedContext(agent, session, requestMessages, responseMessages);
        RunSynchronously(() => history.InvokedAsync(historyContext, cancellationToken), $"{history.GetType().Name}.InvokedAsync");
        foreach (var provider in agent.AIContextProviders)
        {
            var context = new AIContextProvider.InvokedContext(agent, session, requestMessages, responseMessages);
            RunSynchronously(() => provider.InvokedAsync(context, cancellationToken), $"{provider.GetType().Name}.InvokedAsync");
        }
#pragma warning restore MAAI001
    }

    private void NotifyFailure(IReadOnlyList<ChatMessage> requestMessages, Exception exception)
    {
        var history = agent.ChatHistoryProvider;
        var historyContext = new ChatHistoryProvider.InvokedContext(agent, session, requestMessages, exception);
        RunSynchronously(() => history.InvokedAsync(historyContext, cancellationToken), $"{history.GetType().Name}.InvokedAsync");
        foreach (var provider in agent.AIContextProviders)
        {
            var context = new AIContextProvider.InvokedContext(agent, session, requestMessages, exception);
            RunSynchronously(() => provider.InvokedAsync(context, cancellationToken), $"{provider.GetType().Name}.InvokedAsync");
        }
    }

    private async Task<ChatMessage> InvokeToolsAsync(
        IReadOnlyList<FunctionCallContent> calls, IList<AITool>? tools, IReadOnlyList<ChatMessage> requestMessages, CancellationToken token)
    {
        FunctionResultContent[] results;
        if (options.AllowConcurrentInvocation && calls.Count > 1)
        {
            var tasks = calls.Select(call => InvokeToolAsync(call, tools, token)).ToList();
            results = await Workflow.WhenAllAsync(tasks).ConfigureAwait(true);
        }
        else
        {
            results = new FunctionResultContent[calls.Count];
            for (var i = 0; i < calls.Count; i++)
            {
                results[i] = await InvokeToolAsync(calls[i], tools, token).ConfigureAwait(true);
            }
        }

        foreach (var call in calls)
        {
            call.InformationalOnly = true;
        }

        if (Array.Exists(results, result => result.Exception is not null))
        {
            consecutiveToolErrors++;
            if (consecutiveToolErrors > MaximumConsecutiveToolErrors)
            {
                var failures = results.Where(result => result.Exception is not null).Select(result => result.Exception!).ToArray();
                var exception = new MicrosoftAgentFrameworkWorkflowException(
                    $"Tools failed on {consecutiveToolErrors} consecutive rounds; the last failure came from '{string.Join(", ", calls.Select(call => call.Name))}'.",
                    failures.Length == 1 ? failures[0] : new AggregateException(failures));
                NotifyFailure(requestMessages, exception);
                throw exception;
            }
        }
        else
        {
            consecutiveToolErrors = 0;
        }

        return new ChatMessage(ChatRole.Tool, results.Cast<AIContent>().ToList());
    }
}
