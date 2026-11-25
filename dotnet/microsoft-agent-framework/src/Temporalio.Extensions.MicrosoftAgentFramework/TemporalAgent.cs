using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.Extensions.MicrosoftAgentFramework.Internal;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework;

/// <summary>
/// A Microsoft Agent Framework agent that runs inside a Temporal Workflow. Construct it in
/// Workflow code with a model name and <see cref="TemporalAgentOptions"/>, then call
/// <see cref="RunAsync(string, AgentSession?, AgentRunOptions?, CancellationToken)"/>,
/// <see cref="RunAsync{T}(string, AgentSession?, JsonSerializerOptions?, AgentRunOptions?, CancellationToken)"/>
/// or <see cref="RunStreamingAsync(string, AgentSession?, AgentRunOptions?, CancellationToken)"/>.
/// Each model call is one Activity; tools run where you put them: plain <see cref="AIFunction"/>s in
/// the Workflow, <see cref="ActivityAsTool"/> and <see cref="Mcp.TemporalMcpTool"/> as Activities.
/// </summary>
/// <remarks>
/// The run methods hide the <see cref="AIAgent"/> ones with the <c>new</c> modifier on purpose.
/// <c>AIAgent.RunAsync</c> awaits <see cref="RunCoreAsync"/> with <c>ConfigureAwait(false)</c>,
/// which moves its continuation off the Workflow scheduler and leaves the Workflow Task without
/// a result. The overloads here run the same loop without that hop, and <see cref="RunCoreAsync"/>
/// throws when something reaches it through an <see cref="AIAgent"/>-typed reference.
/// </remarks>
public sealed class TemporalAgent : AIAgent
{
    private const string DefaultId = "temporal-agent";

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalAgent"/> class over a new
    /// <see cref="TemporalChatClient"/> for <paramref name="model"/>.
    /// </summary>
    /// <param name="model">Model name, a key of <see cref="MicrosoftAgentFrameworkPluginOptions.ChatClients"/> on the worker.</param>
    /// <param name="options">Agent options; <see cref="TemporalAgentOptions.ModelActivityOptions"/>,
    /// <see cref="TemporalAgentOptions.StreamingTopic"/> and <see cref="TemporalAgentOptions.StreamingBatchInterval"/>
    /// configure the chat client.</param>
    public TemporalAgent(string model, TemporalAgentOptions? options = null)
        : this(
            new TemporalChatClient(model, options?.ModelActivityOptions, options?.StreamingTopic, options?.StreamingBatchInterval),
            options,
            clientFromOptions: true)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalAgent"/> class over an existing chat client.
    /// </summary>
    /// <param name="chatClient">The Workflow-side chat client.</param>
    /// <param name="options">Agent options. The chat client fields must be left null here; set them on <paramref name="chatClient"/>.</param>
    public TemporalAgent(TemporalChatClient chatClient, TemporalAgentOptions? options = null)
        : this(chatClient, options, clientFromOptions: false)
    {
    }

    private TemporalAgent(TemporalChatClient chatClient, TemporalAgentOptions? options, bool clientFromOptions)
    {
        ArgumentNullException.ThrowIfNull(chatClient);
        ChatClient = chatClient;
        Options = options?.Clone() ?? new TemporalAgentOptions();
        if (!clientFromOptions && (Options.ModelActivityOptions is not null || Options.StreamingTopic is not null || Options.StreamingBatchInterval is not null))
        {
            throw new ArgumentException(
                "ModelActivityOptions, StreamingTopic and StreamingBatchInterval configure the TemporalChatClient that the (string model, ...) constructor builds. Set them on the TemporalChatClient you pass in.",
                nameof(options));
        }

        ChatHistoryProvider = Options.ChatHistoryProvider ?? new InMemoryChatHistoryProvider();
        AIContextProviders = Options.AIContextProviders?.ToList() ?? [];
        ValidateStateKeys();
        if (Options.ChatHistoryProvider is not null and not InMemoryChatHistoryProvider && Workflow.InWorkflow)
        {
            Workflow.Logger.LogWarning(
                "TemporalAgent is using {Provider} as its chat history provider. It runs inside the Workflow and persists next to Event History, which already holds the conversation; InMemoryChatHistoryProvider is the default for that reason.",
                Options.ChatHistoryProvider.GetType().Name);
        }
    }

    /// <summary>Gets the Workflow-side chat client.</summary>
    public TemporalChatClient ChatClient { get; }

    /// <summary>Gets a copy of the options the agent was built with.</summary>
    public TemporalAgentOptions Options { get; }

    /// <summary>Gets the chat history provider in use.</summary>
    public ChatHistoryProvider ChatHistoryProvider { get; }

    /// <summary>Gets the context providers in use.</summary>
    public IReadOnlyList<AIContextProvider> AIContextProviders { get; }

    /// <inheritdoc />
    public override string? Name => Options.Name;

    /// <inheritdoc />
    public override string? Description => Options.Description;

    /// <summary>
    /// Gets the agent id: <see cref="TemporalAgentOptions.Id"/>, then <see cref="TemporalAgentOptions.Name"/>,
    /// then <c>temporal-agent</c>. The random id <see cref="AIAgent"/> generates is never read, so
    /// <see cref="AgentResponse.AgentId"/> is the same on every replay.
    /// </summary>
    protected override string? IdCore => Options.Id ?? Options.Name ?? DefaultId;

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (base.GetService(serviceType, serviceKey) is { } own)
        {
            return own;
        }

        if (serviceKey is null)
        {
            if (serviceType == typeof(IChatClient) || serviceType == typeof(TemporalChatClient))
            {
                return ChatClient;
            }

            if (serviceType == typeof(ChatOptions))
            {
                return Options.ChatOptions;
            }

            if (serviceType == typeof(TemporalAgentOptions))
            {
                return Options;
            }
        }

        return AIContextProviders.Select(provider => provider.GetService(serviceType, serviceKey)).FirstOrDefault(service => service is not null)
            ?? ChatHistoryProvider.GetService(serviceType, serviceKey)
            ?? ChatClient.GetService(serviceType, serviceKey);
    }

    /// <summary>Runs the agent with no new message, for example to continue after approvals carried in the session.</summary>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response.</returns>
    public new Task<AgentResponse> RunAsync(AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default) =>
        RunAsync([], session, options, cancellationToken);

    /// <summary>Runs the agent with one user message.</summary>
    /// <param name="message">The user message.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response.</returns>
    public new Task<AgentResponse> RunAsync(string message, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);
        return RunAsync(new ChatMessage(ChatRole.User, message), session, options, cancellationToken);
    }

    /// <summary>Runs the agent with one message.</summary>
    /// <param name="message">The message.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response.</returns>
    public new Task<AgentResponse> RunAsync(ChatMessage message, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(message);
        return RunAsync([message], session, options, cancellationToken);
    }

    /// <summary>Runs the agent with new messages.</summary>
    /// <param name="messages">The messages.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response.</returns>
    public new async Task<AgentResponse> RunAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        var loop = CreateLoop(messages, session, options, cancellationToken);
        var steps = loop.RunAsync(cancellationToken).GetAsyncEnumerator(cancellationToken);
        await using (steps.ConfigureAwait(true))
        {
            while (await steps.MoveNextAsync().ConfigureAwait(true))
            {
                // Each step is a model response or a tool round; RunAsync only needs the end result.
            }
        }

        return loop.Result!;
    }

    /// <summary>Runs the agent and parses its JSON answer as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">Result type; its JSON schema is sent as the response format.</typeparam>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="serializerOptions">Options for the schema and the parse.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response; read <see cref="AgentResponse{T}.Result"/>.</returns>
    public new Task<AgentResponse<T>> RunAsync<T>(AgentSession? session = null, JsonSerializerOptions? serializerOptions = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default) =>
        RunAsync<T>([], session, serializerOptions, options, cancellationToken);

    /// <summary>Runs the agent with one user message and parses its JSON answer as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">Result type; its JSON schema is sent as the response format.</typeparam>
    /// <param name="message">The user message.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="serializerOptions">Options for the schema and the parse.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response; read <see cref="AgentResponse{T}.Result"/>.</returns>
    public new Task<AgentResponse<T>> RunAsync<T>(string message, AgentSession? session = null, JsonSerializerOptions? serializerOptions = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);
        return RunAsync<T>(new ChatMessage(ChatRole.User, message), session, serializerOptions, options, cancellationToken);
    }

    /// <summary>Runs the agent with one message and parses its JSON answer as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">Result type; its JSON schema is sent as the response format.</typeparam>
    /// <param name="message">The message.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="serializerOptions">Options for the schema and the parse.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response; read <see cref="AgentResponse{T}.Result"/>.</returns>
    public new Task<AgentResponse<T>> RunAsync<T>(ChatMessage message, AgentSession? session = null, JsonSerializerOptions? serializerOptions = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(message);
        return RunAsync<T>([message], session, serializerOptions, options, cancellationToken);
    }

    /// <summary>Runs the agent with new messages and parses its JSON answer as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">Result type; its JSON schema is sent as the response format.</typeparam>
    /// <param name="messages">The messages.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="serializerOptions">Options for the schema and the parse.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>The response; read <see cref="AgentResponse{T}.Result"/>.</returns>
    public new async Task<AgentResponse<T>> RunAsync<T>(IEnumerable<ChatMessage> messages, AgentSession? session = null, JsonSerializerOptions? serializerOptions = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        serializerOptions ??= AgentAbstractionsJsonUtilities.DefaultOptions;
        var (format, wrapped) = StructuredOutput.FormatFor<T>(serializerOptions);
        options = options?.Clone() ?? new AgentRunOptions();
        options.ResponseFormat = format;
        var response = await RunAsync(messages, session, options, cancellationToken).ConfigureAwait(true);
        return new AgentResponse<T>(response, serializerOptions) { IsWrappedInObject = wrapped };
    }

    /// <summary>Runs the agent with no new message and yields the response as updates.</summary>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>Updates: the aggregated response of each model call, replayed, and each tool result.</returns>
    public new IAsyncEnumerable<AgentResponseUpdate> RunStreamingAsync(AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default) =>
        RunStreamingAsync([], session, options, cancellationToken);

    /// <summary>Runs the agent with one user message and yields the response as updates.</summary>
    /// <param name="message">The user message.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>Updates: the aggregated response of each model call, replayed, and each tool result.</returns>
    public new IAsyncEnumerable<AgentResponseUpdate> RunStreamingAsync(string message, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);
        return RunStreamingAsync(new ChatMessage(ChatRole.User, message), session, options, cancellationToken);
    }

    /// <summary>Runs the agent with one message and yields the response as updates.</summary>
    /// <param name="message">The message.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>Updates: the aggregated response of each model call, replayed, and each tool result.</returns>
    public new IAsyncEnumerable<AgentResponseUpdate> RunStreamingAsync(ChatMessage message, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(message);
        return RunStreamingAsync([message], session, options, cancellationToken);
    }

    /// <summary>Runs the agent with new messages and yields the response as updates.</summary>
    /// <param name="messages">The messages.</param>
    /// <param name="session">Session to continue; a new one when null.</param>
    /// <param name="options">Per-run options.</param>
    /// <param name="cancellationToken">Cancels the Activities of this run.</param>
    /// <returns>Updates: the aggregated response of each model call, replayed, and each tool result.</returns>
    public new async IAsyncEnumerable<AgentResponseUpdate> RunStreamingAsync(
        IEnumerable<ChatMessage> messages,
        AgentSession? session = null,
        AgentRunOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var loop = CreateLoop(messages, session, options, cancellationToken);
        await foreach (var step in loop.RunAsync(cancellationToken).ConfigureAwait(true))
        {
            if (step.Response is { } response)
            {
                foreach (var update in response.ToChatResponseUpdates())
                {
                    yield return new AgentResponseUpdate(update) { AgentId = Id };
                }

                continue;
            }

            foreach (var message in step.Messages)
            {
                yield return new AgentResponseUpdate(message.Role, message.Contents)
                {
                    AgentId = Id,
                    AuthorName = message.AuthorName,
                    MessageId = message.MessageId,
                    CreatedAt = message.CreatedAt,
                    AdditionalProperties = message.AdditionalProperties,
                };
            }
        }
    }

    /// <inheritdoc />
    protected override ValueTask<AgentSession> CreateSessionCoreAsync(CancellationToken cancellationToken = default) =>
        new(new TemporalAgentSession());

    /// <summary>
    /// Serializes a <see cref="TemporalAgentSession"/>. The state bag already holds JSON, so
    /// <paramref name="jsonSerializerOptions"/> is accepted for the base signature and not used.
    /// </summary>
    /// <param name="session">The session.</param>
    /// <param name="jsonSerializerOptions">Not used.</param>
    /// <param name="cancellationToken">Not used; the call completes synchronously.</param>
    /// <returns>The serialized session.</returns>
    protected override ValueTask<JsonElement> SerializeSessionCoreAsync(AgentSession session, JsonSerializerOptions? jsonSerializerOptions = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        return new(RequireSession(session).Serialize());
    }

    /// <summary>
    /// Restores a session serialized by <see cref="SerializeSessionCoreAsync"/>.
    /// </summary>
    /// <param name="serializedState">The serialized session.</param>
    /// <param name="jsonSerializerOptions">Not used.</param>
    /// <param name="cancellationToken">Not used; the call completes synchronously.</param>
    /// <returns>The session.</returns>
    protected override ValueTask<AgentSession> DeserializeSessionCoreAsync(JsonElement serializedState, JsonSerializerOptions? jsonSerializerOptions = null, CancellationToken cancellationToken = default) =>
        new(TemporalAgentSession.Deserialize(serializedState));

    /// <inheritdoc />
    /// <remarks>
    /// Reached only through <see cref="AIAgent"/>'s own run methods, which resume off the
    /// Workflow scheduler. Always throws <see cref="MicrosoftAgentFrameworkWorkflowException"/>.
    /// </remarks>
    protected override Task<AgentResponse> RunCoreAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default) =>
        throw BaseEntryGuard();

    /// <inheritdoc />
    /// <remarks>
    /// Reached only through <see cref="AIAgent"/>'s own streaming methods. Always throws
    /// <see cref="MicrosoftAgentFrameworkWorkflowException"/>.
    /// </remarks>
    protected override IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default) =>
        throw BaseEntryGuard();

    private static MicrosoftAgentFrameworkWorkflowException BaseEntryGuard() => new(
        "TemporalAgent was entered through AIAgent.RunAsync or AIAgent.RunStreamingAsync, which resume off the Workflow scheduler (ConfigureAwait(false)) and would stall the Workflow. " +
        "Call RunAsync, RunAsync<T> or RunStreamingAsync on a TemporalAgent-typed reference. Agent middleware (AIAgentBuilder, DelegatingAIAgent, AsAIFunction) is not supported inside a Workflow.");

    private static TemporalAgentSession RequireSession(AgentSession session) =>
        session as TemporalAgentSession ?? throw new MicrosoftAgentFrameworkWorkflowException(
            $"The session type '{session.GetType().Name}' is not compatible with TemporalAgent. Create sessions with agent.CreateSessionAsync() or agent.DeserializeSessionAsync().");

    private AgentLoop CreateLoop(IEnumerable<ChatMessage> messages, AgentSession? session, AgentRunOptions? options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var input = messages as IReadOnlyCollection<ChatMessage> ?? messages.ToList();
        var typedSession = session is null ? new TemporalAgentSession() : RequireSession(session);

        // Set for parity with AIAgent.RunAsync so a provider that reads AIAgent.CurrentRunContext
        // sees the run; nothing in the plugin reads it, and the async machinery restores the caller's value.
        CurrentRunContext = new AgentRunContext(this, typedSession, input, options);
        return new AgentLoop(this, Options, typedSession, input, options, cancellationToken);
    }

    private void ValidateStateKeys()
    {
        var seen = new HashSet<string>(StringComparer.Ordinal) { AgentLoop.DeferredApprovalsKey };
        foreach (var key in AIContextProviders.SelectMany(provider => provider.StateKeys).Concat(ChatHistoryProvider.StateKeys))
        {
            if (!seen.Add(key))
            {
                throw new ArgumentException(
                    key == AgentLoop.DeferredApprovalsKey
                        ? $"A provider on this agent declares the session state key '{key}', which TemporalAgent uses for calls held back behind an approval."
                        : $"Two providers on this agent share the session state key '{key}'.");
            }
        }
    }
}
