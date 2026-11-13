using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Temporalio.MicrosoftAgentFramework.Activities;
using Temporalio.MicrosoftAgentFramework.Internal;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// The Workflow-side <see cref="IChatClient"/>. Every <see cref="GetResponseAsync"/> runs one
/// <c>MicrosoftAgentFramework.InvokeModel</c> Activity; the worker resolves <see cref="Model"/> against
/// <see cref="MicrosoftAgentFrameworkPluginOptions.ChatClients"/>. Use it on its own or through
/// <see cref="TemporalAgent"/>.
/// </summary>
public sealed class TemporalChatClient : IChatClient
{
    /// <summary>Default <c>StartToCloseTimeout</c> of the model Activity: 5 minutes.</summary>
    public static readonly TimeSpan DefaultStartToCloseTimeout = TimeSpan.FromMinutes(5);

    /// <summary>Default <c>HeartbeatTimeout</c> of the model Activity: 10 seconds.</summary>
    public static readonly TimeSpan DefaultHeartbeatTimeout = TimeSpan.FromSeconds(10);

    /// <summary>Default interval at which streamed updates are batched into one publish signal: 100 milliseconds.</summary>
    public static readonly TimeSpan DefaultStreamingBatchInterval = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalChatClient"/> class.
    /// </summary>
    /// <param name="model">Model name, a key of <see cref="MicrosoftAgentFrameworkPluginOptions.ChatClients"/> on the worker.</param>
    /// <param name="activityOptions">
    /// Options for the model Activity. Fields left null get the defaults: 5-minute start-to-close
    /// timeout, 10-second heartbeat timeout, summary <c>microsoft-agent-framework model call: {model}</c>.
    /// </param>
    /// <param name="streamingTopic">
    /// Workflow Stream topic that receives each <see cref="ChatResponseUpdate"/> from inside the
    /// Activity, or null. The Workflow must host a <c>WorkflowStream</c>.
    /// </param>
    /// <param name="streamingBatchInterval">How long updates are batched before one publish signal carries them. Defaults to 100 milliseconds.</param>
    public TemporalChatClient(
        string model,
        ActivityOptions? activityOptions = null,
        string? streamingTopic = null,
        TimeSpan? streamingBatchInterval = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        Model = model;
        ActivityOptions = WithDefaults(model, activityOptions);
        StreamingTopic = streamingTopic;
        StreamingBatchInterval = streamingBatchInterval ?? DefaultStreamingBatchInterval;
        if (StreamingBatchInterval <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(streamingBatchInterval), "The streaming batch interval must be positive.");
        }
    }

    /// <summary>Gets the model name sent to the worker.</summary>
    public string Model { get; }

    /// <summary>Gets the options every model Activity is scheduled with, defaults applied.</summary>
    public ActivityOptions ActivityOptions { get; }

    /// <summary>Gets the Workflow Stream topic for streamed updates, or null.</summary>
    public string? StreamingTopic { get; }

    /// <summary>Gets the batch interval for streamed updates.</summary>
    public TimeSpan StreamingBatchInterval { get; }

    /// <summary>
    /// Runs one model Activity and returns its response.
    /// </summary>
    /// <param name="messages">Messages to send.</param>
    /// <param name="options">Chat options; tools travel as declarations.</param>
    /// <param name="cancellationToken">Cancels the Activity.</param>
    /// <returns>The provider response as returned by the Activity.</returns>
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        InvokeAsync(messages, options, stream: false, cancellationToken);

    /// <summary>
    /// Runs one model Activity that consumes the provider's stream and replays the aggregated
    /// response as updates. When <see cref="StreamingTopic"/> is set, each update is also published
    /// from the Activity as the provider produces it.
    /// </summary>
    /// <param name="messages">Messages to send.</param>
    /// <param name="options">Chat options; tools travel as declarations.</param>
    /// <param name="cancellationToken">Cancels the Activity.</param>
    /// <returns>The updates of the aggregated response.</returns>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await InvokeAsync(messages, options, stream: true, cancellationToken).ConfigureAwait(true);
        foreach (var update in response.ToChatResponseUpdates())
        {
            yield return update;
        }
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(ChatClientMetadata))
        {
            return new ChatClientMetadata("temporal", null, Model);
        }

        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <summary>Does nothing; the client holds no resources.</summary>
    public void Dispose()
    {
    }

    /// <summary>
    /// Schedules the model Activity.
    /// </summary>
    /// <param name="messages">Messages to send.</param>
    /// <param name="options">Chat options.</param>
    /// <param name="stream">Whether the Activity consumes the provider stream.</param>
    /// <param name="cancellationToken">Cancels the Activity.</param>
    /// <returns>The Activity result.</returns>
    internal Task<ChatResponse> InvokeAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options, bool stream, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (!Workflow.InWorkflow)
        {
            throw new InvalidOperationException(
                "TemporalChatClient runs model calls as Temporal Activities and must be called inside a Workflow. " +
                "Outside a Workflow, use the provider client that your MicrosoftAgentFrameworkPluginOptions.ChatClients factory builds.");
        }

        var (transportOptions, tools, hostedTools) = ChatOptionsTransport.Prepare(options, Workflow.Logger);
        var input = new ModelCallInput
        {
            Model = Model,
            Messages = messages.ToList(),
            Options = transportOptions,
            Tools = tools,
            HostedTools = hostedTools,
            Stream = stream || StreamingTopic is not null,
            StreamingTopic = StreamingTopic,
            StreamingBatchInterval = StreamingTopic is null ? null : StreamingBatchInterval,
        };
        var activityOptions = ActivityOptions;
        if (cancellationToken.CanBeCanceled && activityOptions.CancellationToken is null)
        {
            activityOptions = (ActivityOptions)activityOptions.Clone();
            activityOptions.CancellationToken = cancellationToken;
        }

        return Workflow.ExecuteActivityAsync(
            (ModelActivities activities) => activities.InvokeModelAsync(input), activityOptions);
    }

    private static ActivityOptions WithDefaults(string model, ActivityOptions? options)
    {
        var result = options is null ? new ActivityOptions() : (ActivityOptions)options.Clone();
        if (result.StartToCloseTimeout is null && result.ScheduleToCloseTimeout is null)
        {
            result.StartToCloseTimeout = DefaultStartToCloseTimeout;
        }

        result.HeartbeatTimeout ??= DefaultHeartbeatTimeout;
        result.Summary ??= $"microsoft-agent-framework model call: {model}";
        return result;
    }
}
