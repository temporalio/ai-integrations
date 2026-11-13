using System.Collections.Concurrent;
using Microsoft.Extensions.AI;
using Temporalio.Activities;
using Temporalio.AgentFramework.Internal;
using Temporalio.Exceptions;
using Temporalio.Extensions.WorkflowStreams;

namespace Temporalio.AgentFramework.Activities;

/// <summary>
/// The model Activity. Registered by <see cref="AgentFrameworkPlugin"/>; the provider client
/// comes from the factory registered under the input's model name and is created once per
/// worker.
/// </summary>
public sealed class ModelActivities
{
    /// <summary>Name of the model Activity.</summary>
    public const string InvokeModelActivityName = "AgentFramework.InvokeModel";

    /// <summary>Updates per publish Signal before the batch is flushed early, whatever the interval.</summary>
    public const int MaxStreamingBatchSize = 256;

    private static readonly TimeSpan MaxHeartbeatInterval = TimeSpan.FromSeconds(3);
    private static readonly TimeSpan MinHeartbeatInterval = TimeSpan.FromMilliseconds(100);

    private readonly IReadOnlyDictionary<string, Func<IChatClient>> factories;
    private readonly ConcurrentDictionary<string, IChatClient> clients = new(StringComparer.Ordinal);
    private readonly object clientLock = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelActivities"/> class.
    /// </summary>
    /// <param name="factories">Chat client factories by model name.</param>
    public ModelActivities(IReadOnlyDictionary<string, Func<IChatClient>> factories)
    {
        ArgumentNullException.ThrowIfNull(factories);
        this.factories = factories;
    }

    /// <summary>
    /// Runs one model call. Heartbeats every few seconds while the provider call runs and on
    /// every streamed update; classifies provider HTTP errors through
    /// <see cref="AgentFrameworkErrorTypes.ModelError"/>.
    /// </summary>
    /// <param name="input">The call.</param>
    /// <returns>The aggregated provider response.</returns>
    [Activity(InvokeModelActivityName)]
    public async Task<ChatResponse> InvokeModelAsync(ModelCallInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var context = ActivityExecutionContext.Current;
        var client = ResolveClient(input.Model);
        var options = ChatOptionsTransport.Rebuild(input);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(context.CancellationToken);
        var pump = HeartbeatAsync(context, stop.Token);
        try
        {
            return input.Stream
                ? await StreamAsync(client, input, options, context).ConfigureAwait(false)
                : await client.GetResponseAsync(input.Messages, options, context.CancellationToken).ConfigureAwait(false);
        }
        catch (Exception exception) when (exception is not OperationCanceledException && ModelErrorClassifier.Classify(exception) is { } failure)
        {
            throw failure;
        }
        finally
        {
#pragma warning disable CA1849 // Cancel is the form the Temporal .NET SDK recommends.
            stop.Cancel();
#pragma warning restore CA1849
            await pump.ConfigureAwait(false);
        }
    }

    private static async Task<ChatResponse> StreamAsync(
        IChatClient client, ModelCallInput input, ChatOptions? options, ActivityExecutionContext context)
    {
        var updates = new List<ChatResponseUpdate>();
        WorkflowStreamClient? streamClient = null;
        if (input.StreamingTopic is not null)
        {
            streamClient = WorkflowStreamClient.FromActivity(new WorkflowStreamClientOptions
            {
                BatchInterval = input.StreamingBatchInterval ?? TemporalChatClient.DefaultStreamingBatchInterval,
                MaxBatchSize = MaxStreamingBatchSize,
            });
        }

        try
        {
            var topic = streamClient?.GetTopic<ChatResponseUpdate>(input.StreamingTopic);
            var stream = client.GetStreamingResponseAsync(input.Messages, options, context.CancellationToken);
            await foreach (var update in stream.WithCancellation(context.CancellationToken).ConfigureAwait(false))
            {
                context.Heartbeat();
                topic?.Publish(update);
                updates.Add(update);
            }

            if (streamClient is not null)
            {
                await streamClient.FlushAsync(context.CancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            if (streamClient is not null)
            {
                await streamClient.DisposeAsync().ConfigureAwait(false);
            }
        }

        return updates.ToChatResponse();
    }

    private static async Task HeartbeatAsync(ActivityExecutionContext context, CancellationToken stop)
    {
        var interval = MaxHeartbeatInterval;
        if (context.Info.HeartbeatTimeout is { } timeout)
        {
            var third = TimeSpan.FromTicks(timeout.Ticks / 3);
            interval = third < MinHeartbeatInterval ? MinHeartbeatInterval : third < MaxHeartbeatInterval ? third : MaxHeartbeatInterval;
        }

        using var timer = new PeriodicTimer(interval);
        try
        {
            while (await timer.WaitForNextTickAsync(stop).ConfigureAwait(false))
            {
                context.Heartbeat();
            }
        }
        catch (OperationCanceledException)
        {
            // The provider call finished; the pump is done.
        }
    }

    private IChatClient ResolveClient(string model)
    {
        if (clients.TryGetValue(model, out var existing))
        {
            return existing;
        }

        if (!factories.TryGetValue(model, out var factory))
        {
            throw new ApplicationFailureException(
                $"No chat client is registered for model '{model}'. Add it to AgentFrameworkPluginOptions.ChatClients on the worker.",
                errorType: AgentFrameworkErrorTypes.ConfigurationError,
                nonRetryable: true);
        }

        lock (clientLock)
        {
            if (clients.TryGetValue(model, out existing))
            {
                return existing;
            }

            var client = factory() ?? throw new ApplicationFailureException(
                $"The chat client factory for model '{model}' returned null.",
                errorType: AgentFrameworkErrorTypes.ConfigurationError,
                nonRetryable: true);
            if (client.GetService<FunctionInvokingChatClient>() is not null)
            {
                throw new ApplicationFailureException(
                    $"The chat client for model '{model}' contains a FunctionInvokingChatClient, which would run tools inside the model Activity. " +
                    "Register the provider client without UseFunctionInvocation(); TemporalAgent runs the tool loop in the Workflow.",
                    errorType: AgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: true);
            }

            clients[model] = client;
            return client;
        }
    }
}
