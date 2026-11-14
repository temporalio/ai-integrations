using System.Collections.Concurrent;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Temporalio.Activities;
using Temporalio.Exceptions;
using Temporalio.Extensions.WorkflowStreams;
using Temporalio.Extensions.MicrosoftAgentFramework.Internal;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Activities;

/// <summary>
/// The model Activity. Registered by <see cref="MicrosoftAgentFrameworkPlugin"/>; the provider client
/// comes from the factory registered under the input's model name and is created once per
/// worker.
/// </summary>
public sealed class ModelActivities
{
    /// <summary>Name of the model Activity.</summary>
    public const string InvokeModelActivityName = "MicrosoftAgentFramework.InvokeModel";

    /// <summary>Updates per publish Signal before the batch is flushed early, whatever the interval.</summary>
    public const int MaxStreamingBatchSize = 256;

    private static readonly TimeSpan MaxHeartbeatInterval = TimeSpan.FromSeconds(3);
    private static readonly TimeSpan MinHeartbeatInterval = TimeSpan.FromMilliseconds(100);

    private readonly IReadOnlyDictionary<string, Func<IChatClient>> factories;
    private readonly IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? hostedMcpServerHeaders;
    private readonly IReadOnlyDictionary<string, Func<IReadOnlyDictionary<string, string>>>? hostedMcpServerHeaderProviders;
    private readonly ConcurrentDictionary<string, IChatClient> clients = new(StringComparer.Ordinal);
    private readonly object clientLock = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelActivities"/> class.
    /// </summary>
    /// <param name="factories">Chat client factories by model name.</param>
    /// <param name="hostedMcpServerHeaders">
    /// HTTP headers added to hosted MCP server tools when the Activity rebuilds them, by server
    /// name; see <see cref="MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaders"/>.
    /// </param>
    public ModelActivities(
        IReadOnlyDictionary<string, Func<IChatClient>> factories,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? hostedMcpServerHeaders = null)
        : this(factories, hostedMcpServerHeaders, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelActivities"/> class with header providers
    /// that are called before each model call, so a token can be refreshed.
    /// </summary>
    /// <param name="factories">Chat client factories by model name.</param>
    /// <param name="hostedMcpServerHeaders">Fixed headers by server name; see <see cref="MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaders"/>.</param>
    /// <param name="hostedMcpServerHeaderProviders">Header providers by server name; see <see cref="MicrosoftAgentFrameworkPluginOptions.HostedMcpServerHeaderProviders"/>.</param>
    public ModelActivities(
        IReadOnlyDictionary<string, Func<IChatClient>> factories,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? hostedMcpServerHeaders,
        IReadOnlyDictionary<string, Func<IReadOnlyDictionary<string, string>>>? hostedMcpServerHeaderProviders)
    {
        ArgumentNullException.ThrowIfNull(factories);
        this.factories = factories;
        this.hostedMcpServerHeaders = hostedMcpServerHeaders;
        this.hostedMcpServerHeaderProviders = hostedMcpServerHeaderProviders;
    }

    /// <summary>
    /// Runs one model call. Heartbeats every few seconds while the provider call runs and on
    /// every streamed update; classifies provider HTTP errors through
    /// <see cref="MicrosoftAgentFrameworkErrorTypes.ModelError"/>.
    /// </summary>
    /// <param name="input">The call.</param>
    /// <returns>The aggregated provider response.</returns>
    [Activity(InvokeModelActivityName)]
    public async Task<ChatResponse> InvokeModelAsync(ModelCallInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var context = ActivityExecutionContext.Current;
        var client = ResolveClient(input.Model, context.Logger);
        var options = ChatOptionsTransport.Rebuild(input, ResolveHostedMcpServerHeaders);
        ChatOptionsTransport.RehydrateToolResults(input.Messages);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(context.CancellationToken);
        var pump = HeartbeatAsync(context, stop.Token);
        try
        {
            return input.Stream
                ? await StreamAsync(client, input, options, context)
                : await client.GetResponseAsync(input.Messages, options, context.CancellationToken);
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
            await pump;
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
            await foreach (var update in stream.WithCancellation(context.CancellationToken))
            {
                context.Heartbeat();
                topic?.Publish(update);
                updates.Add(update);
            }

            if (streamClient is not null)
            {
                await streamClient.FlushAsync(context.CancellationToken);
            }
        }
        finally
        {
            if (streamClient is not null)
            {
                await streamClient.DisposeAsync();
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
            while (await timer.WaitForNextTickAsync(stop))
            {
                context.Heartbeat();
            }
        }
        catch (OperationCanceledException)
        {
            // The provider call finished; the pump is done.
        }
    }

    private IReadOnlyDictionary<string, string>? ResolveHostedMcpServerHeaders(string server)
    {
        if (hostedMcpServerHeaderProviders is not null && hostedMcpServerHeaderProviders.TryGetValue(server, out var provider))
        {
            IReadOnlyDictionary<string, string>? headers;
            try
            {
                headers = provider();
            }
#pragma warning disable CA1031 // The provider is user code; its failure is reported once per attempt and the attempt is retried.
            catch (Exception exception)
#pragma warning restore CA1031
            {
                throw new ApplicationFailureException(
                    $"The header provider for hosted MCP server '{server}' threw {exception.GetType().Name}: {exception.Message}",
                    exception,
                    errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: false);
            }

            return headers ?? throw new ApplicationFailureException(
                $"The header provider for hosted MCP server '{server}' returned null.",
                errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                nonRetryable: true);
        }

        return hostedMcpServerHeaders is not null && hostedMcpServerHeaders.TryGetValue(server, out var fixedHeaders) ? fixedHeaders : null;
    }

    private IChatClient ResolveClient(string model, ILogger logger)
    {
        if (clients.TryGetValue(model, out var existing))
        {
            return existing;
        }

        if (!factories.TryGetValue(model, out var factory))
        {
            throw new ApplicationFailureException(
                $"No chat client is registered for model '{model}'. Add it to MicrosoftAgentFrameworkPluginOptions.ChatClients on the worker.",
                errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                nonRetryable: true);
        }

        lock (clientLock)
        {
            if (clients.TryGetValue(model, out existing))
            {
                return existing;
            }

            IChatClient? client;
            try
            {
                client = factory();
            }
#pragma warning disable CA1031 // The factory is user code; its failure is reported once per attempt and the attempt is retried.
            catch (Exception exception)
#pragma warning restore CA1031
            {
                // A missing environment variable or an unreachable key vault can be fixed in place, so
                // the attempt fails and the Activity's retry policy or schedule-to-close timeout bounds it.
                logger.LogError(exception, "The chat client factory for model '{Model}' threw; the Activity attempt fails and is retried.", model);
                throw new ApplicationFailureException(
                    $"The chat client factory for model '{model}' threw {exception.GetType().Name}: {exception.Message}",
                    exception,
                    errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: false);
            }

            if (client is null)
            {
                throw new ApplicationFailureException(
                    $"The chat client factory for model '{model}' returned null.",
                    errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: true);
            }

            if (client.GetService<FunctionInvokingChatClient>() is not null)
            {
                throw new ApplicationFailureException(
                    $"The chat client for model '{model}' contains a FunctionInvokingChatClient, which would run tools inside the model Activity. " +
                    "Register the provider client without UseFunctionInvocation(); TemporalAgent runs the tool loop in the Workflow.",
                    errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: true);
            }

            clients[model] = client;
            return client;
        }
    }
}
