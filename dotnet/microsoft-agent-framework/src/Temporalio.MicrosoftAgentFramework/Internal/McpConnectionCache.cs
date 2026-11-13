using System.Diagnostics;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using Temporalio.Activities;
using Temporalio.Exceptions;
using Temporalio.MicrosoftAgentFramework.Activities;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// One <see cref="McpClient"/> per server name per plugin instance, shared by every worker running
/// with it, connected on first use and closed after
/// <see cref="MicrosoftAgentFrameworkPluginOptions.McpConnectionIdleTimeout"/> without a call or when
/// the last worker or replayer running with the plugin stops (<see cref="Acquire"/> and
/// <see cref="ReleaseAsync"/> count them). Also holds the tool list of a connection when the caller
/// asked for caching.
/// </summary>
internal sealed class McpConnectionCache : IAsyncDisposable
{
    private static readonly TimeSpan MaxSweepInterval = TimeSpan.FromSeconds(30);
    private static readonly TimeSpan DisposeTimeout = TimeSpan.FromSeconds(10);
    private static readonly TimeSpan InFlightPollInterval = TimeSpan.FromMilliseconds(20);

    private readonly IReadOnlyDictionary<string, Func<IClientTransport>> factories;
    private readonly TimeSpan idleTimeout;
    private readonly Dictionary<string, Entry> entries = new(StringComparer.Ordinal);
    private readonly object sync = new();
    private CancellationTokenSource? sweepStop;
    private Task? sweeper;
    private int runs;

    /// <summary>
    /// Initializes a new instance of the <see cref="McpConnectionCache"/> class.
    /// </summary>
    /// <param name="factories">Transport factories by server name.</param>
    /// <param name="idleTimeout">How long a connection may sit unused.</param>
    public McpConnectionCache(IReadOnlyDictionary<string, Func<IClientTransport>> factories, TimeSpan idleTimeout)
    {
        this.factories = factories;
        this.idleTimeout = idleTimeout;
    }

    /// <summary>
    /// Lists the tools of a server.
    /// </summary>
    /// <param name="server">Server name.</param>
    /// <param name="useCache">Whether a list fetched earlier on the same connection may be reused.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The declarations.</returns>
    public async Task<IReadOnlyList<ToolDeclaration>> ListToolsAsync(string server, bool useCache, CancellationToken cancellationToken)
    {
        var entry = GetEntry(server);
        await entry.Lock.WaitAsync(cancellationToken);
        try
        {
            entry.Touch();
            if (useCache && entry.Tools is { } cached)
            {
                return cached;
            }

            var client = await entry.ConnectAsync(server, factories[server], cancellationToken);
            var tools = await client.ListToolsAsync(cancellationToken: cancellationToken);
            var declarations = tools
                .Select(tool => new ToolDeclaration(tool.Name, tool.Description, tool.JsonSchema, tool.ReturnJsonSchema))
                .ToList();
            entry.Tools = useCache ? declarations : null;
            entry.Touch();
            return declarations;
        }
        finally
        {
            entry.Lock.Release();
        }
    }

    /// <summary>
    /// Runs an action against a server's client, connecting first when needed.
    /// </summary>
    /// <typeparam name="T">Result type.</typeparam>
    /// <param name="server">Server name.</param>
    /// <param name="action">Action to run with the connected client.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The action result.</returns>
    public async Task<T> UseAsync<T>(string server, Func<McpClient, CancellationToken, Task<T>> action, CancellationToken cancellationToken)
    {
        var entry = GetEntry(server);
        McpClient client;
        await entry.Lock.WaitAsync(cancellationToken);
        try
        {
            entry.Touch();
            client = await entry.ConnectAsync(server, factories[server], cancellationToken);
            entry.InFlight++;
        }
        finally
        {
            entry.Lock.Release();
        }

        try
        {
            return await action(client, cancellationToken);
        }
        finally
        {
            await entry.Lock.WaitAsync(CancellationToken.None);
            entry.InFlight--;
            entry.Touch();
            entry.Lock.Release();
        }
    }

    /// <summary>
    /// Counts a worker or replayer that starts running with the plugin. Pairs with <see cref="ReleaseAsync"/>.
    /// </summary>
    public void Acquire()
    {
        lock (sync)
        {
            runs++;
        }
    }

    /// <summary>
    /// Counts a worker or replayer that stopped. When it was the last one running, closes every
    /// connection through <see cref="DisposeAsync"/>; while others run, the connections stay open
    /// for their calls.
    /// </summary>
    /// <returns>A task that completes when the connections are closed, or at once when others still run.</returns>
    public async ValueTask ReleaseAsync()
    {
        lock (sync)
        {
            runs = Math.Max(0, runs - 1);
            if (runs > 0)
            {
                return;
            }
        }

        await DisposeAsync();
    }

    /// <summary>
    /// Closes every connection and stops the idle sweep, waiting up to 10 seconds per connection
    /// for a call in flight. The cache stays usable: the next call reconnects and restarts the
    /// sweep, so one plugin instance can serve a worker that is stopped and started again.
    /// </summary>
    /// <returns>A task that completes when every client is disposed.</returns>
    public async ValueTask DisposeAsync()
    {
        Task? sweep;
        CancellationTokenSource? stop;
        List<Entry> all;
        lock (sync)
        {
            sweep = sweeper;
            stop = sweepStop;
            sweeper = null;
            sweepStop = null;
            all = entries.Values.ToList();
        }

        if (stop is not null)
        {
#pragma warning disable CA1849 // Cancel is the form the Temporal .NET SDK recommends.
            stop.Cancel();
#pragma warning restore CA1849
        }

        if (sweep is not null)
        {
            await sweep;
        }

        foreach (var entry in all)
        {
            await entry.CloseAsync();
        }

        stop?.Dispose();
    }

    private Entry GetEntry(string server)
    {
        if (!factories.ContainsKey(server))
        {
            throw new ApplicationFailureException(
                $"No MCP server is registered under '{server}'. Add it to MicrosoftAgentFrameworkPluginOptions.McpServers on the worker.",
                errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                nonRetryable: true);
        }

        lock (sync)
        {
            if (!entries.TryGetValue(server, out var entry))
            {
                entry = new Entry();
                entries[server] = entry;
            }

            if (sweeper is null)
            {
                var stop = new CancellationTokenSource();
                sweepStop = stop;
                sweeper = Task.Run(() => SweepAsync(stop.Token));
            }

            return entry;
        }
    }

    private async Task SweepAsync(CancellationToken stop)
    {
        var half = TimeSpan.FromTicks(Math.Max(idleTimeout.Ticks / 2, TimeSpan.FromMilliseconds(50).Ticks));
        using var timer = new PeriodicTimer(half < MaxSweepInterval ? half : MaxSweepInterval);
        try
        {
            while (await timer.WaitForNextTickAsync(stop))
            {
                List<Entry> all;
                lock (sync)
                {
                    all = entries.Values.ToList();
                }

                foreach (var entry in all)
                {
                    try
                    {
                        await entry.CloseIfIdleAsync(idleTimeout);
                    }
#pragma warning disable CA1031 // One connection's close failure must not stop the sweep of the others.
                    catch (Exception) when (!stop.IsCancellationRequested)
#pragma warning restore CA1031
                    {
                        // The entry stays open and is retried on the next sweep.
                    }
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Disposal stops the sweep.
        }
    }

    /// <summary>State of one server connection.</summary>
    private sealed class Entry
    {
        private long lastUsed = Stopwatch.GetTimestamp();

        /// <summary>Gets the lock that serializes connect, list and close.</summary>
        public SemaphoreSlim Lock { get; } = new(1, 1);

        /// <summary>Gets or sets the connected client, or null.</summary>
        public McpClient? Client { get; set; }

        /// <summary>Gets or sets the cached tool list for this connection, or null.</summary>
        public IReadOnlyList<ToolDeclaration>? Tools { get; set; }

        /// <summary>Gets or sets the number of calls using the client right now.</summary>
        public int InFlight { get; set; }

        /// <summary>Records a use.</summary>
        public void Touch() => lastUsed = Stopwatch.GetTimestamp();

        /// <summary>
        /// Returns the client, connecting when there is none. A failed connect leaves the entry
        /// empty so the next attempt reconnects. A factory that throws fails the attempt, which the
        /// Activity's retry policy bounds; a factory that returns null is a configuration error and
        /// is not retried.
        /// </summary>
        /// <param name="server">Server name, for the error message.</param>
        /// <param name="factory">Transport factory.</param>
        /// <param name="cancellationToken">Cancels the connect.</param>
        /// <returns>The connected client.</returns>
        public async Task<McpClient> ConnectAsync(string server, Func<IClientTransport> factory, CancellationToken cancellationToken)
        {
            if (Client is { } existing)
            {
                return existing;
            }

            IClientTransport? transport;
            try
            {
                transport = factory();
            }
#pragma warning disable CA1031 // The factory is user code; its failure is reported once per attempt and the attempt is retried.
            catch (Exception exception)
#pragma warning restore CA1031
            {
                if (ActivityExecutionContext.HasCurrent)
                {
                    ActivityExecutionContext.Current.Logger.LogError(exception, "The transport factory for MCP server '{Server}' threw; the Activity attempt fails and is retried.", server);
                }

                throw new ApplicationFailureException(
                    $"The transport factory for MCP server '{server}' threw {exception.GetType().Name}: {exception.Message}",
                    exception,
                    errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: false);
            }

            if (transport is null)
            {
                throw new ApplicationFailureException(
                    $"The transport factory for MCP server '{server}' returned null.",
                    errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
                    nonRetryable: true);
            }

            var client = await McpClient.CreateAsync(transport, cancellationToken: cancellationToken);
            Client = client;
            Tools = null;
            return client;
        }

        /// <summary>
        /// Closes the client when it has been idle for longer than <paramref name="idleTimeout"/>
        /// and no call is using it.
        /// </summary>
        /// <param name="idleTimeout">Idle threshold.</param>
        /// <returns>A task that completes when the check, and any close, is done.</returns>
        public async Task CloseIfIdleAsync(TimeSpan idleTimeout)
        {
            if (!await Lock.WaitAsync(0))
            {
                return;
            }

            try
            {
                if (Client is not null && InFlight == 0 && Stopwatch.GetElapsedTime(lastUsed) > idleTimeout)
                {
                    await CloseLockedAsync();
                }
            }
            finally
            {
                Lock.Release();
            }
        }

        /// <summary>
        /// Closes the client once no call is using it, waiting up to <see cref="DisposeTimeout"/>
        /// for calls in flight.
        /// </summary>
        /// <returns>A task that completes when the client is disposed.</returns>
        public async Task CloseAsync()
        {
            var started = Stopwatch.GetTimestamp();
            while (true)
            {
                await Lock.WaitAsync();
                try
                {
                    if (InFlight == 0 || Stopwatch.GetElapsedTime(started) > DisposeTimeout)
                    {
                        await CloseLockedAsync();
                        return;
                    }
                }
                finally
                {
                    Lock.Release();
                }

                await Task.Delay(InFlightPollInterval);
            }
        }

        private async Task CloseLockedAsync()
        {
            var client = Client;
            Client = null;
            Tools = null;
            if (client is null)
            {
                return;
            }

            try
            {
                // A transport whose read loop never ends would otherwise hold this entry's lock and
                // block the next call on the same server, or the worker's shutdown.
                await client.DisposeAsync().AsTask().WaitAsync(DisposeTimeout);
            }
            catch (TimeoutException)
            {
                // The client is dropped; its transport finishes on its own or with the process.
            }
        }
    }
}
