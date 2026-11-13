using System.Diagnostics;
using ModelContextProtocol.Client;
using Temporalio.AgentFramework.Activities;
using Temporalio.Exceptions;

namespace Temporalio.AgentFramework.Internal;

/// <summary>
/// One <see cref="McpClient"/> per server name per worker, connected on first use and closed
/// after <see cref="AgentFrameworkPluginOptions.McpConnectionIdleTimeout"/> without a call.
/// Also holds the tool list of a connection when the caller asked for caching.
/// </summary>
internal sealed class McpConnectionCache : IAsyncDisposable
{
    private static readonly TimeSpan MaxSweepInterval = TimeSpan.FromSeconds(30);

    private readonly IReadOnlyDictionary<string, Func<IClientTransport>> factories;
    private readonly TimeSpan idleTimeout;
    private readonly Dictionary<string, Entry> entries = new(StringComparer.Ordinal);
    private readonly object sync = new();
    private CancellationTokenSource? sweepStop;
    private Task? sweeper;

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
        await entry.Gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            entry.Touch();
            if (useCache && entry.Tools is { } cached)
            {
                return cached;
            }

            var client = await entry.ConnectAsync(factories[server], cancellationToken).ConfigureAwait(false);
            var tools = await client.ListToolsAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
            var declarations = tools
                .Select(tool => new ToolDeclaration(tool.Name, tool.Description, tool.JsonSchema, tool.ReturnJsonSchema))
                .ToList();
            entry.Tools = useCache ? declarations : null;
            entry.Touch();
            return declarations;
        }
        finally
        {
            entry.Gate.Release();
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
        await entry.Gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            entry.Touch();
            client = await entry.ConnectAsync(factories[server], cancellationToken).ConfigureAwait(false);
            entry.InFlight++;
        }
        finally
        {
            entry.Gate.Release();
        }

        try
        {
            return await action(client, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await entry.Gate.WaitAsync(CancellationToken.None).ConfigureAwait(false);
            entry.InFlight--;
            entry.Touch();
            entry.Gate.Release();
        }
    }

    /// <summary>
    /// Closes every connection and stops the idle sweep. The cache stays usable: the next call
    /// reconnects and restarts the sweep, so one plugin instance can serve a worker that is
    /// stopped and started again.
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
            await sweep.ConfigureAwait(false);
        }

        foreach (var entry in all)
        {
            await entry.CloseAsync().ConfigureAwait(false);
        }

        stop?.Dispose();
    }

    private Entry GetEntry(string server)
    {
        if (!factories.ContainsKey(server))
        {
            throw new ApplicationFailureException(
                $"No MCP server is registered under '{server}'. Add it to AgentFrameworkPluginOptions.McpServers on the worker.",
                errorType: AgentFrameworkErrorTypes.ConfigurationError,
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
            while (await timer.WaitForNextTickAsync(stop).ConfigureAwait(false))
            {
                List<Entry> all;
                lock (sync)
                {
                    all = entries.Values.ToList();
                }

                foreach (var entry in all)
                {
                    await entry.CloseIfIdleAsync(idleTimeout).ConfigureAwait(false);
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

        /// <summary>Gets the gate that serializes connect, list and close.</summary>
        public SemaphoreSlim Gate { get; } = new(1, 1);

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
        /// empty so the next attempt reconnects.
        /// </summary>
        /// <param name="factory">Transport factory.</param>
        /// <param name="cancellationToken">Cancels the connect.</param>
        /// <returns>The connected client.</returns>
        public async Task<McpClient> ConnectAsync(Func<IClientTransport> factory, CancellationToken cancellationToken)
        {
            if (Client is { } existing)
            {
                return existing;
            }

            var transport = factory();
            var client = await McpClient.CreateAsync(transport, cancellationToken: cancellationToken).ConfigureAwait(false);
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
            if (!await Gate.WaitAsync(0).ConfigureAwait(false))
            {
                return;
            }

            try
            {
                if (Client is not null && InFlight == 0 && Stopwatch.GetElapsedTime(lastUsed) > idleTimeout)
                {
                    await CloseLockedAsync().ConfigureAwait(false);
                }
            }
            finally
            {
                Gate.Release();
            }
        }

        /// <summary>
        /// Closes the client.
        /// </summary>
        /// <returns>A task that completes when the client is disposed.</returns>
        public async Task CloseAsync()
        {
            await Gate.WaitAsync().ConfigureAwait(false);
            try
            {
                await CloseLockedAsync().ConfigureAwait(false);
            }
            finally
            {
                Gate.Release();
            }
        }

        private async Task CloseLockedAsync()
        {
            var client = Client;
            Client = null;
            Tools = null;
            if (client is not null)
            {
                await client.DisposeAsync().ConfigureAwait(false);
            }
        }
    }
}
