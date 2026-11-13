using System.Threading.Channels;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Temporalio.MicrosoftAgentFramework.Testing;

/// <summary>
/// An MCP server that runs in the test process over in-memory channels. Register
/// <see cref="ClientTransport"/> in <see cref="MicrosoftAgentFrameworkPluginOptions.McpServers"/>; every
/// connection the plugin opens gets its own server session. Counters show how often the plugin
/// connected, listed tools and called tools.
/// </summary>
public sealed class InMemoryMcpServer : IAsyncDisposable
{
    private readonly McpServerTool[] tools;
    private readonly List<Session> sessions = [];
    private readonly object sync = new();
    private int connectionCount;
    private int listToolsRequestCount;
    private int callToolRequestCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="InMemoryMcpServer"/> class.
    /// </summary>
    /// <param name="tools">Tools to serve, for example <c>McpServerTool.Create((int a, int b) =&gt; a + b, new() { Name = "add" })</c>.</param>
    public InMemoryMcpServer(params McpServerTool[] tools)
    {
        ArgumentNullException.ThrowIfNull(tools);
        this.tools = tools;
        ClientTransport = new Transport(this);
    }

    /// <summary>Gets the transport to register on the plugin.</summary>
    public IClientTransport ClientTransport { get; }

    /// <summary>Gets the number of connections opened so far.</summary>
    public int ConnectionCount
    {
        get
        {
            lock (sync)
            {
                return connectionCount;
            }
        }
    }

    /// <summary>Gets the number of <c>tools/list</c> requests received so far.</summary>
    public int ListToolsRequestCount
    {
        get
        {
            lock (sync)
            {
                return listToolsRequestCount;
            }
        }
    }

    /// <summary>Gets the number of <c>tools/call</c> requests received so far.</summary>
    public int CallToolRequestCount
    {
        get
        {
            lock (sync)
            {
                return callToolRequestCount;
            }
        }
    }

    /// <summary>
    /// Ends every open session, as a server restart would, while new connections keep working.
    /// A client that was connected to one of them fails on its next call.
    /// </summary>
    /// <returns>A task that completes when every open session is closed.</returns>
    public async Task CloseSessionsAsync()
    {
        List<Session> open;
        lock (sync)
        {
            open = sessions.ToList();
            sessions.Clear();
        }

        foreach (var session in open)
        {
            await session.DisposeAsync();
        }
    }

    /// <summary>
    /// Ends every session and waits for the servers to stop.
    /// </summary>
    /// <returns>A task that completes when every session is closed.</returns>
    public async ValueTask DisposeAsync() => await CloseSessionsAsync();

    private void Count(JsonRpcMessage message)
    {
        if (message is not JsonRpcRequest request)
        {
            return;
        }

        lock (sync)
        {
            switch (request.Method)
            {
                case "tools/list":
                    listToolsRequestCount++;
                    break;
                case "tools/call":
                    callToolRequestCount++;
                    break;
                default:
                    break;
            }
        }
    }

    private ChannelTransport Connect()
    {
#pragma warning disable CA2000 // The session owns the server end and the server and disposes both.
        var toServer = Channel.CreateUnbounded<JsonRpcMessage>();
        var toClient = Channel.CreateUnbounded<JsonRpcMessage>();
        var serverEnd = new ChannelTransport(toServer.Reader, toClient.Writer, toServer.Writer, null);
        var clientEnd = new ChannelTransport(toClient.Reader, toServer.Writer, toClient.Writer, Count);
        var options = new McpServerOptions
        {
            ServerInfo = new Implementation { Name = "in-memory", Version = "1.0.0" },
            ToolCollection = [.. tools],
        };
        var server = McpServer.Create(serverEnd, options);
        var session = new Session(server, toServer.Writer, toClient.Writer);
#pragma warning restore CA2000
        lock (sync)
        {
            connectionCount++;
            sessions.Add(session);
        }

        return clientEnd;
    }

    /// <summary>A client transport that opens a new server session per connection.</summary>
    private sealed class Transport : IClientTransport
    {
        private readonly InMemoryMcpServer owner;

        /// <summary>Initializes a new instance of the <see cref="Transport"/> class.</summary>
        /// <param name="owner">The server.</param>
        public Transport(InMemoryMcpServer owner) => this.owner = owner;

        /// <inheritdoc />
        public string Name => "in-memory";

        /// <inheritdoc />
        public Task<ITransport> ConnectAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult<ITransport>(owner.Connect());
    }

    /// <summary>
    /// One end of a channel pair. Disposing either end completes both channels, so the message
    /// loops on both sides finish and <c>McpClient.DisposeAsync</c> returns.
    /// </summary>
    private sealed class ChannelTransport : ITransport
    {
        private readonly ChannelWriter<JsonRpcMessage> outgoing;
        private readonly ChannelWriter<JsonRpcMessage> incomingWriter;
        private readonly Action<JsonRpcMessage>? onSend;

        /// <summary>Initializes a new instance of the <see cref="ChannelTransport"/> class.</summary>
        /// <param name="incoming">Messages for this end.</param>
        /// <param name="outgoing">Messages to the other end.</param>
        /// <param name="incomingWriter">Writer of the incoming channel, completed on dispose.</param>
        /// <param name="onSend">Called for every message sent through this end.</param>
        public ChannelTransport(
            ChannelReader<JsonRpcMessage> incoming,
            ChannelWriter<JsonRpcMessage> outgoing,
            ChannelWriter<JsonRpcMessage> incomingWriter,
            Action<JsonRpcMessage>? onSend)
        {
            MessageReader = incoming;
            this.outgoing = outgoing;
            this.incomingWriter = incomingWriter;
            this.onSend = onSend;
        }

        /// <inheritdoc />
        public string? SessionId => null;

        /// <inheritdoc />
        public ChannelReader<JsonRpcMessage> MessageReader { get; }

        /// <inheritdoc />
        public Task SendMessageAsync(JsonRpcMessage message, CancellationToken cancellationToken = default)
        {
            onSend?.Invoke(message);
            return outgoing.WriteAsync(message, cancellationToken).AsTask();
        }

        /// <inheritdoc />
        public ValueTask DisposeAsync()
        {
            outgoing.TryComplete();
            incomingWriter.TryComplete();
            return default;
        }
    }

    /// <summary>A running server session.</summary>
    private sealed class Session : IAsyncDisposable
    {
        private readonly McpServer server;
        private readonly ChannelWriter<JsonRpcMessage> toServer;
        private readonly ChannelWriter<JsonRpcMessage> toClient;
        private readonly CancellationTokenSource stop = new();
        private readonly Task run;

        /// <summary>Initializes a new instance of the <see cref="Session"/> class and starts the server.</summary>
        /// <param name="server">The server.</param>
        /// <param name="toServer">Writer the client side uses; completed on close.</param>
        /// <param name="toClient">Writer the server side uses; completed on close.</param>
        public Session(McpServer server, ChannelWriter<JsonRpcMessage> toServer, ChannelWriter<JsonRpcMessage> toClient)
        {
            this.server = server;
            this.toServer = toServer;
            this.toClient = toClient;
            run = server.RunAsync(stop.Token);
        }

        /// <summary>Closes the channels, stops the server and disposes it.</summary>
        /// <returns>A task that completes when the server has stopped.</returns>
        public async ValueTask DisposeAsync()
        {
            toServer.TryComplete();
            toClient.TryComplete();
#pragma warning disable CA1849 // Cancel is the form the Temporal .NET SDK recommends.
            stop.Cancel();
#pragma warning restore CA1849
            try
            {
                await run.WaitAsync(TimeSpan.FromSeconds(5));
            }
            catch (OperationCanceledException)
            {
                // Stopping the server is the point.
            }
            catch (TimeoutException)
            {
                // The server did not stop in time; disposal below releases what it can.
            }

            await server.DisposeAsync();
            stop.Dispose();
        }
    }
}
