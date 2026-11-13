using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Testing;

/// <summary>
/// A deterministic <see cref="IChatClient"/> for tests. Register it in
/// <see cref="MicrosoftAgentFrameworkPluginOptions.ChatClients"/> and the test needs no provider
/// credentials. Streaming yields the scripted text a few characters at a time.
/// </summary>
public sealed partial class FakeChatClient : IChatClient
{
    private readonly ScriptedResponse[] script;
    private readonly Func<FakeChatClientCall, ChatResponse>? responder;
    private readonly List<FakeChatClientCall> calls = [];
    private readonly object sync = new();
    private int callCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="FakeChatClient"/> class that answers from a
    /// script, one entry per call; the last entry repeats.
    /// </summary>
    /// <param name="script">The scripted responses.</param>
    public FakeChatClient(params ScriptedResponse[] script)
    {
        ArgumentNullException.ThrowIfNull(script);
        if (script.Length == 0)
        {
            throw new ArgumentException("The script needs at least one response.", nameof(script));
        }

        this.script = script;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FakeChatClient"/> class that computes each
    /// answer from the call.
    /// </summary>
    /// <param name="responder">Function of the call.</param>
    public FakeChatClient(Func<FakeChatClientCall, ChatResponse> responder)
    {
        ArgumentNullException.ThrowIfNull(responder);
        this.responder = responder;
        script = [];
    }

    /// <summary>Gets the calls received so far, oldest first.</summary>
    public IReadOnlyList<FakeChatClientCall> Calls
    {
        get
        {
            lock (sync)
            {
                return calls.ToList();
            }
        }
    }

    /// <summary>Gets the number of calls received so far.</summary>
    public int CallCount
    {
        get
        {
            lock (sync)
            {
                return callCount;
            }
        }
    }

    /// <inheritdoc />
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (response, _) = await ProduceAsync(messages, options, cancellationToken);
        return response;
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var (response, entry) = await ProduceAsync(messages, options, cancellationToken);
        var pause = entry?.StreamingDelay ?? TimeSpan.Zero;
        for (var m = 0; m < response.Messages.Count; m++)
        {
            var message = response.Messages[m];
            var last = m == response.Messages.Count - 1;
            var pieces = new List<AIContent>();
            foreach (var content in message.Contents)
            {
                if (content is TextContent text)
                {
                    pieces.AddRange(ChunkPattern().Split(text.Text).Where(chunk => chunk.Length > 0).Select(chunk => new TextContent(chunk)));
                }
                else
                {
                    pieces.Add(content);
                }
            }

            for (var i = 0; i < pieces.Count; i++)
            {
                if (pause > TimeSpan.Zero)
                {
                    await Task.Delay(pause, cancellationToken);
                }

                yield return new ChatResponseUpdate(message.Role, [pieces[i]])
                {
                    ResponseId = response.ResponseId,
                    MessageId = message.MessageId,
                    ModelId = response.ModelId,
                    FinishReason = last && i == pieces.Count - 1 ? response.FinishReason : null,
                };
            }
        }

        if (response.Usage is { } usage)
        {
            yield return new ChatResponseUpdate(null, [new UsageContent(usage)]) { ResponseId = response.ResponseId };
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
            return new ChatClientMetadata("fake", null, "fake");
        }

        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <summary>Does nothing; the client holds no resources.</summary>
    public void Dispose()
    {
    }

    [GeneratedRegex(@"(?<=\s)")]
    private static partial Regex ChunkPattern();

    private async Task<(ChatResponse Response, ScriptedResponse? Entry)> ProduceAsync(IEnumerable<ChatMessage> messages, ChatOptions? options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var call = new FakeChatClientCall(messages.ToList(), options?.Clone());
        int number;
        lock (sync)
        {
            calls.Add(call);
            number = ++callCount;
        }

        if (responder is not null)
        {
            return (responder(call), null);
        }

        var entry = script[Math.Min(number, script.Length) - 1];
        return (await entry.ProduceAsync(number, cancellationToken), entry);
    }
}
