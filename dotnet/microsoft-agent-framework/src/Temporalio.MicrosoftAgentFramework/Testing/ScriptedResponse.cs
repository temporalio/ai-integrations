using System.Net;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Testing;

/// <summary>
/// One scripted answer of a <see cref="FakeChatClient"/>: text, tool calls, JSON, or a failure.
/// </summary>
public sealed class ScriptedResponse
{
    private readonly Func<int, ChatResponse> produce;
    private readonly TimeSpan delay;

    private ScriptedResponse(Func<int, ChatResponse> produce, TimeSpan delay, TimeSpan streamingDelay)
    {
        this.produce = produce;
        this.delay = delay;
        StreamingDelay = streamingDelay;
    }

    /// <summary>Gets the pause before each streamed update; zero by default.</summary>
    internal TimeSpan StreamingDelay { get; }

    /// <summary>
    /// A plain assistant message.
    /// </summary>
    /// <param name="text">The text.</param>
    /// <returns>The scripted response.</returns>
    public static ScriptedResponse Text(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return new(call => Build(call, [new TextContent(text)], ChatFinishReason.Stop), TimeSpan.Zero, TimeSpan.Zero);
    }

    /// <summary>
    /// An assistant message that requests tool calls. Call ids are <c>call_{call}_{index}</c>.
    /// </summary>
    /// <param name="calls">Tool name and arguments per call.</param>
    /// <returns>The scripted response.</returns>
    public static ScriptedResponse ToolCalls(params (string Name, IDictionary<string, object?> Arguments)[] calls)
    {
        ArgumentNullException.ThrowIfNull(calls);
        return new(
            call => Build(
                call,
                calls.Select((tool, index) => (AIContent)new FunctionCallContent($"call_{call}_{index}", tool.Name, tool.Arguments)).ToList(),
                ChatFinishReason.ToolCalls),
            TimeSpan.Zero,
            TimeSpan.Zero);
    }

    /// <summary>
    /// An assistant message whose text is <paramref name="value"/> serialized as JSON, for
    /// structured output.
    /// </summary>
    /// <param name="value">The value.</param>
    /// <returns>The scripted response.</returns>
    public static ScriptedResponse Json(object value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var json = JsonSerializer.Serialize(value, AIJsonUtilities.DefaultOptions);
        return new(call => Build(call, [new TextContent(json)], ChatFinishReason.Stop), TimeSpan.Zero, TimeSpan.Zero);
    }

    /// <summary>
    /// A provider failure: throws <see cref="HttpRequestException"/> with <paramref name="status"/>,
    /// which the model Activity classifies the way it classifies a real provider error.
    /// </summary>
    /// <param name="status">HTTP status to fail with.</param>
    /// <param name="message">Exception message.</param>
    /// <returns>The scripted response.</returns>
    public static ScriptedResponse Fail(HttpStatusCode status, string? message = null) =>
        new(_ => throw new HttpRequestException(message ?? $"Scripted provider failure {(int)status}", null, status), TimeSpan.Zero, TimeSpan.Zero);

    /// <summary>
    /// Returns a copy that waits <paramref name="delay"/> before answering, to stand in for a slow provider.
    /// </summary>
    /// <param name="delay">How long to wait.</param>
    /// <returns>The delayed response.</returns>
    public ScriptedResponse WithDelay(TimeSpan delay) => new(produce, delay, StreamingDelay);

    /// <summary>
    /// Returns a copy whose streamed updates arrive <paramref name="delay"/> apart, to stand in for a provider that streams slowly.
    /// </summary>
    /// <param name="delay">The pause before each update.</param>
    /// <returns>The response.</returns>
    public ScriptedResponse WithStreamingDelay(TimeSpan delay) => new(produce, this.delay, delay);

    /// <summary>
    /// Produces the response for call number <paramref name="call"/>.
    /// </summary>
    /// <param name="call">1-based call number.</param>
    /// <param name="cancellationToken">Cancels the delay.</param>
    /// <returns>The response.</returns>
    internal async Task<ChatResponse> ProduceAsync(int call, CancellationToken cancellationToken)
    {
        if (delay > TimeSpan.Zero)
        {
            await Task.Delay(delay, cancellationToken);
        }

        return produce(call);
    }

    private static ChatResponse Build(int call, IList<AIContent> contents, ChatFinishReason finishReason) =>
        new(new ChatMessage(ChatRole.Assistant, contents) { MessageId = $"msg_{call}" })
        {
            ResponseId = $"resp_{call}",
            ModelId = "fake",
            FinishReason = finishReason,
            Usage = new UsageDetails { InputTokenCount = 10, OutputTokenCount = 5, TotalTokenCount = 15 },
        };
}
