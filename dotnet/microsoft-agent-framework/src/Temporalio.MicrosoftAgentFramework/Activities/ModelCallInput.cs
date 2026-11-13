using Microsoft.Extensions.AI;

namespace Temporalio.MicrosoftAgentFramework.Activities;

/// <summary>
/// Input of the <c>MicrosoftAgentFramework.InvokeModel</c> Activity: one model call.
/// </summary>
public sealed record ModelCallInput
{
    /// <summary>Gets the model name; the worker resolves it against <see cref="MicrosoftAgentFrameworkPluginOptions.ChatClients"/>.</summary>
    public required string Model { get; init; }

    /// <summary>Gets the messages to send.</summary>
    public required IReadOnlyList<ChatMessage> Messages { get; init; }

    /// <summary>
    /// Gets the chat options without tools, <c>RawRepresentationFactory</c> and
    /// <c>ContinuationToken</c>, which do not cross the Activity boundary.
    /// </summary>
    public ChatOptions? Options { get; init; }

    /// <summary>Gets the function tools to advertise, as declarations.</summary>
    public IReadOnlyList<ToolDeclaration>? Tools { get; init; }

    /// <summary>Gets the provider-executed tools to pass through.</summary>
    public IReadOnlyList<HostedToolDeclaration>? HostedTools { get; init; }

    /// <summary>
    /// Gets a value indicating whether the Activity consumes the provider's streaming response.
    /// The Activity heartbeats on every update and returns the aggregated response either way.
    /// </summary>
    public bool Stream { get; init; }

    /// <summary>
    /// Gets the Workflow Stream topic that receives each <see cref="ChatResponseUpdate"/> as the
    /// provider produces it, or null to publish nothing.
    /// </summary>
    public string? StreamingTopic { get; init; }

    /// <summary>Gets how long updates are batched before one publish signal carries them.</summary>
    public TimeSpan? StreamingBatchInterval { get; init; }
}
