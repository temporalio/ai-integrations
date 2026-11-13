namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// Error types the plugin sets on <see cref="Exceptions.ApplicationFailureException.ErrorType"/>.
/// Match on these constants, never on message text.
/// </summary>
public static class MicrosoftAgentFrameworkErrorTypes
{
    /// <summary>
    /// The provider returned an HTTP error from a model call. The failure is non-retryable for
    /// 400, 401, 403, 404 and 422 and retryable for every other status.
    /// </summary>
    public const string ModelError = "MicrosoftAgentFrameworkModelError";

    /// <summary>
    /// An MCP tool call returned <c>isError</c>, or the server does not know the tool. Non-retryable.
    /// </summary>
    public const string McpError = "MicrosoftAgentFrameworkMcpError";

    /// <summary>
    /// A model or MCP server name has no factory on the worker, or the worker-side chat client is
    /// built in a way the plugin cannot run. Non-retryable.
    /// </summary>
    public const string ConfigurationError = "MicrosoftAgentFrameworkConfigurationError";

    /// <summary>
    /// <see cref="TemporalAgent"/> found a problem on the Workflow side: an approval request the
    /// next run did not answer, an approval response for a request the session never surfaced,
    /// unsupported run options, a session of another type, a chat history or context provider
    /// whose <c>InvokingAsync</c> or <c>InvokedAsync</c> returned an incomplete <c>ValueTask</c>
    /// or scheduled an Activity, a timer or a child Workflow, a fourth consecutive round of
    /// failing tools, or a call through the <see cref="Microsoft.Agents.AI.AIAgent"/> base entry
    /// points. Fails the Workflow.
    /// </summary>
    public const string WorkflowError = "MicrosoftAgentFrameworkWorkflowError";
}
