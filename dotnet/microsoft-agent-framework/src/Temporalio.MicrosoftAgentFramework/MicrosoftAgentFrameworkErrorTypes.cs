namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// Error types the plugin sets on <see cref="Exceptions.ApplicationFailureException.ErrorType"/>.
/// Match on these constants, never on message text.
/// </summary>
public static class MicrosoftAgentFrameworkErrorTypes
{
    /// <summary>
    /// The provider returned an HTTP error from a model call. 408, 409, 429 and 5xx are
    /// retryable and every other 4xx is not, unless the provider's <c>x-should-retry</c> header
    /// says otherwise; <c>retry-after-ms</c> or <c>Retry-After</c> sets the next retry delay.
    /// </summary>
    public const string ModelError = "MicrosoftAgentFrameworkModelError";

    /// <summary>
    /// An MCP server rejected a call at the protocol level: an unknown tool, bad arguments or an
    /// invalid request. Non-retryable. A result with <c>isError</c> is not a failure: it goes back
    /// to the model as the server returned it.
    /// </summary>
    public const string McpError = "MicrosoftAgentFrameworkMcpError";

    /// <summary>
    /// A model or MCP server name has no factory on the worker, a factory returned null, or the
    /// worker-side chat client is built in a way the plugin cannot run: not retried. A factory that
    /// threw carries the same type and is retried under the Activity's retry policy.
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
