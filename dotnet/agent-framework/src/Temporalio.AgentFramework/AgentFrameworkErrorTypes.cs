namespace Temporalio.AgentFramework;

/// <summary>
/// Error types the plugin sets on <see cref="Exceptions.ApplicationFailureException.ErrorType"/>.
/// Match on these constants, never on message text.
/// </summary>
public static class AgentFrameworkErrorTypes
{
    /// <summary>
    /// The provider returned an HTTP error from a model call. The failure is non-retryable for
    /// 400, 401, 403, 404 and 422 and retryable for every other status.
    /// </summary>
    public const string ModelError = "AgentFrameworkModelError";

    /// <summary>
    /// An MCP tool call returned <c>isError</c>, or the server does not know the tool. Non-retryable.
    /// </summary>
    public const string McpError = "AgentFrameworkMcpError";

    /// <summary>
    /// A model or MCP server name has no factory on the worker, or the worker-side chat client is
    /// built in a way the plugin cannot run. Non-retryable.
    /// </summary>
    public const string ConfigurationError = "AgentFrameworkConfigurationError";

    /// <summary>
    /// <see cref="TemporalAgent"/> found a problem on the Workflow side: an unanswered approval
    /// request, unsupported run options, a session of another type, a provider that did not
    /// complete synchronously, too many consecutive tool failures, or a call through the
    /// <see cref="Microsoft.Agents.AI.AIAgent"/> base entry points. Fails the Workflow.
    /// </summary>
    public const string WorkflowError = "AgentFrameworkWorkflowError";
}
