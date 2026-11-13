using Temporalio.Exceptions;

namespace Temporalio.AgentFramework;

/// <summary>
/// Failure raised by <see cref="TemporalAgent"/> on the Workflow side. The plugin registers this
/// type through <c>WorkflowFailureExceptionTypes</c>, so throwing it fails the Workflow Execution
/// instead of failing the Workflow Task.
/// </summary>
public sealed class AgentFrameworkWorkflowException : ApplicationFailureException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AgentFrameworkWorkflowException"/> class.
    /// </summary>
    /// <param name="message">What went wrong and what to change.</param>
    /// <param name="inner">The exception that caused this one, if any.</param>
    public AgentFrameworkWorkflowException(string message, Exception? inner = null)
        : base(message, inner, errorType: AgentFrameworkErrorTypes.WorkflowError)
    {
    }
}
