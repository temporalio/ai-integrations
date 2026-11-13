using Temporalio.Exceptions;

namespace Temporalio.MicrosoftAgentFramework;

/// <summary>
/// Failure raised by <see cref="TemporalAgent"/> on the Workflow side. The plugin registers this
/// type through <c>WorkflowFailureExceptionTypes</c>, so throwing it fails the Workflow Execution
/// instead of failing the Workflow Task.
/// </summary>
public sealed class MicrosoftAgentFrameworkWorkflowException : ApplicationFailureException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MicrosoftAgentFrameworkWorkflowException"/> class.
    /// </summary>
    /// <param name="message">What went wrong and what to change.</param>
    /// <param name="inner">The exception that caused this one, if any.</param>
    public MicrosoftAgentFrameworkWorkflowException(string message, Exception? inner = null)
        : base(message, inner, errorType: MicrosoftAgentFrameworkErrorTypes.WorkflowError)
    {
    }
}
