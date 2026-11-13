using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.AgentFramework;

/// <summary>
/// Options for <see cref="TemporalAgent"/>. The agent clones them at construction.
/// </summary>
public sealed class TemporalAgentOptions
{
    private int maximumIterationsPerRun = 40;

    /// <summary>Gets or sets the agent id. Defaults to <see cref="Name"/>, then <c>temporal-agent</c>.</summary>
    public string? Id { get; set; }

    /// <summary>Gets or sets the agent name. Set as the author name of response messages that have none.</summary>
    public string? Name { get; set; }

    /// <summary>Gets or sets the agent description.</summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the chat options for every run: instructions, tools, response format and
    /// sampling settings. Tools may be plain <see cref="AIFunction"/>s (run in the Workflow),
    /// <see cref="ActivityAsTool"/> or <see cref="Mcp.TemporalMcpTool"/> (run as Activities),
    /// <see cref="ApprovalRequiredAIFunction"/> wrappers, and hosted tools.
    /// </summary>
    public ChatOptions? ChatOptions { get; set; }

    /// <summary>
    /// Gets or sets the chat history provider. Defaults to <see cref="InMemoryChatHistoryProvider"/>,
    /// whose state rides in the session and Event History. Another provider runs in the Workflow,
    /// must complete synchronously, and logs a warning because it persists next to Event History.
    /// </summary>
    public ChatHistoryProvider? ChatHistoryProvider { get; set; }

    /// <summary>
    /// Gets or sets context providers. They run in the Workflow around each run and must complete
    /// synchronously and deterministically; I/O belongs in an Activity.
    /// </summary>
    public IEnumerable<AIContextProvider>? AIContextProviders { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the tool calls of one model response run at the
    /// same time through <c>Workflow.WhenAllAsync</c>. Default false: one after the other, which is
    /// what <c>ChatClientAgent</c> does.
    /// </summary>
    public bool AllowConcurrentInvocation { get; set; }

    /// <summary>
    /// Gets or sets the number of model calls a run may make while the model keeps requesting
    /// tools. At the limit the agent makes one more call without function tools and returns its
    /// answer, as <c>FunctionInvokingChatClient</c> does. Default 40.
    /// </summary>
    public int MaximumIterationsPerRun
    {
        get
        {
            return maximumIterationsPerRun;
        }

        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            maximumIterationsPerRun = value;
        }
    }

    /// <summary>
    /// Gets or sets the options of the model Activity. Used by the constructor that takes a model
    /// name; see <see cref="TemporalChatClient"/> for the defaults.
    /// </summary>
    public ActivityOptions? ModelActivityOptions { get; set; }

    /// <summary>
    /// Gets or sets the Workflow Stream topic for streamed updates. Used by the constructor that
    /// takes a model name.
    /// </summary>
    public string? StreamingTopic { get; set; }

    /// <summary>
    /// Gets or sets the batch interval for streamed updates. Used by the constructor that takes a
    /// model name. Defaults to 100 milliseconds.
    /// </summary>
    public TimeSpan? StreamingBatchInterval { get; set; }

    /// <summary>
    /// Returns a copy; <see cref="ChatOptions"/> is cloned, providers are shared.
    /// </summary>
    /// <returns>The copy.</returns>
    public TemporalAgentOptions Clone() => new()
    {
        Id = Id,
        Name = Name,
        Description = Description,
        ChatOptions = ChatOptions?.Clone(),
        ChatHistoryProvider = ChatHistoryProvider,
        AIContextProviders = AIContextProviders?.ToList(),
        AllowConcurrentInvocation = AllowConcurrentInvocation,
        MaximumIterationsPerRun = MaximumIterationsPerRun,
        ModelActivityOptions = ModelActivityOptions is null ? null : (ActivityOptions)ModelActivityOptions.Clone(),
        StreamingTopic = StreamingTopic,
        StreamingBatchInterval = StreamingBatchInterval,
    };
}
