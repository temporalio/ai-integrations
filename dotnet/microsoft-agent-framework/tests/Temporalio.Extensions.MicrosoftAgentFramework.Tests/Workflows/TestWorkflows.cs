namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>Every test Workflow, registered on each worker and on the replayer.</summary>
public static class TestWorkflows
{
    public static readonly IReadOnlyList<Type> All =
    [
        typeof(HelloWorkflow),
        typeof(CancellationWorkflow),
        typeof(ToolsWorkflow),
        typeof(ApprovalWorkflow),
        typeof(ApprovalUpdateWorkflow),
        typeof(SessionWorkflow),
        typeof(McpWorkflow),
        typeof(FailureWorkflow),
        typeof(StreamingWorkflow),
        typeof(StructuredOutputWorkflow),
        typeof(ModelOptionsWorkflow),
        typeof(TwoAgentsWorkflow),
        typeof(HostedToolWorkflow),
        typeof(DirectChatClientWorkflow),
    ];
}
