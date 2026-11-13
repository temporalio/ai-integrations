using Microsoft.Extensions.AI;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class ApprovalUpdateWorkflow
{
    private readonly Dictionary<string, ToolApprovalResponseContent> decisions = new(StringComparer.Ordinal);
    private ToolApprovalRequestContent? pending;
    private string? finalText;

    [WorkflowQuery]
    public ToolApprovalRequestContent? PendingApproval() => pending;

    [WorkflowUpdateValidator(nameof(DecideAsync))]
    public void ValidateDecision(ToolApprovalResponseContent decision)
    {
        if (pending is null || pending.RequestId != decision.RequestId)
        {
            throw new ApplicationFailureException($"No pending approval with request id '{decision.RequestId}'");
        }
    }

    [WorkflowUpdate]
    public async Task<string> DecideAsync(ToolApprovalResponseContent decision)
    {
        decisions[decision.RequestId] = decision;
        await Workflow.WaitConditionAsync(() => finalText is not null);
        return finalText!;
    }

    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var sendEmail = AIFunctionFactory.Create((string to) => $"email sent to {to}", "send_email", "Sends an email");
        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions { Tools = [new ApprovalRequiredAIFunction(sendEmail)] },
        });
        var session = await agent.CreateSessionAsync();
        var response = await agent.RunAsync(prompt, session);
        var request = response.Messages.SelectMany(message => message.Contents).OfType<ToolApprovalRequestContent>().FirstOrDefault();
        if (request is not null)
        {
            pending = request;
            await Workflow.WaitConditionAsync(() => decisions.ContainsKey(request.RequestId));
            pending = null;
            response = await agent.RunAsync(new ChatMessage(ChatRole.User, [decisions[request.RequestId]]), session);
        }

        finalText = response.Text;
        await Workflow.WaitConditionAsync(() => Workflow.AllHandlersFinished);
        return finalText;
    }
}
