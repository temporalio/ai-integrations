using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

/// <summary>The README's approval Workflow: a presence-only validator, and an Update handler that judges the id once the reply has been read.</summary>
[Workflow]
public class ApprovalUpdateWorkflow
{
    private readonly Dictionary<string, ToolApprovalRequestContent> pending = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ApprovalDecision> decisions = new(StringComparer.Ordinal);
    private bool awaitingReply;

    [WorkflowQuery]
    public IReadOnlyList<ToolApprovalRequestContent> PendingApprovals() => pending.Values.ToList();

    [WorkflowUpdateValidator(nameof(DecideAsync))]
    public void ValidateDecision(ApprovalDecision decision)
    {
        // Validators run before the Workflow sees the model's reply, so this one checks the input
        // and not the state. Bad input never enters history.
        if (string.IsNullOrEmpty(decision.RequestId))
        {
            throw new ApplicationFailureException("RequestId is required");
        }
    }

    [WorkflowUpdate]
    public async Task<string> DecideAsync(ApprovalDecision decision)
    {
        // A decision can arrive in the same Workflow Task as the model's reply, before RunAsync has
        // read the requests in that reply. Judge the id once the reply has been read.
        await Workflow.WaitConditionAsync(() => !awaitingReply);
        if (decisions.ContainsKey(decision.RequestId))
        {
            throw new ApplicationFailureException($"Approval request '{decision.RequestId}' is already decided");
        }

        if (!pending.ContainsKey(decision.RequestId))
        {
            throw new ApplicationFailureException($"No pending approval request '{decision.RequestId}'");
        }

        decisions[decision.RequestId] = decision;
        return decision.Approved ? "approved" : "rejected";
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
        var response = await ReplyAsync(agent, new ChatMessage(ChatRole.User, prompt), session);
        var requests = ApprovalRequests(response);
        while (requests.Count > 0)
        {
            foreach (var request in requests)
            {
                pending[request.RequestId] = request;
            }

            await Workflow.WaitConditionAsync(() => pending.Keys.All(decisions.ContainsKey));
            var replies = requests
                .Select(request => (AIContent)request.CreateResponse(decisions[request.RequestId].Approved, decisions[request.RequestId].Reason))
                .ToList();
            pending.Clear();
            response = await ReplyAsync(agent, new ChatMessage(ChatRole.User, replies), session);
            requests = ApprovalRequests(response);
        }

        await Workflow.WaitConditionAsync(() => Workflow.AllHandlersFinished);
        return response.Text;
    }

    private static List<ToolApprovalRequestContent> ApprovalRequests(AgentResponse response) =>
        response.Messages.SelectMany(message => message.Contents).OfType<ToolApprovalRequestContent>().ToList();

    private async Task<AgentResponse> ReplyAsync(TemporalAgent agent, ChatMessage message, AgentSession session)
    {
        awaitingReply = true;
        try
        {
            return await agent.RunAsync(message, session);
        }
        finally
        {
            awaitingReply = false;
        }
    }
}
