using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class ApprovalWorkflow
{
    private readonly Dictionary<string, ToolApprovalResponseContent> decisions = new(StringComparer.Ordinal);
    private readonly List<ToolApprovalRequestContent> pending = [];
    private int workflowToolInvocations;
    private int plainToolInvocations;
    private int modelFailures;

    [WorkflowSignal]
    public Task DecideAsync(ToolApprovalResponseContent decision)
    {
        decisions[decision.RequestId] = decision;
        return Task.CompletedTask;
    }

    [WorkflowQuery]
    public ToolApprovalRequestContent? PendingApproval() => pending.FirstOrDefault();

    [WorkflowQuery]
    public IReadOnlyList<ToolApprovalRequestContent> PendingApprovals() => pending.ToList();

    [WorkflowQuery]
    public int WorkflowToolInvocations() => workflowToolInvocations;

    [WorkflowQuery]
    public int PlainToolInvocations() => plainToolInvocations;

    [WorkflowQuery]
    public int ModelFailures() => modelFailures;

    [WorkflowRun]
    public async Task<string> RunAsync(ApprovalInput input)
    {
        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions { ChatOptions = new ChatOptions { Tools = BuildTools(input) } });
        var session = input.Session is { } carried ? await agent.DeserializeSessionAsync(carried) : await agent.CreateSessionAsync();
        AgentResponse? response = null;
        List<ToolApprovalRequestContent> requests;
        if (input.PendingRequest is { } carriedRequest)
        {
            requests = [carriedRequest];
        }
        else
        {
            response = await agent.RunAsync(input.Prompt, session);
            requests = ApprovalRequests(response);
        }

        while (requests.Count > 0)
        {
            if (input.ContinueAsNew && input.PendingRequest is null)
            {
                // Carry the session and the pending request into a new run before waiting for the decision.
                var next = input with { Session = await agent.SerializeSessionAsync(session), PendingRequest = requests[0] };
                throw Workflow.CreateContinueAsNewException((ApprovalWorkflow wf) => wf.RunAsync(next));
            }

            // Every request of the round waits for its decision; the decisions go back in one message.
            pending.AddRange(requests);
            var ids = requests.Select(request => request.RequestId).ToList();
            await Workflow.WaitConditionAsync(() => ids.All(decisions.ContainsKey));
            pending.Clear();
            var contents = ids.Select(id => (AIContent)decisions[id]).ToList();
            if (input.DecisionWithText)
            {
                contents.Add(new TextContent("Also tell me a joke."));
            }

            response = await RunWithDecisionAsync(agent, session, new ChatMessage(ChatRole.User, contents), input.RetryAfterModelFailure);
            requests = ApprovalRequests(response);
        }

        return response!.Text;
    }

    private static List<ToolApprovalRequestContent> ApprovalRequests(AgentResponse response) =>
        response.Messages.SelectMany(message => message.Contents).OfType<ToolApprovalRequestContent>().ToList();

    private async Task<AgentResponse> RunWithDecisionAsync(TemporalAgent agent, AgentSession session, ChatMessage decision, bool retry)
    {
        try
        {
            return await agent.RunAsync(decision, session);
        }
        catch (ActivityFailureException) when (retry)
        {
            // The model call after the approved tool failed for good. The failed run stored nothing,
            // so the same run, with the same decision, can be made again.
            modelFailures++;
            return await agent.RunAsync(decision, session);
        }
    }

    private List<AITool> BuildTools(ApprovalInput input)
    {
        var toolOptions = new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(30) };
        AIFunction sendEmail = input.ToolKind == "activity"
            ? ActivityAsTool.Create((WeatherActivities a) => a.SendEmailAsync(default!), toolOptions, "send_email", "Sends an email")
            : AIFunctionFactory.Create(
                (string to) =>
                {
                    workflowToolInvocations++;
                    return $"email sent to {to}";
                },
                "send_email",
                "Sends an email");
        var tools = new List<AITool> { new ApprovalRequiredAIFunction(sendEmail) };
        if (input.TwoApprovalTools)
        {
            tools.Add(new ApprovalRequiredAIFunction(AIFunctionFactory.Create(
                (string path) =>
                {
                    workflowToolInvocations++;
                    return $"deleted {path}";
                },
                "delete_file",
                "Deletes a file")));
        }

        if (input.WithPlainTool)
        {
            tools.Add(AIFunctionFactory.Create(
                (string city) =>
                {
                    plainToolInvocations++;
                    return $"sunny in {city}";
                },
                "get_weather",
                "Gets the weather for a city"));
        }

        return tools;
    }
}
