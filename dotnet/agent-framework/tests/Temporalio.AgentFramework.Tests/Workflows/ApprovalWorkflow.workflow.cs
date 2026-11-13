using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.AgentFramework.Tests.Workflows;

[Workflow]
public class ApprovalWorkflow
{
    private readonly Dictionary<string, ToolApprovalResponseContent> decisions = new(StringComparer.Ordinal);
    private ToolApprovalRequestContent? pending;
    private int workflowToolInvocations;
    private int plainToolInvocations;

    [WorkflowSignal]
    public Task DecideAsync(ToolApprovalResponseContent decision)
    {
        decisions[decision.RequestId] = decision;
        return Task.CompletedTask;
    }

    [WorkflowQuery]
    public ToolApprovalRequestContent? PendingApproval() => pending;

    [WorkflowQuery]
    public int WorkflowToolInvocations() => workflowToolInvocations;

    [WorkflowQuery]
    public int PlainToolInvocations() => plainToolInvocations;

    [WorkflowRun]
    public async Task<string> RunAsync(ApprovalInput input)
    {
        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions { ChatOptions = new ChatOptions { Tools = BuildTools(input) } });
        var session = await agent.CreateSessionAsync();
        var response = await agent.RunAsync(input.Prompt, session);
        var request = response.Messages.SelectMany(message => message.Contents).OfType<ToolApprovalRequestContent>().FirstOrDefault();
        while (request is not null)
        {
            pending = request;
            var requestId = request.RequestId;
            await Workflow.WaitConditionAsync(() => decisions.ContainsKey(requestId));
            pending = null;
            response = await agent.RunAsync(new ChatMessage(ChatRole.User, [decisions[requestId]]), session);
            request = response.Messages.SelectMany(message => message.Contents).OfType<ToolApprovalRequestContent>().FirstOrDefault();
        }

        return response.Text;
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
