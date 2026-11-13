using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class SessionWorkflow
{
    private TemporalAgent? agent;
    private AgentSession? session;

    [WorkflowQuery]
    public IList<string> Transcript() =>
        agent?.ChatHistoryProvider is InMemoryChatHistoryProvider provider && session is not null
            ? provider.GetMessages(session).Select(message => $"{message.Role}:{message.Text}").ToList()
            : [];

    [WorkflowRun]
    public async Task<IList<string>> RunAsync(SessionInput input)
    {
        var options = new TemporalAgentOptions { ChatOptions = new ChatOptions { Instructions = "Be brief." } };
        options.ChatHistoryProvider = input.HistoryProvider switch
        {
            "custom" => new RecordingChatHistoryProvider(),
            "delaying" => new DelayingChatHistoryProvider(),
            _ => null,
        };
        options.AIContextProviders = input.ContextProvider switch
        {
            "prefix" => [new PrefixContextProvider()],
            "delaying" => [new DelayingContextProvider()],
            _ => null,
        };

        agent = new TemporalAgent(TestPlugin.Model, options);
        session = input.Session is { } serialized ? await agent.DeserializeSessionAsync(serialized) : await agent.CreateSessionAsync();
        var replies = (input.Replies ?? []).ToList();
        var response = await agent.RunAsync(input.Prompts[input.Index], session);
        replies.Add(response.Text);
        if (input.Index + 1 >= input.Prompts.Count)
        {
            return replies;
        }

        if (input.ContinueAsNew)
        {
            var carried = await agent.SerializeSessionAsync(session);
            var next = input with { Index = input.Index + 1, Session = carried, Replies = replies };
            throw Workflow.CreateContinueAsNewException((SessionWorkflow wf) => wf.RunAsync(next));
        }

        for (var i = input.Index + 1; i < input.Prompts.Count; i++)
        {
            response = await agent.RunAsync(input.Prompts[i], session);
            replies.Add(response.Text);
        }

        return replies;
    }
}
