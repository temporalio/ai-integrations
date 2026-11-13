using System.Text;
using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.AgentFramework.Tests.Workflows;

[Workflow]
public class DirectChatClientWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(DirectInput input)
    {
        using var client = new TemporalChatClient(TestPlugin.Model);
        var messages = new List<ChatMessage> { new(ChatRole.User, input.Prompt) };
        if (!input.Streaming)
        {
            return (await client.GetResponseAsync(messages)).Text;
        }

        var builder = new StringBuilder();
        await foreach (var update in client.GetStreamingResponseAsync(messages))
        {
            builder.Append(update.Text);
        }

        return builder.ToString();
    }
}
