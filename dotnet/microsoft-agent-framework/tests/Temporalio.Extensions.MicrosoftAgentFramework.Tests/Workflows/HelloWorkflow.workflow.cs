using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

// This class is the README quickstart; ReadmeSnippetTests checks the two stay identical.
[Workflow]
public class HelloWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var agent = new TemporalAgent("gpt-4o-mini", new TemporalAgentOptions
        {
            Name = "assistant",
            ChatOptions = new ChatOptions { Instructions = "Answer in one sentence." },
        });
        var response = await agent.RunAsync(prompt);
        return response.Text;
    }
}
