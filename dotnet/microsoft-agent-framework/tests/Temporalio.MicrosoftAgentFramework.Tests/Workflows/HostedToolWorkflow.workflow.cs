using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class HostedToolWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(HostedToolInput input)
    {
        var docs = new HostedMcpServerTool("docs", "https://example.com/mcp")
        {
            AllowedTools = ["search"],
            ApprovalMode = HostedMcpServerToolApprovalMode.NeverRequire,
            ServerDescription = "Documentation search",
        };
        if (input.WorkflowSideHeaders)
        {
            // The plugin must refuse this: the header would be recorded in Event History with the Activity input.
            docs.Headers = new Dictionary<string, string> { ["Authorization"] = "Bearer workflow-side-secret" };
        }

        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions
            {
                AdditionalProperties = new() { ["strict"] = true, ["seed"] = 7, ["user"] = "ops" },
                Tools = [new HostedWebSearchTool(), docs],
            },
        });
        return (await agent.RunAsync(input.Prompt)).Text;
    }
}
