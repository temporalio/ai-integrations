using Microsoft.Extensions.AI;
using Temporalio.Workflows;

namespace Temporalio.AgentFramework.Tests.Workflows;

[Workflow]
public class HostedToolWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(string prompt)
    {
        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions
            {
                AdditionalProperties = new() { ["strict"] = true, ["seed"] = 7, ["user"] = "ops" },
                Tools =
                [
                    new HostedWebSearchTool(),
                    new HostedMcpServerTool("docs", "https://example.com/mcp")
                    {
                        AllowedTools = ["search"],
                        ApprovalMode = HostedMcpServerToolApprovalMode.NeverRequire,
                        ServerDescription = "Documentation search",
                    },
                ],
            },
        });
        return (await agent.RunAsync(prompt)).Text;
    }
}
