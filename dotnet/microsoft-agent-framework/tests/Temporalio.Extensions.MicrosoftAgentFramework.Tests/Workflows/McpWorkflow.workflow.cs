using System.Text.Json;
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework.Activities;
using Temporalio.Extensions.MicrosoftAgentFramework.Mcp;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class McpWorkflow
{
    private const string AddSchema = """{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}""";

    [WorkflowRun]
    public async Task<McpResult> RunAsync(McpInput input)
    {
        var retry = input.MaxAttempts > 0
            ? new Temporalio.Common.RetryPolicy { MaximumAttempts = input.MaxAttempts, InitialInterval = TimeSpan.FromMilliseconds(50), BackoffCoefficient = 1 }
            : null;
        var options = new McpToolOptions
        {
            AllowedTools = input.AllowedTools?.ToList(),
            ToolNamePrefix = input.ToolNamePrefix,
            CacheTools = input.CacheTools,
            ListToolsActivityOptions = retry is null ? null : new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(10), RetryPolicy = retry },
            CallToolActivityOptions = retry is null ? null : new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(10), RetryPolicy = retry },
        };
        IReadOnlyList<AIFunction> tools;
        if (input.PinnedCatalog)
        {
            var schema = JsonSerializer.Deserialize<JsonElement>(AddSchema);
            tools = [new TemporalMcpTool(input.Server, new ToolDeclaration(input.PinnedTool, "Adds two integers", schema), options)];
        }
        else
        {
            tools = await TemporalMcpServer.ListToolsAsync(input.Server, options);
            if (input.ListTwice)
            {
                tools = await TemporalMcpServer.ListToolsAsync(input.Server, options);
            }
        }

        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions
        {
            ChatOptions = new ChatOptions { Tools = tools.Cast<AITool>().ToList() },
        });
        var response = await agent.RunAsync(input.Prompt);
        return new McpResult(
            response.Text,
            tools.Select(tool => tool.Name).OrderBy(name => name, StringComparer.Ordinal).ToList(),
            tools.FirstOrDefault(tool => tool.Name.EndsWith("add", StringComparison.Ordinal))?.JsonSchema.GetRawText() ?? string.Empty);
    }
}
