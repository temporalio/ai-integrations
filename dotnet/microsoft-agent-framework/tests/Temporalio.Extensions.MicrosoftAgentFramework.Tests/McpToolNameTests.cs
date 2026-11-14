using System.Text.Json;
using Temporalio.Extensions.MicrosoftAgentFramework.Activities;
using Temporalio.Extensions.MicrosoftAgentFramework.Mcp;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>The names the model sees for MCP tools follow Agent Framework's rule.</summary>
public class McpToolNameTests
{
    [Theory]
    [InlineData(null, "tool", "tool")]
    [InlineData("", "tool", "tool")]
    [InlineData("gh_", "tool", "gh_tool")]
    [InlineData("gh", "tool", "gh_tool")]
    [InlineData("gh_", "_tool", "gh_tool")]
    [InlineData("gh.", "tool", "gh_tool")]
    [InlineData("docs", "read file!", "docs_read-file-")]
    [InlineData("__", "tool", "tool")]
    [InlineData("gh", "___", "gh")]
    public void ModelFacingName_FollowsAgentFrameworksMcpRule(string? prefix, string serverToolName, string expected)
    {
        var schema = JsonSerializer.Deserialize<JsonElement>("""{"type":"object"}""");
        var tool = new TemporalMcpTool("calc", new ToolDeclaration(serverToolName, "d", schema), new McpToolOptions { ToolNamePrefix = prefix });
        Assert.Equal(expected, tool.Name);
        Assert.Equal(serverToolName, tool.ServerToolName);
    }
}
