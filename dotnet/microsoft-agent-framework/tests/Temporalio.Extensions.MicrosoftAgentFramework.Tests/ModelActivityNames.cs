using Temporalio.Extensions.MicrosoftAgentFramework.Activities;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

public static class ModelActivityNames
{
    public const string InvokeModel = ModelActivities.InvokeModelActivityName;

    public const string McpListTools = McpActivities.ListToolsActivityName;

    public const string McpCallTool = McpActivities.CallToolActivityName;
}
