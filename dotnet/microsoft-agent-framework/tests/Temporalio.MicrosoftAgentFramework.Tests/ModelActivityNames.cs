using Temporalio.MicrosoftAgentFramework.Activities;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public static class ModelActivityNames
{
    public const string InvokeModel = ModelActivities.InvokeModelActivityName;

    public const string McpListTools = McpActivities.ListToolsActivityName;

    public const string McpCallTool = McpActivities.CallToolActivityName;
}
