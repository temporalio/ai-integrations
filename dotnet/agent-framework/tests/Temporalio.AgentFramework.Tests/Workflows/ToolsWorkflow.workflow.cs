using Microsoft.Extensions.AI;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.AgentFramework.Tests.Workflows;

[Workflow]
public class ToolsWorkflow
{
    [WorkflowRun]
    public async Task<ToolsResult> RunAsync(ToolsInput input)
    {
        var toolOptions = new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(30) };
        IList<AITool> tools = input.ToolKind switch
        {
            "none" => [],
            "in-workflow" => [AIFunctionFactory.Create((string city) => $"sunny in {city} (workflow)", "get_weather", "Gets the weather for a city")],
            "activity-expression" => [ActivityAsTool.Create((WeatherActivities a) => a.GetWeatherAsync(default!), toolOptions, "get_weather")],
            "activity-static" => [ActivityAsTool.Create(() => StaticActivities.LookupTimeZone(default!), toolOptions, "get_weather")],
            "activity-delegate" => [ActivityAsTool.Create(StaticActivities.LookupTimeZone, toolOptions, "get_weather")],
            "activity-methodinfo" => [ActivityAsTool.Create(typeof(WeatherActivities).GetMethod(nameof(WeatherActivities.GetWeatherAsync))!, toolOptions, "get_weather")],
            "activity-report" => [ActivityAsTool.Create((WeatherActivities a) => a.GetReportAsync(default!, default), toolOptions, "get_report")],
            _ => throw new ApplicationFailureException($"Unknown tool kind {input.ToolKind}"),
        };
        var agent = new TemporalAgent(input.Model, new TemporalAgentOptions
        {
            Id = input.AgentId,
            Name = input.AgentName,
            AllowConcurrentInvocation = input.Concurrent,
            ChatOptions = new ChatOptions { Instructions = "Be brief.", Tools = tools },
        });
        var response = await agent.RunAsync(input.Prompt);
        return new ToolsResult(
            response.Text,
            response.AgentId!,
            response.Messages.Count,
            response.Usage?.TotalTokenCount,
            response.Messages.Select(message => message.AuthorName ?? string.Empty).ToList());
    }
}
