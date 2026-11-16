using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class StructuredOutputWorkflow
{
    [WorkflowRun]
    public async Task<StructuredResult> RunAsync(StructuredInput input)
    {
        var agent = new TemporalAgent(TestPlugin.Model);
        if (input.Kind == "string")
        {
            var text = await agent.RunAsync<string>(input.Prompt);
            return new StructuredResult(null, text.Result);
        }

        var report = await agent.RunAsync<WeatherReport>(input.Prompt);
        return new StructuredResult(report.Result, null);
    }
}
