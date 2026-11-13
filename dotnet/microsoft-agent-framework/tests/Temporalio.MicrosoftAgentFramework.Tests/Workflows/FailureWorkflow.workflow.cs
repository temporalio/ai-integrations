using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Exceptions;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class FailureWorkflow
{
    [WorkflowRun]
    public async Task<string> RunAsync(FailureInput input)
    {
        var modelOptions = new ActivityOptions
        {
            StartToCloseTimeout = TimeSpan.FromSeconds(30),
            RetryPolicy = new() { MaximumAttempts = input.MaxAttempts, InitialInterval = TimeSpan.FromMilliseconds(50), BackoffCoefficient = 1 },
        };
        var weather = AIFunctionFactory.Create((string city) => $"sunny in {city}", "get_weather", "Gets the weather for a city");
        var failing = AIFunctionFactory.Create((Func<string, string>)(city => throw new InvalidOperationException($"no data for {city}")), "get_weather", "Gets the weather for a city");
        switch (input.Kind)
        {
            case "model":
                {
                    var agent = new TemporalAgent(input.Model, new TemporalAgentOptions { ModelActivityOptions = modelOptions });
                    return (await agent.RunAsync(input.Prompt)).Text;
                }

            case "base-entry":
                {
                    AIAgent agent = new TemporalAgent(input.Model);
                    return (await agent.RunAsync(input.Prompt)).Text;
                }

            case "iteration-cap":
                {
                    var agent = new TemporalAgent(input.Model, new TemporalAgentOptions
                    {
                        MaximumIterationsPerRun = input.MaximumIterations,
                        ChatOptions = new ChatOptions { Tools = [weather] },
                    });
                    return (await agent.RunAsync(input.Prompt)).Text;
                }

            case "unknown-tool":
                {
                    var agent = new TemporalAgent(input.Model, new TemporalAgentOptions { ChatOptions = new ChatOptions { Tools = [weather] } });
                    return (await agent.RunAsync(input.Prompt)).Text;
                }

            case "tool-errors":
                {
                    var agent = new TemporalAgent(input.Model, new TemporalAgentOptions { ChatOptions = new ChatOptions { Tools = [failing] } });
                    return (await agent.RunAsync(input.Prompt)).Text;
                }

            case "unsupported-run-options":
                {
                    var agent = new TemporalAgent(input.Model);
                    var options = new ChatClientAgentRunOptions { ChatClientFactory = client => client };
                    return (await agent.RunAsync(input.Prompt, options: options)).Text;
                }

            case "wrong-session":
                {
                    var agent = new TemporalAgent(input.Model);
                    return (await agent.RunAsync(input.Prompt, new OtherSession())).Text;
                }

            case "unanswered-approval":
                {
                    var agent = new TemporalAgent(input.Model, new TemporalAgentOptions
                    {
                        ChatOptions = new ChatOptions { Tools = [new ApprovalRequiredAIFunction(weather)] },
                    });
                    var session = await agent.CreateSessionAsync();
                    await agent.RunAsync(input.Prompt, session);
                    return (await agent.RunAsync("never mind, just answer", session)).Text;
                }

            case "declaration-only":
                {
                    var schema = JsonSerializer.Deserialize<JsonElement>("""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");
                    var declaration = AIFunctionFactory.CreateDeclaration("get_weather", "Gets the weather for a city", schema);
                    var agent = new TemporalAgent(input.Model, new TemporalAgentOptions { ChatOptions = new ChatOptions { Tools = [declaration] } });
                    var response = await agent.RunAsync(input.Prompt);
                    var calls = response.Messages.SelectMany(message => message.Contents).OfType<FunctionCallContent>().Count(call => !call.InformationalOnly);
                    return $"calls returned to caller: {calls}";
                }

            default:
                throw new ApplicationFailureException($"Unknown failure kind {input.Kind}");
        }
    }

    private sealed class OtherSession : AgentSession
    {
    }
}
