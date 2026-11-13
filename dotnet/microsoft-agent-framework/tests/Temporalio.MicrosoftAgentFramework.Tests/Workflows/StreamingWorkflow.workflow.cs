using System.Text;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Extensions.WorkflowStreams;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

[Workflow]
public class StreamingWorkflow
{
    // Constructing the stream registers its publish Signal, poll Update and offset Query before
    // the first Activity publishes to it.
#pragma warning disable IDE0052 // The stream is used through its registered handlers.
    private readonly WorkflowStream stream;
#pragma warning restore IDE0052

    [WorkflowInit]
    public StreamingWorkflow(StreamingInput input) => stream = new(input.StreamState);

    [WorkflowRun]
    public async Task<StreamingResult> RunAsync(StreamingInput input)
    {
        var agent = new TemporalAgent(TestPlugin.Model, new TemporalAgentOptions
        {
            StreamingTopic = input.Topic,
            StreamingBatchInterval = input.Topic is null ? null : TimeSpan.FromMilliseconds(50),
            ChatOptions = new ChatOptions
            {
                Tools = input.WithTool ? [AIFunctionFactory.Create((string city) => $"sunny in {city}", "get_weather", "Gets the weather for a city")] : null,
            },
        });
        var texts = new List<string>();
        var updates = new List<AgentResponseUpdate>();
        var builder = new StringBuilder();
        var count = 0;
        var toolUpdates = 0;
        await foreach (var update in agent.RunStreamingAsync(input.Prompt))
        {
            updates.Add(update);
            count++;
            if (update.Role == ChatRole.Tool)
            {
                toolUpdates++;
            }

            if (!string.IsNullOrEmpty(update.Text))
            {
                texts.Add(update.Text);
                if (update.Role == ChatRole.Assistant)
                {
                    builder.Append(update.Text);
                }
            }
        }

        if (input.Topic is not null)
        {
            // Give subscribers time to poll the last batch before the Workflow closes.
            await Workflow.DelayAsync(TimeSpan.FromMilliseconds(500));
        }

        return new StreamingResult(texts, count, builder.ToString(), toolUpdates, updates.ToAgentResponse().Usage?.TotalTokenCount);
    }
}
