using System.ClientModel;
using System.ClientModel.Primitives;
using System.Text.Json.Nodes;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

#pragma warning disable OPENAI001 // The Responses client is marked experimental in OpenAI 2.13 and Microsoft.Extensions.AI.OpenAI 10.10.

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>
/// What the OpenAI Responses client sends when the plugin drives it, compared with the same
/// client driven by <see cref="ChatClientAgent"/> in process. No network: the client's transport
/// records the request and answers with an HTTP 400.
/// </summary>
public class OpenAiRequestTests : WorkflowTestBase
{
    public OpenAiRequestTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task StructuredOutput_ProviderRequest_MatchesChatClientAgent()
    {
        var (nativeClient, nativeRequests) = ResponsesClient();
        var nativeAgent = new ChatClientAgent(nativeClient);
        await Assert.ThrowsAnyAsync<Exception>(() => nativeAgent.RunAsync<WeatherReport>("Weather in Kingston?", cancellationToken: TestContext.Current.CancellationToken));
        var native = Assert.Single(nativeRequests.Bodies);

        var (pluginClient, pluginRequests) = ResponsesClient();
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { [TestPlugin.Model] = pluginClient });
        var input = new StructuredInput("Weather in Kingston?", "report");
        var (error, _) = await RunFailingAsync(plugin, (StructuredOutputWorkflow wf) => wf.RunAsync(input));
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.ModelError, AppFailure(error)?.ErrorType);
        var viaPlugin = Assert.Single(pluginRequests.Bodies);

        Assert.True(JsonNode.DeepEquals(native, viaPlugin), $"native:\n{native}\nvia the plugin:\n{viaPlugin}");
        var format = viaPlugin["text"]!["format"]!;
        Assert.Equal("json_schema", (string?)format["type"]);
        Assert.Equal("WeatherReport", (string?)format["name"]);
        Assert.Equal(false, (bool?)format["schema"]!["additionalProperties"]);
        Assert.Null(viaPlugin["previous_response_id"]);
        Assert.Null(viaPlugin["conversation"]);
        Assert.Null(viaPlugin["store"]);
    }

    [Fact]
    public async Task WorkerSideRawRepresentation_DisablesStoredResponses()
    {
        var (client, requests) = ResponsesClient(storeDisabled: true);
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { ["gpt-4o-mini"] = client });
        var (error, _) = await RunFailingAsync(plugin, (HelloWorkflow wf) => wf.RunAsync("hi"));
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.ModelError, AppFailure(error)?.ErrorType);
        var body = Assert.Single(requests.Bodies);
        Assert.Equal(false, (bool?)body["store"]);
        Assert.Equal("Answer in one sentence.", (string?)body["instructions"]);
        Assert.Equal("gpt-4o-mini", (string?)body["model"]);
    }

    private static (IChatClient Client, CapturingHandler Requests) ResponsesClient(bool storeDisabled = false)
    {
        var requests = new CapturingHandler();
#pragma warning disable CA2000 // The transport owns the HttpClient for the life of the test client.
        var options = new OpenAIClientOptions
        {
            Transport = new HttpClientPipelineTransport(new HttpClient(requests)),
            RetryPolicy = new ClientRetryPolicy(maxRetries: 0),
        };
#pragma warning restore CA2000
        IChatClient client = new OpenAIClient(new ApiKeyCredential("test-key"), options).GetResponsesClient().AsIChatClient("gpt-4o-mini");
        if (storeDisabled)
        {
            // The README's recipe: a worker-side wrapper sets the raw request options the plugin cannot build.
            client = client.AsBuilder()
                .Use((messages, chatOptions, next, cancellationToken) =>
                {
                    chatOptions ??= new ChatOptions();
                    chatOptions.RawRepresentationFactory = _ => new CreateResponseOptions { StoredOutputEnabled = false };
                    return next(messages, chatOptions, cancellationToken);
                })
                .Build();
        }

        return (client, requests);
    }
}
