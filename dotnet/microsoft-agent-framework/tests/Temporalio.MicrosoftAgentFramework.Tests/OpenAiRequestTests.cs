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
/// client driven by <see cref="ChatClientAgent"/> in process: one script per case, run through
/// both, the provider requests compared. No network: a recording transport answers from the
/// script. The nightly run with the newest Microsoft.Agents.AI runs these too, so a change in the
/// framework's loop shows up as a request difference.
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
    public async Task StrictFunction_ProviderRequest_MatchesChatClientAgent()
    {
        var (nativeClient, nativeRequests) = ResponsesClient();
        var nativeAgent = new ChatClientAgent(nativeClient, new ChatClientAgentOptions
        {
            ChatOptions = new ChatOptions { Instructions = "Be brief.", Tools = [StrictWeatherTool()] },
        });
        await Assert.ThrowsAnyAsync<Exception>(() => nativeAgent.RunAsync("Weather?", cancellationToken: TestContext.Current.CancellationToken));
        var native = Assert.Single(nativeRequests.Bodies);

        var (pluginClient, pluginRequests) = ResponsesClient();
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { [TestPlugin.Model] = pluginClient });
        var (error, _) = await RunFailingAsync(plugin, (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather?", "in-workflow-strict")));
        Assert.Equal(MicrosoftAgentFrameworkErrorTypes.ModelError, AppFailure(error)?.ErrorType);
        var viaPlugin = Assert.Single(pluginRequests.Bodies);

        Assert.True(JsonNode.DeepEquals(native, viaPlugin), $"native:\n{native}\nvia the plugin:\n{viaPlugin}");
        var tool = Assert.Single(viaPlugin["tools"]!.AsArray());
        Assert.Equal(true, (bool?)tool!["strict"]);
        Assert.Equal("get_weather", (string?)tool["name"]);
    }

    [Fact]
    public async Task ToolLoop_ProviderRequests_MatchChatClientAgent()
    {
        string[] script = [ResponsesApi.FunctionCall("call_1", "get_weather", """{"city":"Kingston"}"""), ResponsesApi.Text("Final: sunny")];
        var (nativeClient, nativeRequests) = ScriptedClient(script);
        var nativeAgent = new ChatClientAgent(nativeClient, new ChatClientAgentOptions
        {
            ChatOptions = new ChatOptions { Instructions = "Be brief.", Tools = [WeatherTool()] },
        });
        var nativeResponse = await nativeAgent.RunAsync("Weather in Kingston?", cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("Final: sunny", nativeResponse.Text);

        var (pluginClient, pluginRequests) = ScriptedClient(script);
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { [TestPlugin.Model] = pluginClient });
        var (result, _) = await RunAsync(plugin, (ToolsWorkflow wf) => wf.RunAsync(new ToolsInput("Weather in Kingston?", "in-workflow")));
        Assert.Equal("Final: sunny", result.Text);

        AssertSameRequests(nativeRequests.Requests, pluginRequests.Requests, 2);
    }

    [Fact]
    public async Task Approvals_ProviderRequests_MatchChatClientAgent()
    {
        string[] script = [ResponsesApi.FunctionCall("call_1", "send_email", """{"to":"ops@example.com"}"""), ResponsesApi.Text("Done")];
        var (nativeClient, nativeRequests) = ScriptedClient(script);
        var nativeAgent = new ChatClientAgent(nativeClient, new ChatClientAgentOptions
        {
            ChatOptions = new ChatOptions { Tools = [new ApprovalRequiredAIFunction(EmailTool())] },
        });
        var session = await nativeAgent.CreateSessionAsync(TestContext.Current.CancellationToken);
        var first = await nativeAgent.RunAsync("Email ops", session, cancellationToken: TestContext.Current.CancellationToken);
        var request = first.Messages.SelectMany(message => message.Contents).OfType<ToolApprovalRequestContent>().Single();
        var second = await nativeAgent.RunAsync(new ChatMessage(ChatRole.User, [request.CreateResponse(true)]), session, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("Done", second.Text);

        var (pluginClient, pluginRequests) = ScriptedClient(script);
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { [TestPlugin.Model] = pluginClient });
        using var worker = Fixture.NewWorker(plugin);
        await WithWorkerAsync(worker, async () =>
        {
            var input = new ApprovalInput("Email ops", "workflow");
            var handle = await Client.StartWorkflowAsync((ApprovalWorkflow wf) => wf.RunAsync(input), Fixture.WorkflowOptions(worker, "parity-approval"));
            ToolApprovalRequestContent? pending = null;
            for (var i = 0; i < 200 && pending is null; i++)
            {
                await Task.Delay(50, TestContext.Current.CancellationToken);
                pending = await handle.QueryAsync(wf => wf.PendingApproval());
            }

            Assert.NotNull(pending);
            await handle.SignalAsync(wf => wf.DecideAsync(pending.CreateResponse(true)));
            Assert.Equal("Done", await handle.GetResultAsync());
        });

        AssertSameRequests(nativeRequests.Requests, pluginRequests.Requests, 2);
    }

    [Fact]
    public async Task MultiTurn_ProviderRequests_MatchChatClientAgent()
    {
        string[] script = [ResponsesApi.Text("echo one"), ResponsesApi.Text("echo two")];
        var (nativeClient, nativeRequests) = ScriptedClient(script);
        var nativeAgent = new ChatClientAgent(nativeClient, new ChatClientAgentOptions { ChatOptions = new ChatOptions { Instructions = "Be brief." } });
        var session = await nativeAgent.CreateSessionAsync(TestContext.Current.CancellationToken);
        await nativeAgent.RunAsync("first", session, cancellationToken: TestContext.Current.CancellationToken);
        var second = await nativeAgent.RunAsync("second", session, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("echo two", second.Text);

        var (pluginClient, pluginRequests) = ScriptedClient(script);
        var plugin = TestPlugin.For(new Dictionary<string, IChatClient> { [TestPlugin.Model] = pluginClient });
        var turns = new SessionInput(["first", "second"]);
        var (replies, _) = await RunAsync(plugin, (SessionWorkflow wf) => wf.RunAsync(turns));
        Assert.Equal(["echo one", "echo two"], replies);

        AssertSameRequests(nativeRequests.Requests, pluginRequests.Requests, 2);
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

    // The same tool definitions as the test Workflows build, so only the loop differs.
    private static AIFunction WeatherTool() => AIFunctionFactory.Create((string city) => $"sunny in {city} (workflow)", "get_weather", "Gets the weather for a city");

    private static AIFunction StrictWeatherTool() => AIFunctionFactory.Create(
        (string city) => $"sunny in {city} (workflow)",
        new AIFunctionFactoryOptions { Name = "get_weather", Description = "Gets the weather for a city", AdditionalProperties = new Dictionary<string, object?> { ["strict"] = true } });

    private static AIFunction EmailTool() => AIFunctionFactory.Create((string to) => $"email sent to {to}", "send_email", "Sends an email");

    // Two things about earlier output items do not cross the Activity boundary, because the framework keeps
    // the provider's response objects in memory and the plugin keeps content: the ids the service assigned
    // (fc_1, msg_1) and the exact text of a function call's arguments, which the provider client writes
    // again from the parsed arguments (same JSON, different whitespace). Both are accepted by the API; the
    // comparison removes the ids and parses the arguments. Everything else must be the same.
    private static JsonNode Normalize(JsonNode request)
    {
        var copy = request.DeepClone();
        foreach (var item in copy["input"]?.AsArray() ?? [])
        {
            if (item is not JsonObject itemObject)
            {
                continue;
            }

            itemObject.Remove("id");
            itemObject.Remove("status");
            if ((string?)itemObject["type"] == "function_call" && itemObject["arguments"] is JsonValue arguments && arguments.TryGetValue<string>(out var text))
            {
                itemObject["arguments"] = JsonNode.Parse(text);
            }
        }

        return copy;
    }

    private static void AssertSameRequests(IReadOnlyList<JsonNode> native, IReadOnlyList<JsonNode> viaPlugin, int expectedCount)
    {
        Assert.Equal(expectedCount, native.Count);
        Assert.Equal(expectedCount, viaPlugin.Count);
        for (var i = 0; i < expectedCount; i++)
        {
            var left = Normalize(native[i]);
            var right = Normalize(viaPlugin[i]);
            Assert.True(JsonNode.DeepEquals(left, right), $"request {i + 1} differs.\nnative:\n{left}\nvia the plugin:\n{right}");
        }
    }

    private static (IChatClient Client, CapturingHandler Requests) ResponsesClient(bool storeDisabled = false)
    {
        var requests = new CapturingHandler();
        var client = NewClient(requests);
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

    private static (IChatClient Client, ScriptedResponsesHandler Requests) ScriptedClient(string[] script)
    {
        var requests = new ScriptedResponsesHandler(script);
        return (NewClient(requests), requests);
    }

    private static IChatClient NewClient(HttpMessageHandler handler)
    {
#pragma warning disable CA2000 // The transport owns the HttpClient for the life of the test client.
        var options = new OpenAIClientOptions
        {
            Transport = new HttpClientPipelineTransport(new HttpClient(handler)),
            RetryPolicy = new ClientRetryPolicy(maxRetries: 0),
        };
#pragma warning restore CA2000
        return new OpenAIClient(new ApiKeyCredential("test-key"), options).GetResponsesClient().AsIChatClient("gpt-4o-mini");
    }
}
