using Microsoft.Extensions.AI;
using Temporalio.MicrosoftAgentFramework.Testing;
using Temporalio.MicrosoftAgentFramework.Tests.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class StructuredOutputTests : WorkflowTestBase
{
    public StructuredOutputTests(DevServerFixture fixture)
        : base(fixture)
    {
    }

    [Fact]
    public async Task RunAsyncOfT_SchemaSentToActivity_ResultParsed()
    {
        var fake = new FakeChatClient(ScriptedResponse.Json(new WeatherReport("Kingston", "sunny")));
        var (result, _) = await RunAsync(TestPlugin.For(fake), (StructuredOutputWorkflow wf) => wf.RunAsync(new StructuredInput("Weather in Kingston?", "object")));
        Assert.Equal(new WeatherReport("Kingston", "sunny"), result.Report);
        var format = Assert.IsType<ChatResponseFormatJson>(fake.Calls[0].Options!.ResponseFormat);
        var schema = format.Schema!.Value.GetRawText();
        Assert.Contains("city", schema, StringComparison.Ordinal);
        Assert.Contains("forecast", schema, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RunAsyncOfT_NonObjectSchema_WrappedAndUnwrapped()
    {
        var fake = new FakeChatClient(ScriptedResponse.Json(new Dictionary<string, object?> { ["data"] = "Kingston is sunny" }));
        var (result, _) = await RunAsync(TestPlugin.For(fake), (StructuredOutputWorkflow wf) => wf.RunAsync(new StructuredInput("Weather in Kingston?", "string")));
        Assert.Equal("Kingston is sunny", result.Text);
        var format = Assert.IsType<ChatResponseFormatJson>(fake.Calls[0].Options!.ResponseFormat);
        Assert.Contains("\"data\"", format.Schema!.Value.GetRawText(), StringComparison.Ordinal);
    }
}
