using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Converters;
using Temporalio.MicrosoftAgentFramework.Activities;

namespace Temporalio.MicrosoftAgentFramework.Tests;

public class SerializationTests
{
    [Fact]
    public void ModelCallInput_AllContentKinds_RoundTrip()
    {
        var message = new ChatMessage(
            ChatRole.Assistant,
            [
                new TextContent("hello"),
                new FunctionCallContent("call-1", "get_weather", new Dictionary<string, object?> { ["city"] = "Kingston", ["days"] = 3 }),
                new FunctionResultContent("call-0", "sunny"),
                new DataContent(new byte[] { 1, 2, 3 }, "application/octet-stream"),
                new UriContent("https://example.com/a.png", "image/png"),
                new ToolApprovalRequestContent("ficc_call-2", new FunctionCallContent("call-2", "send_email", new Dictionary<string, object?> { ["to"] = "x@y.z" })),
                new TextReasoningContent("thinking"),
                new UsageContent(new UsageDetails { InputTokenCount = 1, OutputTokenCount = 2 }),
            ])
        {
            MessageId = "m-1",
            AuthorName = "assistant",
        };
        var input = new ModelCallInput
        {
            Model = "gpt",
            Messages = [message, new ChatMessage(ChatRole.User, "hi")],
            Options = new ChatOptions
            {
                Instructions = "be brief",
                Temperature = 0.1f,
                MaxOutputTokens = 50,
                ResponseFormat = ChatResponseFormat.ForJsonSchema<Workflows.WeatherReport>(),
                AdditionalProperties = new() { ["seed"] = 7 },
            },
            Tools = [new ToolDeclaration("get_weather", "Gets the weather", JsonSerializer.Deserialize<JsonElement>("""{"type":"object"}"""))],
            HostedTools = [new HostedToolDeclaration { Kind = HostedToolDeclaration.WebSearch }],
            Stream = true,
            StreamingTopic = "deltas",
            StreamingBatchInterval = TimeSpan.FromMilliseconds(100),
        };

        var back = RoundTrip(input);

        Assert.Equal("gpt", back.Model);
        Assert.Equal(2, back.Messages.Count);
        Assert.Equal(
            ["TextContent", "FunctionCallContent", "FunctionResultContent", "DataContent", "UriContent", "ToolApprovalRequestContent", "TextReasoningContent", "UsageContent"],
            back.Messages[0].Contents.Select(c => c.GetType().Name));
        Assert.Equal("m-1", back.Messages[0].MessageId);
        var call = back.Messages[0].Contents.OfType<FunctionCallContent>().First();
        Assert.Equal("Kingston", call.Arguments!["city"]?.ToString());
        Assert.Equal("be brief", back.Options!.Instructions);
        Assert.Equal(0.1f, back.Options.Temperature);
        Assert.IsType<ChatResponseFormatJson>(back.Options.ResponseFormat);
        Assert.Null(back.Options.Tools);
        Assert.Equal("get_weather", back.Tools![0].Name);
        Assert.Equal("object", back.Tools[0].JsonSchema.GetProperty("type").GetString());
        Assert.Equal(HostedToolDeclaration.WebSearch, back.HostedTools![0].Kind);
        Assert.True(back.Stream);
        Assert.Equal("deltas", back.StreamingTopic);
        Assert.Equal(TimeSpan.FromMilliseconds(100), back.StreamingBatchInterval);
    }

    [Fact]
    public void ApprovalResponseAndMcpCall_RoundTrip()
    {
        var response = new ToolApprovalResponseContent("ficc_call-2", false, new FunctionCallContent("call-2", "send_email", null)) { Reason = "not now" };
        var back = RoundTrip(response);
        Assert.Equal("ficc_call-2", back.RequestId);
        Assert.False(back.Approved);
        Assert.Equal("not now", back.Reason);
        Assert.Equal("send_email", ((FunctionCallContent)back.ToolCall).Name);

        var call = new McpToolCall("calc", "add", new Dictionary<string, object?> { ["a"] = JsonSerializer.Deserialize<JsonElement>("2"), ["b"] = 3 });
        var callBack = RoundTrip(call);
        Assert.Equal("calc", callBack.Server);
        Assert.Equal(2, ((JsonElement)callBack.Arguments!["a"]!).GetInt32());
        Assert.Equal(3, ((JsonElement)callBack.Arguments["b"]!).GetInt32());
    }

    [Fact]
    public void Session_SerializeDeserialize_KeepsHistory()
    {
        var provider = new InMemoryChatHistoryProvider();
        var session = new TemporalAgentSession();
        provider.SetMessages(session, [new ChatMessage(ChatRole.User, "one"), new ChatMessage(ChatRole.Assistant, "two")]);

        var element = RoundTrip(session.Serialize());
        Assert.True(element.TryGetProperty("stateBag", out var bag));
        Assert.True(bag.TryGetProperty(nameof(InMemoryChatHistoryProvider), out _));

        var restored = TemporalAgentSession.Deserialize(element);
        Assert.Equal(["one", "two"], provider.GetMessages(restored).Select(m => m.Text));
    }

    private static T RoundTrip<T>(T value)
    {
        var converter = DataConverter.Default.PayloadConverter;
        return (T)converter.ToValue(converter.ToPayload(value), typeof(T))!;
    }
}
