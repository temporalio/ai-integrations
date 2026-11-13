using System.Text.Json;

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>Response bodies in the shape of the OpenAI Responses API, with storage off so the client keeps no conversation id.</summary>
public static class ResponsesApi
{
    public static string Text(string text) => Envelope(
        $$"""{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":{{JsonSerializer.Serialize(text)}},"annotations":[]}]}""");

    public static string FunctionCall(string callId, string name, string argumentsJson) => Envelope(
        $$"""{"type":"function_call","id":"fc_1","call_id":{{JsonSerializer.Serialize(callId)}},"name":{{JsonSerializer.Serialize(name)}},"arguments":{{JsonSerializer.Serialize(argumentsJson)}},"status":"completed"}""");

    private static string Envelope(string outputItem) =>
        $$$"""{"id":"resp_1","object":"response","created_at":1700000000,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-4o-mini","output":[{{{outputItem}}}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":false,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":0},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":15},"user":null,"metadata":{}}""";
}
