using System.Net;
using System.Text;
using System.Text.Json.Nodes;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>
/// Answers an OpenAI Responses client from a script of response bodies, one per request, and
/// records every request body, so a test can compare what two callers send without a provider.
/// </summary>
public sealed class ScriptedResponsesHandler : HttpMessageHandler
{
    private readonly Queue<string> script;
    private readonly List<JsonNode> requests = [];

    public ScriptedResponsesHandler(params string[] bodies) => script = new Queue<string>(bodies);

    public IReadOnlyList<JsonNode> Requests
    {
        get
        {
            lock (requests)
            {
                return requests.ToList();
            }
        }
    }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var text = request.Content is null ? "null" : await request.Content.ReadAsStringAsync(cancellationToken);
        string body;
        lock (requests)
        {
            requests.Add(JsonNode.Parse(text) ?? new JsonObject());
            body = script.Count > 0 ? script.Dequeue() : ResponsesApi.Text("script exhausted");
        }

        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    }
}
