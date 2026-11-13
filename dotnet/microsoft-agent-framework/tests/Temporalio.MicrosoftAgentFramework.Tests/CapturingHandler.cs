using System.Net;
using System.Text;
using System.Text.Json.Nodes;

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>
/// Records every request body a provider client sends and answers each with an HTTP 400, so a
/// test can read what the provider would have received without a provider or a network.
/// </summary>
public sealed class CapturingHandler : HttpMessageHandler
{
    private readonly List<JsonNode> bodies = [];

    public IReadOnlyList<JsonNode> Bodies
    {
        get
        {
            lock (bodies)
            {
                return bodies.ToList();
            }
        }
    }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var text = request.Content is null ? "null" : await request.Content.ReadAsStringAsync(cancellationToken);
        lock (bodies)
        {
            bodies.Add(JsonNode.Parse(text) ?? new JsonObject());
        }

        return new HttpResponseMessage(HttpStatusCode.BadRequest)
        {
            Content = new StringContent(
                """{"error":{"message":"captured by the test","type":"invalid_request_error","param":null,"code":null}}""",
                Encoding.UTF8,
                "application/json"),
        };
    }
}
