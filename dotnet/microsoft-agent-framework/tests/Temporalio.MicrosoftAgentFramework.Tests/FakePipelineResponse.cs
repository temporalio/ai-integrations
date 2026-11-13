using System.ClientModel.Primitives;

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>A System.ClientModel response with a status and optional headers, for provider error tests.</summary>
public sealed class FakePipelineResponse : PipelineResponse
{
    private readonly FakeHeaders headers;

    public FakePipelineResponse(int status, string? retryAfter)
        : this(status, retryAfter is null ? new Dictionary<string, string>() : new Dictionary<string, string> { ["Retry-After"] = retryAfter })
    {
    }

    public FakePipelineResponse(int status, IReadOnlyDictionary<string, string> headers)
    {
        Status = status;
        this.headers = new FakeHeaders(headers);
    }

    public override int Status { get; }

    public override string ReasonPhrase => "fake";

    public override Stream? ContentStream { get; set; }

    public override BinaryData Content => BinaryData.Empty;

    protected override PipelineResponseHeaders HeadersCore => headers;

    public override BinaryData BufferContent(CancellationToken cancellationToken = default) => BinaryData.Empty;

    public override ValueTask<BinaryData> BufferContentAsync(CancellationToken cancellationToken = default) => new(BinaryData.Empty);

    public override void Dispose()
    {
    }

    private sealed class FakeHeaders : PipelineResponseHeaders
    {
        private readonly Dictionary<string, string> values = new(StringComparer.OrdinalIgnoreCase);

        public FakeHeaders(IReadOnlyDictionary<string, string> headers)
        {
            foreach (var (name, value) in headers)
            {
                values[name] = value;
            }
        }

        public override IEnumerator<KeyValuePair<string, string>> GetEnumerator() => values.GetEnumerator();

        public override bool TryGetValue(string name, out string? value) => values.TryGetValue(name, out value);

        public override bool TryGetValues(string name, out IEnumerable<string>? values)
        {
            if (this.values.TryGetValue(name, out var value))
            {
                values = [value];
                return true;
            }

            values = null;
            return false;
        }
    }
}
