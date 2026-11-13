namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>Shaped like <c>Azure.RequestFailedException</c>: an int <c>Status</c> and no raw response.</summary>
public sealed class FakeRequestFailedException : Exception
{
    public FakeRequestFailedException(int status, string message)
        : base(message) => Status = status;

    public int Status { get; }
}
