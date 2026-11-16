namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>The dev server retry matches the SDK's three startup failures and nothing else, as AGENTS.md allows.</summary>
public class DevServerStartupTests
{
    public static IEnumerable<TheoryDataRow<string>> Retryable()
    {
        const string prefix = "Failed starting Temporal dev server: ";
        var timeout = "ephemeral server at 127.0.0.1:60249 did not start within 5s; last connection error: no connection attempt completed before the deadline";
        yield return new(timeout);
        yield return new(prefix + timeout);
        yield return new("ephemeral server at 127.0.0.1:54380 did not start within 200ms; last connection error: Server connection error: tonic::transport::Error(Transport, ConnectError(ConnectError(\"tcp connect error\", 127.0.0.1:54380, Os { code: 61, kind: ConnectionRefused, message: \"Connection refused\" })))");
        foreach (var duration in new[] { "1.5s", "1.25ms", "3µs", "0ns" })
        {
            yield return new(timeout.Replace("5s;", duration + ";", StringComparison.Ordinal));
        }

        yield return new("failed to download ephemeral server executable: error sending request");
        foreach (var message in new[] { "could not read tar archive", "could not read tar archive entry", "could not extract tar archive entry", "could not read ZIP archive", "could not extract ZIP archive entry" })
        {
            yield return new("invalid ephemeral server download: " + message);
        }
    }

    public static IEnumerable<TheoryDataRow<string>> NotRetryable()
    {
        yield return new("Server connection error: tonic::transport::Error(Transport, timed out)");
        yield return new("Failed starting Temporal dev server: unknown flag --foo");
        yield return new("the download directory is not writable");
        yield return new("extract: permission denied");
        yield return new("the ephemeral server at 127.0.0.1:1 did not start within 5s");
    }

    [Theory]
    [MemberData(nameof(Retryable))]
    public void StartupFailures_AreRetried(string message) => Assert.True(DevServerFixture.IsRetryable(message), message);

    [Theory]
    [MemberData(nameof(NotRetryable))]
    public void OtherErrors_AreNot(string message) => Assert.False(DevServerFixture.IsRetryable(message), message);
}
