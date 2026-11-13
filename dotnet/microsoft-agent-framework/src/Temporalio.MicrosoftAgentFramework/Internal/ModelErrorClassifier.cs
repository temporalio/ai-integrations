using System.Globalization;
using System.Reflection;
using Temporalio.Exceptions;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// Turns a provider exception into an <see cref="ApplicationFailureException"/> with
/// <see cref="MicrosoftAgentFrameworkErrorTypes.ModelError"/>. The rule is the one the Python
/// plugin applies: 408, 409, 429 and 5xx are retryable, every other 4xx is not; a provider
/// <c>x-should-retry</c> header overrides it, and <c>retry-after-ms</c> or <c>Retry-After</c>
/// becomes the next retry delay. Classification reads HTTP status codes and headers, never
/// message text. The status extraction paths follow Cecil Phillip's draft at
/// https://github.com/temporal-community/temporal-dotnet-agents.
/// </summary>
internal static class ModelErrorClassifier
{
    /// <summary>
    /// Classifies <paramref name="exception"/>.
    /// </summary>
    /// <param name="exception">Exception thrown by the provider client.</param>
    /// <returns>
    /// The typed failure to throw, or null when the exception carries no HTTP status and should
    /// propagate unchanged (the SDK then marks it retryable with the exception type as the name).
    /// </returns>
    public static ApplicationFailureException? Classify(Exception exception)
    {
        foreach (var candidate in Unwrap(exception))
        {
            if (TryGetHttpStatus(candidate, out var status))
            {
                var headers = new ResponseHeaders(candidate);
                var retryable = headers.Get("x-should-retry") switch
                {
                    { } flag when flag.Equals("false", StringComparison.OrdinalIgnoreCase) => false,
                    { } flag when flag.Equals("true", StringComparison.OrdinalIgnoreCase) => true,
                    _ => IsRetryableStatus(status),
                };
                return new ApplicationFailureException(
                    $"Model call failed with HTTP {status}: {candidate.Message}",
                    exception,
                    errorType: MicrosoftAgentFrameworkErrorTypes.ModelError,
                    nonRetryable: !retryable,
                    nextRetryDelay: retryable ? RetryAfter(headers) : null);
            }
        }

        return null;
    }

    private static bool IsRetryableStatus(int status) => status is 408 or 409 or 429 || status >= 500;

    private static IEnumerable<Exception> Unwrap(Exception? exception)
    {
        while (exception is not null)
        {
            if (exception is AggregateException aggregate)
            {
                foreach (var flattened in aggregate.Flatten().InnerExceptions)
                {
                    foreach (var nested in Unwrap(flattened))
                    {
                        yield return nested;
                    }
                }

                yield break;
            }

            yield return exception;
            exception = exception.InnerException;
        }
    }

    private static bool TryGetHttpStatus(Exception exception, out int status)
    {
        status = exception is HttpRequestException { StatusCode: { } httpStatus } ? (int)httpStatus : ClientModelStatus(exception);
        return status is >= 400 and < 600;
    }

    // System.ClientModel.ClientResultException is the OpenAI and Azure OpenAI error type. It is
    // read by reflection so the plugin does not depend on the OpenAI package. Status is 0 when the
    // exception carries no response (a transport failure); that is not a status.
    private static int ClientModelStatus(Exception exception) =>
        exception.GetType().FullName == "System.ClientModel.ClientResultException" &&
        exception.GetType().GetProperty("Status") is { PropertyType.FullName: "System.Int32" } statusProperty &&
        statusProperty.GetValue(exception) is int clientModelStatus
            ? clientModelStatus
            : 0;

    private static TimeSpan? RetryAfter(ResponseHeaders headers)
    {
        if (headers.Get("retry-after-ms") is { } millisecondsHeader)
        {
            return long.TryParse(millisecondsHeader, NumberStyles.Integer, CultureInfo.InvariantCulture, out var milliseconds) && milliseconds >= 0
                ? TimeSpan.FromMilliseconds(milliseconds)
                : null;
        }

        var header = headers.Get("Retry-After");
        if (string.IsNullOrWhiteSpace(header))
        {
            return null;
        }

        if (int.TryParse(header, NumberStyles.Integer, CultureInfo.InvariantCulture, out var seconds) && seconds >= 0)
        {
            return TimeSpan.FromSeconds(seconds);
        }

        if (DateTimeOffset.TryParse(header, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var at))
        {
            var delay = at - DateTimeOffset.UtcNow;
            return delay > TimeSpan.Zero ? delay : null;
        }

        return null;
    }

    /// <summary>
    /// Header access on a <c>ClientResultException</c>'s raw response: <c>GetRawResponse()</c>
    /// returns a <c>PipelineResponse</c> whose <c>Headers</c> expose <c>TryGetValue(string, out string)</c>.
    /// All of it is reached by reflection; other exception types have no headers.
    /// </summary>
    private sealed class ResponseHeaders
    {
        private readonly object? headers;
        private readonly MethodInfo? tryGetValue;

        public ResponseHeaders(Exception exception)
        {
            try
            {
                var getRawResponse = exception.GetType().GetMethod("GetRawResponse", BindingFlags.Public | BindingFlags.Instance, Type.EmptyTypes);
                var response = getRawResponse?.Invoke(exception, null);
                headers = response?.GetType().GetProperty("Headers")?.GetValue(response);
                tryGetValue = headers?.GetType().GetMethod(
                    "TryGetValue",
                    BindingFlags.Public | BindingFlags.Instance,
                    [typeof(string), typeof(string).MakeByRefType()]);
            }
            catch (TargetInvocationException)
            {
                headers = null;
                tryGetValue = null;
            }
        }

        public string? Get(string name)
        {
            if (headers is null || tryGetValue is null)
            {
                return null;
            }

            try
            {
                var args = new object?[] { name, null };
                return tryGetValue.Invoke(headers, args) is true ? args[1] as string : null;
            }
            catch (TargetInvocationException)
            {
                return null;
            }
        }
    }
}
