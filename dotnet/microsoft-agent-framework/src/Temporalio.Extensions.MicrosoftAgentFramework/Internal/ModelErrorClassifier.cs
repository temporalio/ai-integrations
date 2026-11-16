using System.Globalization;
using System.Net;
using System.Reflection;
using Temporalio.Exceptions;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Internal;

/// <summary>
/// Turns a provider exception into an <see cref="ApplicationFailureException"/> with
/// <see cref="MicrosoftAgentFrameworkErrorTypes.ModelError"/>. The rule is the one the Python
/// plugin applies: 408, 409, 429 and 5xx are retryable, every other 4xx is not; a provider
/// <c>x-should-retry</c> header overrides it, and <c>retry-after-ms</c> or <c>Retry-After</c>
/// (seconds, with a fraction, or an HTTP date) becomes the next retry delay, capped at 60 seconds
/// as the OpenAI SDK caps it; a value that cannot be read is skipped. Classification reads HTTP
/// status codes and headers, never message text. The status extraction paths follow Cecil
/// Phillip's draft at https://github.com/temporal-community/temporal-dotnet-agents.
/// </summary>
internal static class ModelErrorClassifier
{
    /// <summary>Longest delay a provider header can ask for; the OpenAI SDK ignores longer ones.</summary>
    public static readonly TimeSpan MaxRetryAfter = TimeSpan.FromSeconds(60);

    private static readonly string[] StatusPropertyNames = ["Status", "StatusCode", "HttpStatusCode"];

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
        status = exception is HttpRequestException { StatusCode: { } httpStatus } ? (int)httpStatus : StatusProperty(exception);
        return status is >= 400 and < 600;
    }

    // The provider SDKs' exception types carry the HTTP status as an int or HttpStatusCode property:
    // System.ClientModel.ClientResultException (OpenAI, Azure OpenAI) and Azure.RequestFailedException
    // (Azure AI Inference) expose Status, others StatusCode. Read by reflection, so the plugin depends
    // on no provider package. A value of 0 means the exception carries no response; that is not a status.
    private static int StatusProperty(Exception exception)
    {
        foreach (var name in StatusPropertyNames)
        {
            var property = exception.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(candidate => candidate.Name == name && candidate.GetIndexParameters().Length == 0);
            switch (property?.GetValue(exception))
            {
                case int value:
                    return value;
                case HttpStatusCode code:
                    return (int)code;
                default:
                    continue;
            }
        }

        return 0;
    }

    private static TimeSpan? RetryAfter(ResponseHeaders headers)
    {
        if (headers.Get("retry-after-ms") is { } millisecondsHeader &&
            double.TryParse(millisecondsHeader, NumberStyles.Float, CultureInfo.InvariantCulture, out var milliseconds) &&
            milliseconds >= 0)
        {
            return Cap(TimeSpan.FromMilliseconds(milliseconds));
        }

        var header = headers.Get("Retry-After")?.Trim();
        if (string.IsNullOrEmpty(header))
        {
            return null;
        }

        if (double.TryParse(header, NumberStyles.Float, CultureInfo.InvariantCulture, out var seconds) && seconds >= 0)
        {
            return Cap(TimeSpan.FromSeconds(seconds));
        }

        if (DateTimeOffset.TryParse(header, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var at))
        {
            var delay = at - DateTimeOffset.UtcNow;
            return delay > TimeSpan.Zero ? Cap(delay) : null;
        }

        return null;
    }

    private static TimeSpan Cap(TimeSpan delay) => delay > MaxRetryAfter ? MaxRetryAfter : delay;

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
