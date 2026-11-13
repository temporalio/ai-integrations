using System.Globalization;
using System.Reflection;
using Temporalio.Exceptions;

namespace Temporalio.AgentFramework.Internal;

/// <summary>
/// Turns a provider exception into an <see cref="ApplicationFailureException"/> with
/// <see cref="AgentFrameworkErrorTypes.ModelError"/>, non-retryable for request errors that
/// repeat identically on retry, retryable otherwise, with <c>Retry-After</c> honored when the
/// provider sent one. Classification reads HTTP status codes, never message text. The status
/// extraction paths follow Cecil Phillip's draft at
/// https://github.com/temporal-community/temporal-dotnet-agents.
/// </summary>
internal static class ModelErrorClassifier
{
    private static readonly HashSet<int> NonRetryableStatuses = [400, 401, 403, 404, 422];

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
                var nonRetryable = NonRetryableStatuses.Contains(status);
                var retryAfter = nonRetryable ? null : TryGetRetryAfter(candidate);
                return new ApplicationFailureException(
                    $"Model call failed with HTTP {status}: {candidate.Message}",
                    exception,
                    errorType: AgentFrameworkErrorTypes.ModelError,
                    nonRetryable: nonRetryable,
                    nextRetryDelay: retryAfter);
            }
        }

        return null;
    }

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
        if (exception is HttpRequestException { StatusCode: { } httpStatus })
        {
            status = (int)httpStatus;
            return true;
        }

        // System.ClientModel.ClientResultException is the OpenAI and Azure OpenAI error type. It is
        // read by reflection so the plugin does not depend on the OpenAI package.
        if (exception.GetType().FullName == "System.ClientModel.ClientResultException" &&
            exception.GetType().GetProperty("Status") is { PropertyType.FullName: "System.Int32" } statusProperty &&
            statusProperty.GetValue(exception) is int clientModelStatus)
        {
            status = clientModelStatus;
            return true;
        }

        status = 0;
        return false;
    }

    private static TimeSpan? TryGetRetryAfter(Exception exception)
    {
        var header = TryGetRetryAfterHeader(exception);
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

    private static string? TryGetRetryAfterHeader(Exception exception)
    {
        // ClientResultException.GetRawResponse() returns a PipelineResponse whose Headers expose
        // TryGetValue(string, out string). All of it is reached by reflection, see above.
        try
        {
            var getRawResponse = exception.GetType().GetMethod("GetRawResponse", BindingFlags.Public | BindingFlags.Instance, Type.EmptyTypes);
            var response = getRawResponse?.Invoke(exception, null);
            var headers = response?.GetType().GetProperty("Headers")?.GetValue(response);
            if (headers is null)
            {
                return null;
            }

            var tryGetValue = headers.GetType().GetMethod(
                "TryGetValue",
                BindingFlags.Public | BindingFlags.Instance,
                [typeof(string), typeof(string).MakeByRefType()]);
            if (tryGetValue is null)
            {
                return null;
            }

            var args = new object?[] { "Retry-After", null };
            return tryGetValue.Invoke(headers, args) is true ? args[1] as string : null;
        }
        catch (TargetInvocationException)
        {
            return null;
        }
    }
}
