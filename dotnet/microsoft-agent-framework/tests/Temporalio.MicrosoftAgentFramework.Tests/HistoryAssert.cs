using Google.Protobuf;
using Google.Protobuf.Reflection;
using Temporalio.Api.Common.V1;
using Temporalio.Api.Enums.V1;
using Temporalio.Api.History.V1;
using Temporalio.Client;
using Temporalio.Common;
using Temporalio.Converters;

namespace Temporalio.MicrosoftAgentFramework.Tests;

/// <summary>Reads Event History the way the assertions need it.</summary>
public static class HistoryAssert
{
    public static IReadOnlyList<HistoryEvent> Scheduled(WorkflowHistory history, string? activityType = null) =>
        history.Events
            .Where(e => e.ActivityTaskScheduledEventAttributes is not null)
            .Where(e => activityType is null || e.ActivityTaskScheduledEventAttributes!.ActivityType.Name == activityType)
            .ToList();

    public static IReadOnlyDictionary<string, int> ActivityCounts(WorkflowHistory history) =>
        Scheduled(history)
            .GroupBy(e => e.ActivityTaskScheduledEventAttributes!.ActivityType.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);

    public static IReadOnlyList<string> ActivityTypesInOrder(WorkflowHistory history) =>
        Scheduled(history).Select(e => e.ActivityTaskScheduledEventAttributes!.ActivityType.Name).ToList();

    public static string? Summary(HistoryEvent scheduled, ITemporalClient client) =>
        scheduled.UserMetadata?.Summary is { } payload
            ? client.Options.DataConverter.PayloadConverter.ToValue<string>(payload)
            : null;

    public static int Count(WorkflowHistory history, EventType type) => history.Events.Count(e => e.EventType == type);

    public static int LastAttempt(WorkflowHistory history, string activityType)
    {
        var started = history.Events.Where(e => e.ActivityTaskStartedEventAttributes is not null).ToList();
        var scheduledIds = new HashSet<long>(Scheduled(history, activityType).Select(e => e.EventId));
        return started
            .Where(e => scheduledIds.Contains(e.ActivityTaskStartedEventAttributes!.ScheduledEventId))
            .Select(e => e.ActivityTaskStartedEventAttributes!.Attempt)
            .DefaultIfEmpty(0)
            .Max();
    }

    // The history as JSON plus the decoded body of every payload in it (inputs, results, memos, headers,
    // failure details), so a test can assert that a value never reached the server in any form.
    public static string AllText(WorkflowHistory history)
    {
        var parts = new List<string> { history.ToJson() };
        foreach (var @event in history.Events)
        {
            CollectPayloads(@event, parts);
        }

        return string.Join("\n", parts);
    }

    private static void CollectPayloads(IMessage message, List<string> parts)
    {
        if (message is Payload payload)
        {
            parts.Add(payload.Data.ToStringUtf8());
            return;
        }

        foreach (var field in message.Descriptor.Fields.InDeclarationOrder())
        {
            if (field.FieldType != FieldType.Message)
            {
                continue;
            }

            switch (field.Accessor.GetValue(message))
            {
                case IMessage single:
                    CollectPayloads(single, parts);
                    break;
                case System.Collections.IDictionary map:
                    foreach (var value in map.Values)
                    {
                        if (value is IMessage entry)
                        {
                            CollectPayloads(entry, parts);
                        }
                    }

                    break;
                case System.Collections.IEnumerable list:
                    foreach (var value in list)
                    {
                        if (value is IMessage entry)
                        {
                            CollectPayloads(entry, parts);
                        }
                    }

                    break;
                default:
                    break;
            }
        }
    }
}
