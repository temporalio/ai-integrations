using Temporalio.Api.Enums.V1;
using Temporalio.Api.History.V1;
using Temporalio.Client;
using Temporalio.Common;
using Temporalio.Converters;

namespace Temporalio.AgentFramework.Tests;

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
}
