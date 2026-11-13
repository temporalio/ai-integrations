"""Shared helpers for the Deep Agents plugin test suite."""

from collections import Counter

from temporalio.api.enums.v1 import EventType
from temporalio.api.history.v1 import ActivityTaskScheduledEventAttributes
from temporalio.client import WorkflowHandle


async def count_scheduled_activities(handle: WorkflowHandle) -> Counter:
    """Count ``ActivityTaskScheduled`` events by activity-type name."""
    counts: Counter = Counter()
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            name = event.activity_task_scheduled_event_attributes.activity_type.name
            counts[name] += 1
    return counts


async def scheduled_attributes(
    handle: WorkflowHandle, activity_type: str
) -> ActivityTaskScheduledEventAttributes:
    """Return the schedule attributes of the first ``activity_type`` scheduled."""
    async for event in handle.fetch_history_events():
        if event.event_type != EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            continue
        attrs = event.activity_task_scheduled_event_attributes
        if attrs.activity_type.name == activity_type:
            return attrs
    raise AssertionError(f"No {activity_type!r} activity was scheduled")


async def all_scheduled_attributes(
    handle: WorkflowHandle, activity_type: str
) -> list[ActivityTaskScheduledEventAttributes]:
    """Return the schedule attributes of every ``activity_type`` scheduled, in order."""
    found: list[ActivityTaskScheduledEventAttributes] = []
    async for event in handle.fetch_history_events():
        if event.event_type != EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            continue
        attrs = event.activity_task_scheduled_event_attributes
        if attrs.activity_type.name == activity_type:
            found.append(attrs)
    if not found:
        raise AssertionError(f"No {activity_type!r} activity was scheduled")
    return found
