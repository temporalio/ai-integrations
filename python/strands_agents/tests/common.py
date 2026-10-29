from pathlib import Path

from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHistory


def get_activities(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


def load_history(name: str, workflow_id: str = "legacy-replay") -> WorkflowHistory:
    """Load the recorded history ``tests/histories/<name>.json``."""
    path = Path(__file__).parent / "histories" / f"{name}.json"
    return WorkflowHistory.from_json(workflow_id, path.read_text())
