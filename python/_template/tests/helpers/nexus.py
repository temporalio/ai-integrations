"""Nexus helpers vendored from temporalio/sdk-python ``tests/helpers/nexus.py`` (origin/main).

Trimmed to what this plugin's tests import. Re-sync by hand (scripts/migrate/README.md).
"""


def make_nexus_endpoint_name(task_queue: str) -> str:
    # Create endpoints for different task queues without name collisions.
    return f"nexus-endpoint-{task_queue}"
