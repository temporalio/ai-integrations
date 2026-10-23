"""The plugin leaves repeated activity names to the Worker, which refuses them."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

from temporalio import activity
from temporalio.client import Client
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.worker import Worker


@activity.defn
async def fetch(key: str) -> str:
    return key


@activity.defn(name="fetch")
async def fetch_v1(key: str) -> str:
    return key


@activity.defn(name="deepagents.invoke_tool")
async def caller_invoke_tool(key: str) -> str:
    return key


def _client_with(client: Client, plugins: list[DeepAgentsPlugin]) -> Client:
    config = client.config()
    config["plugins"] = plugins
    return Client(**config)


@pytest.mark.parametrize(
    ("client_plugins", "worker_plugins", "activities"),
    [
        (2, 0, [fetch]),
        (0, 2, [fetch]),
        (1, 2, [fetch]),
        (1, 0, [fetch_v1, fetch]),
        (1, 0, [caller_invoke_tool]),
    ],
    ids=[
        "two-on-client",
        "two-on-worker",
        "one-on-client-two-on-worker",
        "caller-activities-share-a-name",
        "caller-activity-uses-plugin-name",
    ],
)
async def test_repeated_activity_names_are_refused(
    client: Client,
    client_plugins: int,
    worker_plugins: int,
    activities: list[Callable[..., object]],
) -> None:
    with pytest.raises(ValueError, match="More than one activity named"):
        Worker(
            _client_with(client, [DeepAgentsPlugin() for _ in range(client_plugins)]),
            task_queue=f"activity-registration-{uuid.uuid4()}",
            activities=activities,
            plugins=[DeepAgentsPlugin() for _ in range(worker_plugins)],
        )
