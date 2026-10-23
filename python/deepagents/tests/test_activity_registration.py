"""The plugin leaves repeated activity names to the Worker, which refuses them.

The one exception is one instance on the Client and one on the Worker: the
Worker keeps what the Client side registered, and the Worker instance does not
add it again.
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest

from temporalio import activity
from temporalio.client import Client
from temporalio.client import Plugin as ClientPlugin
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker, WorkerConfig


@activity.defn
async def fetch(key: str) -> str:
    return key


@activity.defn(name="fetch")
async def fetch_v1(key: str) -> str:
    return key


@activity.defn(name="deepagents.invoke_tool")
async def caller_invoke_tool(key: str) -> str:
    return key


def _client_with(client: Client, plugins: Sequence[ClientPlugin]) -> Client:
    config = client.config()
    config["plugins"] = list(plugins)
    return Client(**config)


def _own(plugin: DeepAgentsPlugin) -> list[Callable[..., object]]:
    return list(cast(Sequence[Callable[..., object]], plugin.activities))


@pytest.mark.parametrize(
    ("client_plugins", "worker_plugins", "activities"),
    [
        (2, 0, [fetch]),
        (0, 2, [fetch]),
        (1, 2, [fetch]),
        (1, 0, [fetch_v1, fetch]),
        (1, 0, [caller_invoke_tool]),
        (1, 1, [fetch]),
    ],
    ids=[
        "two-on-client",
        "two-on-worker",
        "one-on-client-two-on-worker",
        "caller-activities-share-a-name",
        "caller-activity-uses-plugin-name",
        "worker-instance-activity-repeats-caller-name",
    ],
)
async def test_repeated_activity_names_are_refused(
    client: Client,
    client_plugins: int,
    worker_plugins: int,
    activities: list[Callable[..., object]],
) -> None:
    on_worker = [DeepAgentsPlugin() for _ in range(worker_plugins)]
    if client_plugins == worker_plugins == 1:
        # A second activity named "fetch", given to the Worker instance.
        on_worker[0].activities = [*_own(on_worker[0]), fetch_v1]
    with pytest.raises(ValueError, match="More than one activity named"):
        Worker(
            _client_with(client, [DeepAgentsPlugin() for _ in range(client_plugins)]),
            task_queue=f"activity-registration-{uuid.uuid4()}",
            activities=activities,
            plugins=on_worker,
        )


def _name(act: Callable[..., object]) -> str:
    return activity._Definition.must_from_callable(act).name or ""  # pyright: ignore[reportPrivateUsage]


def _wrap(act: Callable[..., Any]) -> Callable[..., Any]:
    # updated=(): act.__dict__ holds its activity definition.
    @functools.wraps(act, updated=())
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await act(*args, **kwargs)

    return activity.defn(name=_name(act))(wrapper)


def _wrap_all(
    acts: Sequence[Callable[..., Any]] | None,
) -> Sequence[Callable[..., Any]]:
    return [_wrap(act) for act in acts or []]


class _CopyConfig(SimplePlugin):
    """A Client plugin that hands the next plugin a copy of the config."""

    def __init__(self) -> None:
        super().__init__("copy-config")

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        return WorkerConfig(**super().configure_worker(config))


@pytest.mark.parametrize(
    "setup",
    [
        "same-instance",
        "two-instances",
        "client-wrapper",
        "same-instance-client-wrapper",
        "worker-extra",
        "same-instance-extra",
        "same-instance-callable-extra",
        "same-instance-copying-client-plugin",
        "same-instance-caller-copy",
    ],
)
async def test_one_instance_on_each_side_keeps_the_client_registration(
    client: Client, setup: str
) -> None:
    on_client = DeepAgentsPlugin()
    on_worker = on_client if setup.startswith("same-instance") else DeepAgentsPlugin()
    client_own = _own(on_client)
    client_plugins: list[ClientPlugin] = [on_client]
    activities: list[Callable[..., object]] = [fetch]
    if setup.endswith("client-wrapper"):
        client_plugins.append(SimplePlugin("wrap", activities=_wrap_all))
    if setup.endswith("copying-client-plugin"):
        client_plugins.append(_CopyConfig())
    if setup.endswith("caller-copy"):
        activities = [fetch, client_own[0]]
    if setup.endswith("-extra"):
        given: list[Callable[..., Any]] = [*_own(on_worker), fetch]
        if setup.endswith("callable-extra"):

            def add(
                existing: Sequence[Callable[..., Any]] | None,
            ) -> Sequence[Callable[..., Any]]:
                return [*(existing or []), *given]

            on_worker.activities = add
        else:
            on_worker.activities = given
        activities = []

    worker = Worker(
        _client_with(client, client_plugins),
        task_queue=f"activity-registration-{uuid.uuid4()}",
        activities=activities,
        plugins=[on_worker],
    )

    registered = list(worker.config(active_config=True).get("activities") or [])
    assert sorted(_name(act) for act in registered) == [
        "deepagents.backend_op",
        "deepagents.invoke_model",
        "deepagents.invoke_model_streaming",
        "deepagents.invoke_tool",
        "fetch",
    ]
    for act in registered:
        if _name(act) == "fetch":
            continue
        if setup.endswith("client-wrapper"):
            assert getattr(act, "__wrapped__", None) in client_own
        else:
            assert act in client_own
