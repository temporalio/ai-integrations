"""The plugin leaves repeated activity names to the Worker, which refuses them.

The one exception is one instance on the Client and one on the Worker: the
Worker keeps what the Client side registered, and the Worker instance does not
add it again.
"""

from __future__ import annotations

import dataclasses
import functools
import uuid
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest

from temporalio import activity
from temporalio.client import Client
from temporalio.client import Plugin as ClientPlugin
from temporalio.common import RawValue
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


@activity.defn(dynamic=True)
async def fallback(_args: Sequence[RawValue]) -> str:
    return activity.info().activity_type


@activity.defn(dynamic=True)
async def other_fallback(_args: Sequence[RawValue]) -> str:
    return activity.info().activity_type


@dataclasses.dataclass
class _Endpoint:
    """An async callable; any two instances compare equal."""

    endpoint: str = dataclasses.field(compare=False)

    async def __call__(self, key: str) -> str:
        return f"{self.endpoint}:{key}"


@dataclasses.dataclass
class _DynamicEndpoint:
    """An async callable for a dynamic activity; any two instances compare equal."""

    endpoint: str = dataclasses.field(compare=False)

    async def __call__(self, _args: Sequence[RawValue]) -> str:
        return f"{self.endpoint}:{activity.info().activity_type}"


def _endpoint(endpoint: str) -> Callable[..., object]:
    return activity.defn(name="endpoint")(_Endpoint(endpoint))


def _dynamic_endpoint(endpoint: str) -> Callable[..., object]:
    return activity.defn(dynamic=True)(_DynamicEndpoint(endpoint))


def _client_with(client: Client, plugins: Sequence[ClientPlugin]) -> Client:
    config = client.config()
    config["plugins"] = list(plugins)
    return Client(**config)


def _own(plugin: DeepAgentsPlugin) -> list[Callable[..., object]]:
    return list(cast(Sequence[Callable[..., object]], plugin.activities))


@pytest.mark.parametrize(
    ("client_plugins", "worker_plugins", "activities", "worker_extra"),
    [
        (2, 0, [fetch], None),
        (0, 2, [fetch], None),
        (1, 2, [fetch], None),
        (1, 0, [fetch_v1, fetch], None),
        (1, 0, [caller_invoke_tool], None),
        (1, 1, [fetch], fetch_v1),
        (1, 1, [_endpoint("client")], _endpoint("worker")),
    ],
    ids=[
        "two-on-client",
        "two-on-worker",
        "one-on-client-two-on-worker",
        "caller-activities-share-a-name",
        "caller-activity-uses-plugin-name",
        "worker-instance-activity-repeats-caller-name",
        "worker-instance-activity-equals-caller-activity",
    ],
)
async def test_repeated_activity_names_are_refused(
    client: Client,
    client_plugins: int,
    worker_plugins: int,
    activities: list[Callable[..., object]],
    worker_extra: Callable[..., object] | None,
) -> None:
    on_worker = [DeepAgentsPlugin() for _ in range(worker_plugins)]
    if worker_extra is not None:
        # Another activity with a caller activity's name, given to the Worker
        # instance; the _Endpoint pair is distinct but compares equal.
        on_worker[0].activities = [*_own(on_worker[0]), worker_extra]
    with pytest.raises(ValueError, match="More than one activity named"):
        Worker(
            _client_with(client, [DeepAgentsPlugin() for _ in range(client_plugins)]),
            task_queue=f"activity-registration-{uuid.uuid4()}",
            activities=activities,
            plugins=on_worker,
        )


@pytest.mark.parametrize("handlers", ["functions", "equal-objects"])
@pytest.mark.parametrize("same_instance", [True, False])
async def test_two_dynamic_activities_are_refused(
    client: Client, same_instance: bool, handlers: str
) -> None:
    on_client = DeepAgentsPlugin()
    on_worker = on_client if same_instance else DeepAgentsPlugin()
    caller: Callable[..., object] = fallback
    extra: Callable[..., object] = other_fallback
    if handlers == "equal-objects":
        caller, extra = _dynamic_endpoint("client"), _dynamic_endpoint("worker")
    on_worker.activities = [*_own(on_worker), extra]
    with pytest.raises(TypeError, match="More than one dynamic activity"):
        Worker(
            _client_with(client, [on_client]),
            task_queue=f"activity-registration-{uuid.uuid4()}",
            activities=[caller],
            plugins=[on_worker],
        )


def _name(act: Callable[..., object]) -> str:
    return activity._Definition.must_from_callable(act).name or ""  # pyright: ignore[reportPrivateUsage]


def _wrap(act: Callable[..., Any], name: str | None = None) -> Callable[..., Any]:
    # updated=(): act.__dict__ holds its activity definition.
    @functools.wraps(act, updated=())
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await act(*args, **kwargs)

    return activity.defn(name=name or _name(act))(wrapper)


def _wrap_all(
    acts: Sequence[Callable[..., Any]] | None,
) -> Sequence[Callable[..., Any]]:
    return [_wrap(act) for act in acts or []]


def _alias_all(
    acts: Sequence[Callable[..., Any]] | None,
) -> Sequence[Callable[..., Any]]:
    # Replaces each plugin activity with a wrapper named "alias.<name>".
    out: list[Callable[..., Any]] = []
    for act in acts or []:
        if _name(act).startswith("deepagents."):
            act = _wrap(act, f"alias.{_name(act)}")
        out.append(act)
    return out


_BUILTIN_NAMES = [
    "deepagents.backend_op",
    "deepagents.invoke_model",
    "deepagents.invoke_model_streaming",
    "deepagents.invoke_tool",
]


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
        "same-instance-dynamic-extra",
        "same-instance-callable-dynamic-extra",
        "same-instance-copying-client-plugin",
        "same-instance-caller-copy",
        "client-alias",
        "same-instance-client-alias",
        "worker-callable-replaces",
    ],
)
async def test_one_instance_on_each_side_keeps_the_client_registration(
    client: Client, setup: str
) -> None:
    on_client = DeepAgentsPlugin()
    on_worker = on_client if setup.startswith("same-instance") else DeepAgentsPlugin()
    client_own = _own(on_client)
    worker_own = _own(on_worker)
    client_plugins: list[ClientPlugin] = [on_client]
    activities: list[Callable[..., object]] = [fetch]
    if setup.endswith("client-wrapper"):
        client_plugins.append(SimplePlugin("wrap", activities=_wrap_all))
    if setup.endswith("client-alias"):
        client_plugins.append(SimplePlugin("alias", activities=_alias_all))
    if setup.endswith("callable-replaces"):

        def replace_builtins(
            existing: Sequence[Callable[..., Any]] | None,
        ) -> Sequence[Callable[..., Any]]:
            kept = [a for a in existing or [] if not _name(a).startswith("deepagents.")]
            return [*kept, *worker_own]

        on_worker.activities = replace_builtins
    if setup.endswith("copying-client-plugin"):
        client_plugins.append(_CopyConfig())
    if setup.endswith("caller-copy"):
        activities = [fetch, client_own[0]]
    if setup.endswith("-extra"):
        extra = fallback if setup.endswith("dynamic-extra") else fetch
        given: list[Callable[..., Any]] = [*_own(on_worker), extra]
        if setup.endswith("callable-extra"):

            def add(
                existing: Sequence[Callable[..., Any]] | None,
            ) -> Sequence[Callable[..., Any]]:
                return [*(existing or []), *given]

            on_worker.activities = add
        else:
            on_worker.activities = given
        activities = [fetch] if extra is fallback else []

    worker = Worker(
        _client_with(client, client_plugins),
        task_queue=f"activity-registration-{uuid.uuid4()}",
        activities=activities,
        plugins=[on_worker],
    )

    registered = list(worker.config(active_config=True).get("activities") or [])
    aliases = [f"alias.{name}" for name in _BUILTIN_NAMES]
    expected = [*_BUILTIN_NAMES, "fetch"]
    if setup.endswith("dynamic-extra"):
        # The dynamic activity has no name; it is registered once.
        expected.append("")
    if setup.endswith("client-alias"):
        # The aliases do not stand in for the plugin's own names; both stay.
        expected = [*aliases, *expected]
    assert sorted(_name(act) for act in registered) == sorted(expected)
    for act in registered:
        if _name(act) in ("fetch", ""):
            continue
        if _name(act) in aliases:
            assert getattr(act, "__wrapped__", None) in client_own
        elif setup.endswith(("client-alias", "callable-replaces")):
            assert act in worker_own
        elif setup.endswith("client-wrapper"):
            assert getattr(act, "__wrapped__", None) in client_own
        else:
            assert act in client_own
