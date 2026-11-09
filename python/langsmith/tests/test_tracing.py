"""Plugin destinations act as defaults without overriding native addressing."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import langsmith
import pytest
from langsmith import (
    get_current_run_tree,
    run_helpers,
    trace,
    traceable,
    tracing_context,
)
from langsmith.run_trees import RunTree

from temporalio.langsmith import LangSmithInterceptor, _tracing
from temporalio.langsmith._tracing import destination_context
from tests.helpers.langsmith import langsmith_release

ADDRESS = "lrn:agents/support/environments/staging"
OTHER_ADDRESS = "lrn:agents/support/environments/production"
_DESTINATION = {"project_name": "plugin-project"}
_ADDRESS_DEFAULT = pytest.param(
    {"address": ADDRESS},
    id="address",
    marks=pytest.mark.skipif(
        not hasattr(langsmith, "address"), reason="Agent addressing requires 0.14.4"
    ),
)


def _destination(run: RunTree | None) -> dict[str, str]:
    assert run is not None
    address = getattr(run, "address", None)
    if address:
        return {"address": address}
    assert isinstance(run.session_name, str)
    return {"project_name": run.session_name}


@pytest.mark.parametrize(
    "entry", ["workflow", "activity", "nexus-start", "nexus-cancel"]
)
@pytest.mark.parametrize(
    "fallback", [{"project_name": "plugin-project"}, _ADDRESS_DEFAULT]
)
@pytest.mark.parametrize("explicit", [False, True])
async def test_interceptor_destination_defaults(
    entry: str, fallback: dict[str, Any], explicit: bool
) -> None:
    """Sync/async decorators and trace blocks can override every inbound default."""
    mock = MagicMock()
    config = LangSmithInterceptor(client=mock, **fallback)
    override: dict[str, Any] = {"project_name": "explicit-project"} if explicit else {}
    expected = override or fallback

    @traceable(**override)
    def sync_call() -> dict[str, str]:
        return _destination(get_current_run_tree())

    @traceable(**override)
    async def async_call() -> dict[str, str]:
        return _destination(get_current_run_tree())

    async def execute(_input: Any) -> None:
        assert sync_call() == expected
        assert await async_call() == expected
        with trace("sync-block", **override) as run:
            assert _destination(run) == expected
        async with trace("async-block", **override) as run:
            assert _destination(run) == expected

    next_interceptor = MagicMock()
    input = MagicMock()
    input.headers = {}
    input.ctx.headers = {}
    interceptor: Any
    if entry == "workflow":
        cls = config.workflow_interceptor_class(MagicMock())
        interceptor = cls(next_interceptor)
        method = "execute_workflow"
    elif entry == "activity":
        interceptor = config.intercept_activity(next_interceptor)
        method = "execute_activity"
    else:
        interceptor = config.intercept_nexus_operation(next_interceptor)
        method = f"execute_nexus_operation_{entry.removeprefix('nexus-')}"
    setattr(next_interceptor, method, AsyncMock(side_effect=execute))
    try:
        with (
            patch("temporalio.workflow.info"),
            patch("temporalio.activity.info"),
        ):
            await getattr(interceptor, method)(input)
        # The plugin's default must not leak to unrelated LangSmith code.
        with tracing_context(client=mock):
            assert sync_call() == (override or {"project_name": "default"})
    finally:
        config._executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "fallback", [{"project_name": "plugin-project"}, _ADDRESS_DEFAULT]
)
@pytest.mark.parametrize(
    "source", ["decorator", "extra", "context", "parent", "configure", "env"]
)
def test_traceable_destination_precedence(
    monkeypatch: pytest.MonkeyPatch, fallback: dict[str, Any], source: str
) -> None:
    """Explicit destinations outrank the plugin; configure and env follow it."""
    mock = MagicMock()
    config = LangSmithInterceptor(client=mock, **fallback)
    # Cross destination modes where supported, including address over project.
    explicit: dict[str, Any] = (
        {"address": OTHER_ADDRESS}
        if hasattr(langsmith, "address") and "project_name" in fallback
        else {"project_name": "explicit-project"}
    )
    monkeypatch.setenv("LANGSMITH_PROJECT", "env-project")
    if source == "configure":
        from langsmith._internal import _context

        monkeypatch.setattr(_context, "_GLOBAL_PROJECT_NAME", "configured-project")

    @traceable(**(explicit if source == "decorator" else {}))
    def call() -> dict[str, str]:
        return _destination(get_current_run_tree())

    kwargs: dict[str, Any] = {}
    if source == "context":
        kwargs.update(explicit)
    elif source == "parent":
        kwargs["parent"] = RunTree(name="parent", ls_client=mock, **explicit)
    try:
        with destination_context(fallback), tracing_context(client=mock, **kwargs):
            actual = call(langsmith_extra=explicit if source == "extra" else {})  # type: ignore[call-arg]
        assert actual == (fallback if source in ("configure", "env") else explicit)
    finally:
        config._executor.shutdown(wait=True)


async def test_concurrent_destination_defaults() -> None:
    """Defaults stay local to each task and do not mutate a shared decorator."""
    mock = MagicMock()
    config = LangSmithInterceptor(client=mock)

    @traceable
    async def call() -> str:
        await asyncio.sleep(0)
        run = get_current_run_tree()
        assert run is not None
        return str(run.session_name)

    async def execute(project: str) -> str:
        with (
            destination_context({"project_name": project}),
            tracing_context(client=mock),
        ):
            await asyncio.sleep(0)
            return await call()

    try:
        assert await asyncio.gather(execute("first"), execute("second")) == [
            "first",
            "second",
        ]
        with tracing_context(client=mock):
            assert await call() == "default"
    finally:
        config._executor.shutdown(wait=True)


@pytest.fixture
def originals(monkeypatch: pytest.MonkeyPatch) -> dict[str, Callable[..., Any]]:
    """Remove the addressing override for one test, returning LangSmith's originals."""
    monkeypatch.setattr(_tracing, "_installed", False)
    found: dict[str, Callable[..., Any]] = {}
    for name in (
        "_resolve_traceable_addressing",
        "_addressing_tiers",
        "_setup_run",
        "_get_project_name",
    ):
        found[name] = inspect.unwrap(getattr(run_helpers, name))
        monkeypatch.setattr(run_helpers, name, found[name])
    return found


def _install_over(
    monkeypatch: pytest.MonkeyPatch, name: str, release: Callable[..., Any]
) -> None:
    if name != "_resolve_traceable_addressing":
        # LangSmith before 0.14.2 resolves addressing in _setup_run instead.
        monkeypatch.delattr(run_helpers, "_resolve_traceable_addressing")
        monkeypatch.delattr(run_helpers, "_addressing_tiers")
    monkeypatch.setattr(run_helpers, name, release)
    _tracing.install_addressing_override()


@pytest.mark.parametrize("in_context", [False, True])
@pytest.mark.parametrize(
    ("name", "args", "kwargs", "read", "default"),
    [
        pytest.param(
            "_resolve_traceable_addressing",
            (None, {}, {}),
            {},
            "container_input",
            _DESTINATION,
            id="_resolve_traceable_addressing",
        ),
        pytest.param(
            "_setup_run",
            (len,),
            {"container_input": {}},
            "container_input",
            _DESTINATION,
            id="_setup_run",
        ),
        pytest.param(
            "_get_project_name",
            (None,),
            {},
            "project_name",
            "plugin-project",
            id="_get_project_name",
        ),
    ],
)
def test_addressing_override_forwards_unknown_arguments(
    originals: dict[str, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
    in_context: bool,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    read: str,
    default: Any,
) -> None:
    """A keyword from a newer LangSmith reaches the original through each override."""
    calls: list[inspect.BoundArguments] = []
    release = langsmith_release(originals[name], calls)
    _install_over(monkeypatch, name, release)

    with destination_context(_DESTINATION) if in_context else contextlib.nullcontext():
        getattr(run_helpers, name)(*args, **kwargs, added=1)

    if not in_context:
        assert calls == [inspect.signature(release).bind(*args, **kwargs, added=1)]
    else:
        assert [call.arguments["added"] for call in calls] == [1]
        assert calls[0].arguments[read] == default


@pytest.mark.parametrize("in_context", [False, True])
@pytest.mark.parametrize(
    ("name", "args", "renamed"),
    [
        pytest.param(
            "_resolve_traceable_addressing",
            (None, {}, {}),
            "container_input",
            id="_resolve_traceable_addressing",
        ),
        pytest.param("_setup_run", (len, {}), "container_input", id="_setup_run"),
        pytest.param(
            "_get_project_name", (None,), "project_name", id="_get_project_name"
        ),
    ],
)
def test_addressing_override_calls_original_unchanged_after_renames(
    originals: dict[str, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
    in_context: bool,
    name: str,
    args: tuple[Any, ...],
    renamed: str,
) -> None:
    """An override that can't find the parameter it reads leaves the call alone."""
    calls: list[inspect.BoundArguments] = []
    release = langsmith_release(originals[name], calls, renamed=(renamed,))
    _install_over(monkeypatch, name, release)

    with destination_context(_DESTINATION) if in_context else contextlib.nullcontext():
        getattr(run_helpers, name)(*args)

    assert calls == [inspect.signature(release).bind(*args)]
