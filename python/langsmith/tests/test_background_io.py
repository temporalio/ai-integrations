"""Native LangSmith run defaults, background I/O, and workflow isolation."""

from __future__ import annotations

import logging
import random
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langsmith import get_current_run_tree, trace, traceable, tracing_context
from langsmith.run_trees import RunTree

from temporalio import workflow
from temporalio.langsmith import _run_tree
from tests.helpers.langsmith import InMemoryRunCollector, make_mock_ls_client

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def workflow_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[ThreadPoolExecutor]:
    """Install native hooks and bind a deterministic workflow runtime."""
    _run_tree.install_run_tree_overrides()
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(workflow, "now", lambda: NOW)
    monkeypatch.setattr(
        workflow.unsafe,
        "is_replaying_history_events",
        lambda: False,
    )
    monkeypatch.setattr(workflow.unsafe, "is_read_only", lambda: False)
    rng = random.Random(42)
    instance = SimpleNamespace()
    monkeypatch.setattr(workflow, "instance", lambda: instance)
    monkeypatch.setattr(workflow, "new_random", lambda: rng)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with _run_tree.workflow_run_tree_context(executor):
            yield executor


@pytest.mark.parametrize("api", ["run-tree", "trace", "traceable"])
def test_native_root_and_child_defaults(
    workflow_runtime: ThreadPoolExecutor, api: str
) -> None:
    """All public tracing entry points create real roots and deterministic children."""
    client = make_mock_ls_client(InMemoryRunCollector())
    if api == "run-tree":
        root = RunTree(name="root", ls_client=client)
        child = root.create_child(name="child")
        root.post()
        child.post()
    elif api == "trace":
        with trace("root", client=client) as root:
            with trace("child") as child:
                pass
    else:

        @traceable
        def child_call() -> RunTree:
            run = get_current_run_tree()
            assert run is not None
            return run

        @traceable
        def root_call() -> tuple[RunTree, RunTree]:
            run = get_current_run_tree()
            assert run is not None
            return run, child_call()

        with tracing_context(client=client):
            root, child = root_call()

    expected = random.Random(42)
    assert type(root) is RunTree
    assert root.id == uuid.UUID(int=expected.getrandbits(128), version=4)
    assert root.parent_run_id is None
    assert root.trace_id == root.id
    assert root.start_time == NOW
    assert type(child) is RunTree
    assert child.id == uuid.UUID(int=expected.getrandbits(128), version=4)
    assert child.parent_run_id == root.id
    assert child.trace_id == root.id
    assert child.start_time == NOW
    assert child.dotted_order.startswith(root.dotted_order + ".")
    workflow_runtime.submit(lambda: None).result()
    assert client.create_run.call_count == 2


def test_explicit_ids_and_times_are_preserved(
    workflow_runtime: ThreadPoolExecutor,  # type: ignore[reportUnusedParameter]
) -> None:
    """Native constructor, child creation, and trace preserve explicit caller values."""
    explicit_id = uuid.uuid4()
    explicit_time = NOW + timedelta(seconds=1)
    root = RunTree(name="root", id=explicit_id, start_time=explicit_time)
    child = root.create_child(
        name="child", run_id=explicit_id, start_time=explicit_time
    )
    assert (root.id, root.start_time) == (explicit_id, explicit_time)
    assert (child.id, child.start_time) == (explicit_id, explicit_time)
    client = make_mock_ls_client(InMemoryRunCollector())
    with trace("explicit-root", run_id=explicit_id, client=client) as run:
        assert run.id == explicit_id


@pytest.mark.parametrize("replaying", [False, True])
def test_end_keeps_in_memory_state_on_replay(
    workflow_runtime: ThreadPoolExecutor,
    monkeypatch: pytest.MonkeyPatch,
    replaying: bool,
) -> None:
    monkeypatch.setattr(
        workflow.unsafe,
        "is_replaying_history_events",
        lambda: replaying,
    )
    client = make_mock_ls_client(InMemoryRunCollector())
    run = RunTree(name="root", ls_client=client)
    run.post()
    run.end(outputs={"result": "done"})
    run.patch()
    workflow_runtime.submit(lambda: None).result()

    assert run.end_time == NOW
    assert run.outputs == {"result": "done"}
    assert client.create_run.call_count == (0 if replaying else 1)
    assert client.update_run.call_count == (0 if replaying else 1)


def test_delayed_post_and_patch_use_executor_in_order(
    workflow_runtime: ThreadPoolExecutor,
) -> None:
    """Queued writes preserve ordering and include finalized data without blocking."""
    client = make_mock_ls_client(InMemoryRunCollector())
    threads: list[threading.Thread] = []
    client.create_run.side_effect = lambda **_: threads.append(
        threading.current_thread()
    )
    client.update_run.side_effect = lambda **_: threads.append(
        threading.current_thread()
    )
    blocker = threading.Event()
    workflow_runtime.submit(blocker.wait, 5)
    run = RunTree(name="root", ls_client=client)
    try:
        run.post()
        run.end(outputs={"result": "done"})
        run.patch()
        client.create_run.assert_not_called()
        client.update_run.assert_not_called()
    finally:
        blocker.set()
    workflow_runtime.submit(lambda: None).result()

    assert [
        call[0] for call in client.mock_calls if call[0] in ("create_run", "update_run")
    ] == ["create_run", "update_run"]
    assert all(thread is not threading.current_thread() for thread in threads)
    for call in (client.create_run.call_args, client.update_run.call_args):
        assert call.kwargs["outputs"] == {"result": "done"}
        assert call.kwargs["end_time"] == NOW


@pytest.mark.parametrize("operation", ["post", "patch"])
def test_background_errors_are_logged(
    workflow_runtime: ThreadPoolExecutor,
    caplog: pytest.LogCaptureFixture,
    operation: str,
) -> None:
    client = make_mock_ls_client(InMemoryRunCollector())
    client.create_run.side_effect = RuntimeError("LangSmith API error")
    client.update_run.side_effect = RuntimeError("LangSmith API error")
    run = RunTree(name="root", ls_client=client)
    with caplog.at_level(logging.ERROR):
        getattr(run, operation)()
        workflow_runtime.submit(lambda: None).result()
    assert "LangSmith background I/O error: LangSmith API error" in caplog.text


@pytest.mark.parametrize("scope", ["outside-workflow", "without-plugin"])
def test_native_behavior_outside_plugin_workflow(
    workflow_runtime: ThreadPoolExecutor,  # type: ignore[reportUnusedParameter]
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    """Global hooks leave activity/client calls and other workflows synchronous."""
    if scope == "outside-workflow":
        monkeypatch.setattr(workflow, "in_workflow", lambda: False)
    token = _run_tree._executor.set(
        None if scope == "without-plugin" else workflow_runtime
    )
    try:
        client = make_mock_ls_client(InMemoryRunCollector())
        threads: list[threading.Thread] = []
        client.create_run.side_effect = lambda **_: threads.append(
            threading.current_thread()
        )
        run = RunTree(name="root", ls_client=client)
        run.post()
        run.end(outputs={"native": True})
        run.patch()
        assert run.start_time != NOW
        assert run.end_time != NOW
        assert threads == [threading.current_thread()]
        client.update_run.assert_called_once()
    finally:
        _run_tree._executor.reset(token)


def test_read_only_runs_do_not_consume_workflow_random(
    workflow_runtime: ThreadPoolExecutor,  # type: ignore[reportUnusedParameter]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries must leave the deterministic run stream untouched."""
    random_source = MagicMock()
    monkeypatch.setattr(workflow.unsafe, "is_read_only", lambda: True)
    monkeypatch.setattr(workflow, "new_random", random_source)
    run = RunTree(name="query")
    child = run.create_child(name="child")
    assert run.start_time == child.start_time == NOW
    random_source.assert_not_called()


def test_nested_context_retains_worker_client_fallback(
    workflow_runtime: ThreadPoolExecutor,
) -> None:
    """Clearing ambient context still uses the configured local client for roots."""
    fallback = make_mock_ls_client(InMemoryRunCollector())
    explicit = make_mock_ls_client(InMemoryRunCollector())
    with _run_tree.workflow_run_tree_context(workflow_runtime, fallback):
        with tracing_context():
            with trace("worker-client") as run:
                assert run.client is fallback
            with trace("explicit-client", client=explicit) as run:
                assert run.client is explicit
    workflow_runtime.submit(lambda: None).result()
    fallback.create_run.assert_called_once()
    explicit.create_run.assert_called_once()
