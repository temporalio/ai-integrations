"""The local streaming shim preserves contexts and refuses foreign threads."""

import asyncio
import contextvars
import threading
from typing import Any
from uuid import UUID

import pytest

from temporalio import workflow
from temporalio.langgraph._event_loop import (
    _workflow_run_id,
    _workflow_uuid_generator,
    enable_workflow_streams,
)


async def test_stream_callbacks_stay_on_the_workflow_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()

    def unsupported(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError

    # Model only the SDK loop's missing thread-safe method; scheduling uses the real loop.
    monkeypatch.setattr(loop, "call_soon_threadsafe", unsupported)
    monkeypatch.setattr(workflow, "patched", lambda _: True)
    try:
        enable_workflow_streams()
        marker = contextvars.ContextVar("marker", default="outside")
        context = contextvars.copy_context()
        context.run(marker.set, "callback")
        future = loop.create_future()
        loop.call_soon_threadsafe(
            lambda: future.set_result(marker.get()), context=context
        )
        assert await future == "callback"

        errors: list[type[BaseException]] = []

        def foreign_thread() -> None:
            try:
                loop.call_soon_threadsafe(lambda: None)
            except BaseException as exc:
                errors.append(type(exc))

        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == [NotImplementedError]
    finally:
        monkeypatch.undo()
        if hasattr(loop, "_temporal_langchain_streams"):
            delattr(loop, "_temporal_langchain_streams")


@pytest.mark.parametrize("in_workflow", [False, True])
def test_run_ids_preserve_native_behavior_without_patch(
    monkeypatch: pytest.MonkeyPatch, in_workflow: bool
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: in_workflow)
    monkeypatch.setattr(workflow, "patched", lambda _: False)
    monkeypatch.setattr(workflow.unsafe, "is_read_only", lambda: False)
    monkeypatch.setattr(
        workflow,
        "extern_functions",
        lambda: {"temporalio.langchain.run_id": _workflow_run_id},
    )
    calls: list[int | None] = []

    def original(nanoseconds: int | None = None) -> UUID:
        calls.append(nanoseconds)
        return UUID(int=17)

    generate = _workflow_uuid_generator(original)
    assert generate(123) == UUID(int=17)
    assert calls == [123]


@pytest.mark.parametrize("read_only", [True, False])
async def test_run_ids_do_not_patch_read_only_or_unconfigured_workflows(
    monkeypatch: pytest.MonkeyPatch, read_only: bool
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(workflow.unsafe, "is_read_only", lambda: read_only)
    monkeypatch.setattr(
        workflow,
        "extern_functions",
        lambda: {"temporalio.langchain.run_id": _workflow_run_id} if read_only else {},
    )

    def patched(_name: str) -> bool:
        pytest.fail("must not record a patch in this workflow")

    monkeypatch.setattr(workflow, "patched", patched)
    generate = _workflow_uuid_generator(lambda: UUID(int=17))
    assert generate() == UUID(int=17)


def test_run_ids_delegate_without_adding_an_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: False)
    generate = _workflow_uuid_generator(lambda: UUID(int=17))
    assert generate() == UUID(int=17)
