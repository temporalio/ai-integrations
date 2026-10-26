"""The conversation lives in the Workflow (the default): read with a Query, spliced per step.

Without a session store, each Workflow holds its agent's conversation. A step reads it
with a Query on its own Workflow, a page at a time, so the history grows only with what
each step adds; it returns what it changed, and every attempt starts from what the
Workflow committed. These tests pin down the parts that are new: paging, payload
limits, Continue-As-New, Workers configured differently, and nothing left on disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.api.common.v1 import Payload, WorkflowExecution
from temporalio.api.enums.v1 import EventType
from temporalio.api.workflowservice.v1 import ResetWorkflowExecutionRequest
from temporalio.claude_agent_sdk import (
    AgentState,
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    ConversationRef,
    DurableClaudeAgent,
    FileSessionStore,
    SegmentInput,
    ToolOutcome,
    ToolSpec,
    _conversation,
    _runner,
)
from temporalio.claude_agent_sdk._workflow import _Conversations
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowQueryFailedError
from temporalio.converter import DataConverter, ExternalStorage
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Worker
from tests.conversation.workflows import HandOverWorkflow, PatientWorkflow
from tests.endless.activities import ALL as COUNTING
from tests.endless.policy import count_policy
from tests.endless.workflows import LongTaskWorkflow, TaskOptions
from tests.helpers.fake_messages_api import engine_env, start_with_policy
from tests.helpers.storage import FolderStorageDriver
from tests.helpers.workers import FAIL_FAST
from tests.lifecycle.policy import big_result_policy
from tests.lifecycle.workflows import BigResultWorkflow
from tests.refund import shop
from tests.refund.policy import refund_policy
from tests.storage.workflows import (
    FetchWorkflow,
    ParallelFetchWorkflow,
    fetch_document,
    pages_policy,
    parallel_fetch_policy,
)

pytestmark = pytest.mark.timeout(240)
QUERY = _conversation.QUERY
PAGE = list[dict[str, Any]]
COUNT_TOOLS = [ToolSpec("count", "Count one step.", {"type": "object"})]


def worker(client: Client, queue: str, runner: Any) -> Worker:
    """A Worker for every Workflow these tests use."""
    return Worker(
        client,
        task_queue=queue,
        workflows=[
            LongTaskWorkflow,
            PatientWorkflow,
            HandOverWorkflow,
            BigResultWorkflow,
            FetchWorkflow,
            ParallelFetchWorkflow,
        ],
        activities=[*COUNTING, fetch_document],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    )


def with_external_storage(client: Client, folder: Path) -> Client:
    """The same server, with External Storage in a folder."""
    config = client.config()
    config["data_converter"] = DataConverter(
        external_storage=ExternalStorage(drivers=[FolderStorageDriver(folder)])
    )
    return Client(**config)


# ---- pages and sizes ----


def test_pages_hold_whole_entries_up_to_the_limit() -> None:
    entries = [
        {"uuid": str(i), "text": "x" * n} for i, n in enumerate((10, 10, 90, 10))
    ]
    sizes = [_conversation.entry_bytes(e) for e in entries]
    two = len(json.dumps(entries[:2], separators=(",", ":")))  # the page, as JSON
    assert _conversation.page(entries, sizes, 0, limit=two) == entries[:2]
    assert _conversation.page(entries, sizes, 0, limit=two - 1) == entries[:1]
    alone = _conversation.page(entries, sizes, 2, limit=10)  # over the limit, alone
    assert alone == entries[2:3]
    assert _conversation.page(entries, sizes, 3) == entries[3:]
    assert _conversation.page(entries, sizes, 4) == []


def test_a_page_is_the_same_payload_however_long_the_conversation() -> None:
    """A content-addressed store keeps a page over the threshold once, not per step."""
    converter = DataConverter.default.payload_converter
    short = [{"uuid": "a", "text": "y"}, {"uuid": "b", "text": "x" * 500}]
    pages = []
    for entries in (short, [*short, {"uuid": "c", "text": "z"}]):
        sizes = [_conversation.entry_bytes(e) for e in entries]
        pages.append(_conversation.page(entries, sizes, 1, limit=100))
    first, second = (converter.to_payloads([p])[0].SerializeToString() for p in pages)
    assert pages[0] == short[1:] and first == second
    # page() counts the bytes as Temporal's converter writes them.
    sizes = [_conversation.entry_bytes(e) for e in short]
    data = converter.to_payloads([short])[0].data
    assert len(data) == 1 + sum(n + 1 for n in sizes)
    assert _conversation.page(short, sizes, 0, limit=len(data)) == short


@pytest.mark.parametrize(
    ("threshold", "limit"),
    [
        (None, _conversation.PAGE_BYTES),
        (256 * 1024, 252 * 1024),
        (16 * 1024, _conversation.SMALLEST_PAGE),
        (8 * 1024 * 1024, _conversation.PAGE_BYTES),
    ],
    ids=["no-storage", "default-threshold", "low-threshold", "high-threshold"],
)
async def test_pages_stay_under_the_external_storage_threshold(
    tmp_path: Path, threshold: int | None, limit: int
) -> None:
    converter = DataConverter.default
    if threshold is not None:
        converter = DataConverter(
            external_storage=ExternalStorage(
                drivers=[FolderStorageDriver(tmp_path)],
                payload_size_threshold=threshold,
            )
        )
    client = _FakeClient(_FakeHandle([], []), converter)
    env = ActivityEnvironment(client=client)  # type: ignore[arg-type]

    async def in_a_segment() -> int:  # the client is there for async Activities only
        return _conversation.page_limit()

    assert await env.run(in_a_segment) == limit


def test_sizes_match_temporals_json_converter() -> None:
    """Non-ASCII text is escaped in payloads: Arabic and Chinese grow, emoji double."""
    entry = {"uuid": "u", "text": "مرحبا 你好 😀", "n": [1, 2.5, None, True]}
    payload = DataConverter.default.payload_converter.to_payloads([entry])[0]
    assert _conversation.entry_bytes(entry) == len(payload.data)
    assert _conversation.entry_bytes(entry) > len(json.dumps(entry, ensure_ascii=False))


class _FakeHandle:
    def __init__(self, entries: list[dict[str, Any]], asked: list[int]) -> None:
        self.entries, self.asked = entries, asked

    async def query(self, name: str, args: list[Any], result_type: Any) -> Any:
        assert name == QUERY and result_type == PAGE
        agent, start, total, limit = args
        assert agent == "0" and limit == _conversation.PAGE_BYTES
        self.asked.append(start)
        if total != len(self.entries):
            raise WorkflowQueryFailedError(f"scheduled with {total}")
        sizes = [_conversation.entry_bytes(e) for e in self.entries]
        return _conversation.page(self.entries, sizes, start, limit=100)


class _FakeClient:
    def __init__(
        self, handle: _FakeHandle, converter: DataConverter = DataConverter.default
    ) -> None:
        self.handle, self.data_converter = handle, converter

    def get_workflow_handle(self, workflow_id: str, run_id: str | None = None) -> Any:
        assert workflow_id and run_id
        return self.handle


async def test_a_step_reads_the_conversation_a_page_at_a_time() -> None:
    entries = [{"uuid": str(i), "text": "x" * 60} for i in range(5)]
    asked: list[int] = []
    env = ActivityEnvironment(client=_FakeClient(_FakeHandle(entries, asked)))  # type: ignore[arg-type]
    inp = SegmentInput(
        session_id="s",
        prompt=None,
        tools=[],
        checkpoint="4",
        conversation=ConversationRef(query=QUERY, agent="0", entries=5),
    )
    got = await env.run(_conversation.read_conversation, inp)
    assert got == entries and asked == [0, 1, 2, 3, 4]  # one 80-byte entry per page
    inp.conversation = ConversationRef(query=QUERY, agent="0", entries=4)
    with pytest.raises(RuntimeError, match="scheduled with 4. Retrying"):  # changed
        await env.run(_conversation.read_conversation, inp)


def _held(entries: int) -> SegmentInput:
    """A segment of a conversation of ``entries`` entries that the Workflow holds."""
    return SegmentInput(
        session_id="s",
        prompt=None,
        tools=[],
        checkpoint=str(entries - 1) if entries else None,
        conversation=ConversationRef(query=QUERY, agent="0", entries=entries),
    )


async def test_a_conversation_of_one_entry_is_read() -> None:
    entries = [{"uuid": "0", "text": "the only entry"}]
    asked: list[int] = []
    env = ActivityEnvironment(client=_FakeClient(_FakeHandle(entries, asked)))  # type: ignore[arg-type]
    assert await env.run(_conversation.read_conversation, _held(1)) == entries
    assert asked == [0]


async def test_a_step_with_no_conversation_to_read_asks_nothing() -> None:
    """No Query for a new conversation; a transcript passed inline wins."""
    asked: list[int] = []
    env = ActivityEnvironment(client=_FakeClient(_FakeHandle([], asked)))  # type: ignore[arg-type]
    for inp in (_held(0), SegmentInput(session_id="s", prompt="hi", tools=[])):
        assert await env.run(_conversation.read_conversation, inp) == []
        assert await _conversation.read_conversation(inp) == []  # even outside one
    inline = _held(3)
    inline.transcript = [{"uuid": "t"}]
    got = await env.run(_conversation.read_conversation, inline)
    assert got == [{"uuid": "t"}] and got is not inline.transcript
    assert asked == []


class _EmptyHandle(_FakeHandle):
    async def query(self, name: str, args: list[Any], result_type: Any) -> Any:
        self.asked.append(args[1])
        return []


async def test_an_empty_page_fails_the_step_so_temporal_retries_it() -> None:
    asked: list[int] = []
    env = ActivityEnvironment(client=_FakeClient(_EmptyHandle([], asked)))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError) as raised:
        await env.run(_conversation.read_conversation, _held(2))
    assert str(raised.value) == "The Workflow served no entries at 0 of 2. Retrying."
    assert asked == [0]  # asked once: an empty page never loops


async def test_a_held_conversation_is_read_only_in_a_segment_activity() -> None:
    with pytest.raises(RuntimeError) as outside:
        await _conversation.read_conversation(_held(2))
    assert str(outside.value) == (
        "A conversation the Workflow holds can only be read inside the segment "
        "Activity; pass SegmentInput.transcript to run a segment directly."
    )
    asked: list[int] = []
    env = ActivityEnvironment(client=_FakeClient(_FakeHandle([], asked)))  # type: ignore[arg-type]
    env.info = dataclasses.replace(env.info, workflow_id=None)  # a standalone Activity
    with pytest.raises(RuntimeError) as standalone:
        await env.run(_conversation.read_conversation, _held(2))
    assert str(standalone.value) == (
        "This segment Activity was not started by a Workflow, so no Workflow holds "
        "its conversation; pass SegmentInput.transcript instead."
    )
    assert asked == []


async def test_a_query_that_fails_fails_the_step_so_temporal_retries_it() -> None:
    env = ActivityEnvironment(client=_FakeClient(_FakeHandle([{"uuid": "0"}], [])))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError) as raised:
        await env.run(_conversation.read_conversation, _held(2))  # it holds 1
    assert str(raised.value) == (
        "The Workflow did not serve the conversation: scheduled with 2. Retrying."
    )
    assert isinstance(raised.value.__cause__, WorkflowQueryFailedError)


async def test_without_a_client_a_step_asks_for_whole_pages() -> None:
    """For example in an ActivityEnvironment: no client, so no External Storage."""

    async def in_a_segment() -> tuple[int, bool]:
        return _conversation.page_limit(), _conversation.external_storage_on()

    assert await ActivityEnvironment().run(in_a_segment) == (
        _conversation.PAGE_BYTES,
        False,
    )
    assert _conversation.page_limit() == _conversation.PAGE_BYTES  # outside one


async def test_a_step_knows_when_its_worker_has_external_storage(
    tmp_path: Path,
) -> None:
    storage = DataConverter(
        external_storage=ExternalStorage(drivers=[FolderStorageDriver(tmp_path)])
    )
    for converter, on in ((DataConverter.default, False), (storage, True)):
        client = _FakeClient(_FakeHandle([], []), converter)
        env = ActivityEnvironment(client=client)  # type: ignore[arg-type]

        async def in_a_segment() -> bool:
            return _conversation.external_storage_on()

        assert await env.run(in_a_segment) is on


def test_the_query_serves_each_agent_a_page_of_its_own_conversation() -> None:
    """The handler a segment's Query reaches (read-only; refuses out-of-date steps)."""
    big = [{"uuid": str(i), "text": "x" * (200 * 1024)} for i in range(3)]
    small = [{"uuid": "a"}, {"uuid": "b"}]
    registry = _Conversations()
    for key, entries in (("0", big), ("1", small)):
        state = AgentState(conversation=[_conversation.entry_text(e) for e in entries])
        registry.agents[key] = DurableClaudeAgent(state=state)
    assert registry.serve("1", 0, 2, 10**9) == small
    assert registry.serve("1", 1, 2, 10**9) == small[1:]
    assert registry.serve("1", 0, 2, 0) == small[:1]  # a limit below 1 counts as 1
    assert registry.serve("0", 0, 3, 10**9) == big[:2]  # never more than a full page
    with pytest.raises(ValueError) as stale:
        registry.serve("1", 0, 3, 10**9)
    assert str(stale.value) == (
        "The conversation has 2 entries, but the step that asks was scheduled with 3."
    )
    with pytest.raises(ValueError, match="This Workflow has no agent '2'"):
        registry.serve("2", 0, 2, 10**9)


# ---- the scripted runner holds its conversation too ----


async def test_scripted_claude_keeps_its_state_in_the_conversation(
    tmp_path: Path,
) -> None:
    runner = ScriptedClaude(count_policy)
    transcript: list[dict[str, Any]] = []
    out = await runner.run(
        SegmentInput(session_id="s", prompt="count to 2", tools=[], transcript=[]), 1
    )
    calls = []
    while out.deferred is not None:
        assert out.transcript_keep == len(transcript) and len(out.transcript_add) == 1
        transcript += out.transcript_add
        calls.append(out.deferred.input["n"])
        out = await runner.run(
            SegmentInput(
                session_id="s",
                prompt=None,
                tools=[],
                checkpoint=out.checkpoint,
                injected={out.deferred.id: ToolOutcome({"n": out.deferred.input["n"]})},
                transcript=list(transcript),
            ),
            1,
        )
    assert out.result == "counted to 2" and calls == [1, 2]
    # Where the conversation is must agree with the runner, or the step stops.
    wrong = SegmentInput(
        session_id="s", prompt=None, tools=[], checkpoint="cp_x", transcript=transcript
    )
    bad = await runner.run(wrong, 1)
    assert bad.is_error and bad.error is not None and "cp_x" in bad.error
    with pytest.raises(RuntimeError, match="kept in a folder"):
        await runner.run(
            SegmentInput(session_id="s", prompt="hi", tools=[], checkpoint="cp_x"), 1
        )
    with pytest.raises(RuntimeError, match="kept in its Workflow"):
        await ScriptedClaude(count_policy, tmp_path / "folder").run(
            SegmentInput(session_id="s", prompt="hi", tools=[], transcript=transcript),
            1,
        )


# ---- in a Workflow ----


@pytest.mark.usefixtures("shop_dir")
async def test_each_step_reads_the_conversation_and_records_only_what_it_added(
    client: Client,
) -> None:
    queue = f"held-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 3", TaskOptions(continue_as_new=False), None],
            id=queue,
            task_queue=queue,
        )
        assert await asyncio.wait_for(handle.result(), 60) == "counted to 3"
        held = await handle.query(
            QUERY, args=["0", 0, 4, _conversation.PAGE_BYTES], result_type=PAGE
        )
        with pytest.raises(WorkflowQueryFailedError, match="scheduled with 3"):
            await handle.query(QUERY, args=["0", 0, 3, 1000], result_type=PAGE)
        steps_in: list[dict[str, Any]] = []
        steps_out: list[dict[str, Any]] = []
        scheduled: set[int] = set()
        async for event in handle.fetch_history_events():
            if event.HasField("activity_task_scheduled_event_attributes"):
                sched = event.activity_task_scheduled_event_attributes
                if sched.activity_type.name == "run_claude_segment":
                    scheduled.add(event.event_id)
                    steps_in.append(json.loads(sched.input.payloads[0].data))
            if event.HasField("activity_task_completed_event_attributes"):
                done = event.activity_task_completed_event_attributes
                if done.scheduled_event_id in scheduled:
                    steps_out.append(json.loads(done.result.payloads[0].data))
    assert len(held) == 4  # one entry per step
    assert [s["conversation"]["entries"] for s in steps_in] == [0, 1, 2, 3]
    assert all(s["transcript"] is None for s in steps_in)  # read, not copied in
    assert [(s["transcript_keep"], len(s["transcript_add"])) for s in steps_out] == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    assert held == [e for s in steps_out for e in s["transcript_add"]]


@pytest.mark.usefixtures("shop_dir")
async def test_a_reset_takes_the_conversation_back_with_the_workflow(
    client: Client,
) -> None:
    """Reset to the Workflow task after the first count: the new run holds the
    conversation as it was then, and Claude decides again from there."""
    queue = f"reset-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 3", TaskOptions(continue_as_new=False), None],
            id=queue,
            task_queue=queue,
        )
        assert await asyncio.wait_for(handle.result(), 60) == "counted to 3"
        before = await handle.query(QUERY, args=["0", 0, 4, 1 << 19], result_type=PAGE)
        kinds: dict[int, str] = {}
        counted_at = reset_to = None
        async for event in handle.fetch_history_events():
            if event.HasField("activity_task_scheduled_event_attributes"):
                attrs = event.activity_task_scheduled_event_attributes
                kinds[event.event_id] = attrs.activity_type.name
            done = event.activity_task_completed_event_attributes
            if (
                event.HasField("activity_task_completed_event_attributes")
                and counted_at is None
                and kinds.get(done.scheduled_event_id) == "count"
            ):
                counted_at = event.event_id
            if (
                event.HasField("workflow_task_completed_event_attributes")
                and counted_at is not None
                and reset_to is None
            ):
                reset_to = event.event_id
        assert reset_to is not None
        reset = await client.workflow_service.reset_workflow_execution(
            ResetWorkflowExecutionRequest(
                namespace=client.namespace,
                workflow_execution=WorkflowExecution(
                    workflow_id=queue, run_id=handle.result_run_id or ""
                ),
                reason="test",
                workflow_task_finish_event_id=reset_to,
                request_id=str(uuid.uuid4()),
            )
        )
        again = client.get_workflow_handle(queue, run_id=reset.run_id)
        assert await asyncio.wait_for(again.result(), 60) == "counted to 3"
        after = await again.query(QUERY, args=["0", 0, 4, 1 << 19], result_type=PAGE)
    assert [e["detail"] for e in shop.executions("count")] == ["1", "2", "3", "2", "3"]
    assert after[0] == before[0]  # committed before the reset point
    assert {e["uuid"] for e in after[1:]}.isdisjoint(e["uuid"] for e in before[1:])


@pytest.mark.usefixtures("shop_dir")
async def test_a_long_conversation_moves_on_within_a_workflow_task(
    client: Client, tmp_path: Path
) -> None:
    """30,000 entries of many small objects (7 MB) through Continue-As-New, with
    External Storage: no Workflow task fails. The Workflow keeps each entry as JSON
    text; kept as objects, Temporal's converter copied them one by one in the
    Workflow task (3 seconds here), and Python's deadlock detection failed it."""
    stored = with_external_storage(client, tmp_path / "blobs")
    blocks = [{"type": "text", "n": [1, 2, 3]} for _ in range(8)]
    entries = [
        {"uuid": f"cp_{i:012x}", "turn": i + 1, "blocks": blocks} for i in range(30_000)
    ]
    state = AgentState(
        session_id="s",
        checkpoint=f"cp_{len(entries) - 1:012x}",
        task_prompt="count to 2",
        conversation=[_conversation.entry_text(e) for e in entries],
        external_storage=True,
    )
    queue = f"long-{uuid.uuid4().hex[:8]}"
    async with worker(stored, queue, ScriptedClaude(count_policy)):
        handle = await stored.start_workflow(
            LongTaskWorkflow.run,
            args=["count to 2", TaskOptions(after_events=10), state],
            id=queue,
            task_queue=queue,
        )
        result = await asyncio.wait_for(handle.result(), 180)
        runs, failed = 0, []
        async for execution in stored.list_workflows(f'WorkflowId = "{queue}"'):
            runs += 1
            run = stored.get_workflow_handle(queue, run_id=execution.run_id)
            async for event in run.fetch_history_events():
                if event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED:
                    failure = event.workflow_task_failed_event_attributes.failure
                    failed.append(failure.message)
    assert result == "counted to 2"
    assert runs >= 2 and failed == []


@pytest.mark.usefixtures("shop_dir")
async def test_a_conversation_larger_than_one_query_result_reaches_every_step(
    client: Client,
) -> None:
    """2.7 MB of conversation: more than one Query result may carry, so it comes in pages."""
    queue = f"pages-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(pages_policy)):
        answer = await client.execute_workflow(
            FetchWorkflow.run, "fetch 900 KB 3 times", id=queue, task_queue=queue
        )
    assert answer == f"saw {3 * 900 * 1024} characters in 3 documents"


@pytest.mark.usefixtures("shop_dir")
async def test_with_external_storage_pages_stay_in_the_query_response(
    client: Client, tmp_path: Path
) -> None:
    """Ten 60 KB results: about 600 KB of conversation, read at every step in pages
    under the threshold, so none of it goes to the store again (and no other payload
    reaches the 256 KiB threshold)."""
    blobs = tmp_path / "blobs"
    stored = with_external_storage(client, blobs)
    queue = f"inline-{uuid.uuid4().hex[:8]}"
    async with worker(stored, queue, ScriptedClaude(pages_policy)):
        answer = await stored.execute_workflow(
            FetchWorkflow.run, "fetch 60 KB 10 times", id=queue, task_queue=queue
        )
    assert answer == f"saw {10 * 60 * 1024} characters in 10 documents"
    assert list(blobs.iterdir()) == []


@pytest.mark.usefixtures("shop_dir")
async def test_with_external_storage_a_large_entry_is_stored_once(
    client: Client, tmp_path: Path
) -> None:
    """Four 300 KB results: each is a page of its own, over the threshold. Every
    later step reads the same payload, so the store keeps one copy of each."""
    blobs = tmp_path / "blobs"
    stored = with_external_storage(client, blobs)
    queue = f"large-{uuid.uuid4().hex[:8]}"
    async with worker(stored, queue, ScriptedClaude(pages_policy)):
        answer = await stored.execute_workflow(
            FetchWorkflow.run, "fetch 300 KB 4 times", id=queue, task_queue=queue
        )
    assert answer == f"saw {4 * 300 * 1024} characters in 4 documents"
    pages: list[list[Any]] = []
    for blob in blobs.iterdir():
        payload = Payload()
        payload.ParseFromString(blob.read_bytes())
        value = json.loads(payload.data)
        if isinstance(value, list):
            pages.append(value)
    # Three results were read again (by 3, 2 and 1 later steps): one page each.
    assert sorted(len(p) for p in pages) == [1, 1, 1]


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("external", [False, True], ids=["inline", "external"])
async def test_results_too_large_to_deliver_together_become_notes(
    client: Client, tmp_path: Path, external: bool
) -> None:
    """Two 1.1 MB results of one message: each fits a payload, both together do not.
    The Workflow task that schedules the next step used to fail again and again.
    Without External Storage, the larger result becomes a note for Claude; with it,
    both arrive."""
    if external:
        client = with_external_storage(client, tmp_path / "blobs")
    queue = f"together-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(parallel_fetch_policy)):
        answer = await asyncio.wait_for(
            client.execute_workflow(
                ParallelFetchWorkflow.run,
                "fetch 1100 KB 2 at once",
                id=queue,
                task_queue=queue,
            ),
            120,
        )
    assert answer == (
        "got 2 documents and 0 notes" if external else "got 1 documents and 1 notes"
    )
    assert len(shop.executions("fetch_document")) == 2  # both ran, once


@pytest.mark.usefixtures("shop_dir")
async def test_workers_that_keep_the_conversation_elsewhere_hold_the_step(
    client: Client, tmp_path: Path
) -> None:
    """A Worker set up for a session store cannot continue it: the step waits, then
    continues once a Worker set up like the first one picks it up."""
    queue = f"mixed-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(count_policy)):
        handle = await client.start_workflow(
            PatientWorkflow.run, "count to 2 and publish", id=queue, task_queue=queue
        )
        for _ in range(300):
            pending = await handle.query(PatientWorkflow.pending_approvals)
            if pending:
                break
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("no approval request")
    async with worker(client, queue, ScriptedClaude(count_policy, tmp_path / "folder")):
        await handle.execute_update(
            PatientWorkflow.review, args=[pending[0]["id"], True]
        )
        message = ""
        for _ in range(300):
            description = await handle.describe()
            for activity in description.raw_description.pending_activities:
                if activity.attempt > 1:
                    message = activity.last_failure.message
            if message:
                break
            await asyncio.sleep(0.1)
    assert "kept in its Workflow" in message
    async with worker(client, queue, ScriptedClaude(count_policy)):
        result = await asyncio.wait_for(handle.result(), 60)
    assert result == "counted to 2"
    assert [e["detail"] for e in shop.executions("count")] == ["1", "2"]
    assert len(shop.executions("publish")) == 1


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("external", [False, True], ids=["inline", "external"])
async def test_continue_as_new_hands_the_conversation_over_when_it_fits(
    client: Client, tmp_path: Path, external: bool
) -> None:
    """Two 1.9 MB results, Continue-As-New due after each tool call. Without External
    Storage, the first handover fits (one result waiting), the second does not (the
    conversation holds the first result too), so the agent keeps going in run 2; with
    it, the agent hands over both times."""
    if external:
        client = with_external_storage(client, tmp_path / "blobs")
    queue = f"carry-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(big_result_policy)):
        handle = await client.start_workflow(
            BigResultWorkflow.run,
            args=["fetch 1900 KB twice", None],
            id=queue,
            task_queue=queue,
        )
        result = await asyncio.wait_for(handle.result(), 120)
        runs = await handle.query(BigResultWorkflow.runs)
    assert result == "fetched 2 documents"
    assert runs == (3 if external else 2)
    assert len(shop.executions("fetch_document")) == 2


@pytest.mark.usefixtures("shop_dir")
@pytest.mark.parametrize("external", [False, True], ids=["inline", "external"])
async def test_an_explicit_continue_as_new_without_room_fails_clearly(
    client: Client, tmp_path: Path, external: bool
) -> None:
    if external:
        client = with_external_storage(client, tmp_path / "blobs")
    queue = f"handover-{uuid.uuid4().hex[:8]}"
    async with worker(client, queue, ScriptedClaude(big_result_policy)):
        result = await client.execute_workflow(
            HandOverWorkflow.run,
            args=["fetch 1900 KB twice", None],
            id=queue,
            task_queue=queue,
        )
    if external:
        assert result == "continued with 3 entries"
    else:
        assert result.startswith("fetched 2 documents; The agent's state is 3.")
        assert "External Storage" in result


# ---- the real engine ----


def _make(tmp_path: Path, api: Any, store: bool = False) -> ClaudeAgentSdkRunner:
    (tmp_path / "work").mkdir(exist_ok=True)
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store") if store else None,
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )


def _left_on_disk(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "cfg").glob("projects/*/*"))


async def test_real_engine_continues_on_workers_with_nothing_in_common(
    tmp_path: Path,
) -> None:
    """Each step on another "machine": its own working directory and config folder."""
    api = start_with_policy(refund_policy)
    tools = [
        ToolSpec(name, name, {"type": "object"})
        for name in ("look_up_order", "issue_refund", "email_customer")
    ]
    results = {
        "look_up_order": shop.ORDERS["A-1001"],
        "issue_refund": {"refund_id": "R-000001", "amount": 49.99},
        "email_customer": {"sent": True},
    }
    transcript: list[dict[str, Any]] = []
    inp = SegmentInput(
        session_id=str(uuid.uuid4()),
        prompt="Order A-1001 arrived broken, I want my money back.",
        tools=tools,
        transcript=[],
    )
    paused: list[str] = []
    answer: str | None = None
    try:
        for step in range(6):
            machine = tmp_path / f"machine-{step}"
            machine.mkdir()
            runner = ClaudeAgentSdkRunner(
                cwd=str(machine), env=engine_env(api, str(machine / "cfg"))
            )
            out = await runner.run(inp, 1)
            assert not out.is_error, out.error
            assert out.transcript_keep is not None
            transcript = transcript[: out.transcript_keep] + out.transcript_add
            if out.deferred is None:
                answer = out.result
                break
            paused.append(out.deferred.name)
            inp = SegmentInput(
                session_id=out.session_id,
                prompt=None,
                tools=tools,
                checkpoint=out.checkpoint,
                injected={out.deferred.id: ToolOutcome(results[out.deferred.name])},
                segment_index=step + 1,
                transcript=list(transcript),
            )
    finally:
        api.stop()
    assert paused == ["look_up_order", "issue_refund", "email_customer"]
    assert answer is not None and answer.startswith("Done. Refunded 49.99")
    assert api.errors == []


@pytest.mark.parametrize("store", [False, True], ids=["held", "store"])
async def test_real_engine_leaves_no_copy_of_a_held_conversation(
    tmp_path: Path, store: bool
) -> None:
    """The engine writes a new session to its config folder; held, that copy goes."""
    api = start_with_policy(refund_policy)
    runner = _make(tmp_path, api, store)
    sid = str(uuid.uuid4())
    try:
        out = await runner.run(
            SegmentInput(
                session_id=sid,
                prompt="Order A-1001 arrived broken, I want my money back.",
                tools=[
                    ToolSpec("look_up_order", "Look up an order.", {"type": "object"})
                ],
                transcript=None if store else [],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.deferred is not None and out.deferred.name == "look_up_order"
    assert _left_on_disk(tmp_path) == ([f"{sid}.jsonl"] if store else [])


async def test_real_engine_step_too_large_for_one_payload_is_not_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without External Storage, a step whose output cannot be recorded fails cleanly
    (instead of retrying forever); with it, the same step commits."""
    api = start_with_policy(count_policy)
    runner = _make(tmp_path, api)
    inp = SegmentInput(
        session_id=str(uuid.uuid4()),
        prompt="count to 1",
        tools=COUNT_TOOLS,
        transcript=[],
    )
    monkeypatch.setattr(_conversation, "PAYLOAD_LIMIT_BYTES", 2000)
    try:
        refused = await runner.run(inp, 1)
        monkeypatch.setattr(_runner, "external_storage_on", lambda: True)
        stored = await runner.run(inp, 2)
    finally:
        api.stop()
    assert refused.is_error and refused.error is not None
    assert "was not committed" in refused.error and "External Storage" in refused.error
    assert not stored.is_error and stored.deferred is not None
    assert stored.external_storage and stored.transcript_add


async def test_real_engine_refuses_a_conversation_kept_elsewhere(
    tmp_path: Path,
) -> None:
    """Workers must agree on where conversations live; the step says how to fix it."""
    api = start_with_policy(count_policy)
    held, stored = _make(tmp_path, api), _make(tmp_path, api, store=True)
    try:
        first = await held.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="count to 2",
                tools=COUNT_TOOLS,
                transcript=[],
            ),
            1,
        )
        assert first.deferred is not None and first.checkpoint is not None
        call = {first.deferred.id: ToolOutcome({"n": 1})}
        nxt = SegmentInput(
            session_id=first.session_id,
            prompt=None,
            tools=COUNT_TOOLS,
            checkpoint=first.checkpoint,
            injected=call,
            transcript=first.transcript_add,
        )
        with pytest.raises(
            RuntimeError, match="create this runner without session_store"
        ):
            await stored.run(nxt, 1)
        nxt.transcript = None
        with pytest.raises(
            RuntimeError, match="session store this runner does not have"
        ):
            await held.run(nxt, 1)
        nxt.transcript, nxt.checkpoint = first.transcript_add, str(uuid.uuid4())
        lost = await held.run(nxt, 1)
    finally:
        api.stop()
    assert lost.is_error and lost.error is not None
    assert "is not in the conversation" in lost.error


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("shop_dir")
async def test_real_engine_reads_a_held_conversation_through_external_storage(
    client: Client, tmp_path: Path
) -> None:
    """Two 2 MB results: each step's output and each page of the Query go through
    External Storage, so Claude reads the whole conversation."""
    api = start_with_policy(pages_policy)
    runner = _make(tmp_path, api)
    blobs = tmp_path / "blobs"
    stored = with_external_storage(client, blobs)
    queue = f"heldext-{uuid.uuid4().hex[:8]}"
    try:
        async with worker(stored, queue, runner):
            answer = await stored.execute_workflow(
                FetchWorkflow.run, "fetch 2000 KB 2 times", id=queue, task_queue=queue
            )
            history = await stored.get_workflow_handle(queue).fetch_history()
    finally:
        api.stop()
    assert answer == f"saw {2 * 2000 * 1024} characters in 2 documents"
    assert max(e.ByteSize() for e in history.events) < 64 * 1024  # references only
    assert sum(f.stat().st_size for f in blobs.iterdir()) > 3 * 2000 * 1024
    assert api.errors == [] and runner.stub_calls == 0
