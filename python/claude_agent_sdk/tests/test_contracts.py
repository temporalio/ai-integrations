"""Names and limits that running Workflows, other Workers and Claude depend on.

A Workflow's history records the Activity names, the Query name and the patch IDs, so
renaming one breaks every Workflow that is running or recorded: a replay fails, or a
step one version schedules finds no Activity on the next. The size limits keep each
payload under Temporal's defaults. Each is pinned here with its reason, as behavior
where it can be measured.
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Any

import pytest

from temporalio.claude_agent_sdk import AgentState, SegmentOutput
from temporalio.claude_agent_sdk import _conversation as conversation
from temporalio.claude_agent_sdk import _defer_hook as hook
from temporalio.claude_agent_sdk import _events as events
from temporalio.claude_agent_sdk import _runner as runner
from temporalio.claude_agent_sdk import _stand_in as stand_in
from temporalio.claude_agent_sdk import _workflow as wf
from temporalio.contrib.workflow_streams import WorkflowStreamState
from temporalio.contrib.workflow_streams._types import _WorkflowStreamWireItem
from temporalio.converter import DataConverter

CONVERTER = DataConverter.default.payload_converter
KIB = 1024
MIB = 1024 * KIB


def payload_bytes(value: Any) -> int:
    """Size of ``value`` as one payload of Temporal's default converter."""
    return CONVERTER.to_payloads([value])[0].ByteSize()


def data_bytes(value: Any) -> int:
    """Size of the JSON text in that payload."""
    return len(CONVERTER.to_payloads([value])[0].data)


# ---- names recorded in Workflow histories ----


def test_names_in_workflow_histories_never_change() -> None:
    """Running and recorded Workflows name these: a new version must keep them."""
    assert wf.SEGMENT_ACTIVITY_NAME == "run_claude_segment"
    assert wf.TOOL_STEP_ACTIVITY_NAME == "run_claude_tool_step"
    assert conversation.QUERY == "__claude_agent_conversation"
    assert wf._CHECKS_PATCH == "temporalio-claude-agent-sdk-continue-as-new-checks"
    assert wf._CARRY_PATCH == "temporalio-claude-agent-sdk-measured-stream-carry"
    assert events.TOPIC == "claude"  # carried across Continue-As-New in the stream


def test_durable_tools_have_one_name_on_both_sides_of_the_hook() -> None:
    """The runner names durable tools for the engine; the hook recognizes them."""
    assert runner.PREFIX == hook.DURABLE_PREFIX == "mcp__durable__"


# ---- payload and page limits ----


def test_a_full_page_stays_under_temporals_payload_warning() -> None:
    """Temporal logs a warning from 512 KiB; a page leaves 4 KiB for its encoding."""
    assert conversation.PAGE_ROOM == 4 * KIB
    assert conversation.PAGE_BYTES == 512 * KIB - conversation.PAGE_ROOM
    entry = {"uuid": "u", "text": "x" * 1000}
    entries = [entry] * 1000
    sizes = [conversation.entry_bytes(e) for e in entries]
    full = conversation.page(entries, sizes, 0)
    assert len(full) < len(entries)  # the limit cut it
    data = data_bytes(full)
    assert data <= conversation.PAGE_BYTES < data + sizes[0] + 1  # no room for one more
    assert payload_bytes(full) < 512 * KIB  # with the payload's own fields


def test_pages_hold_at_least_128_kib_under_a_low_storage_threshold() -> None:
    """So a low External Storage threshold does not mean many small Queries."""
    assert conversation.SMALLEST_PAGE == 128 * KIB


def test_one_payload_leaves_64_kib_under_temporals_2_mib_limit() -> None:
    assert conversation.PAYLOAD_LIMIT_BYTES == 2 * MIB - 64 * KIB


@pytest.mark.parametrize("over", [0, 1])
def test_a_step_output_is_too_large_one_byte_past_the_limit(over: int) -> None:
    out = SegmentOutput(session_id="s", checkpoint="c", transcript_keep=0)
    empty = conversation.json_bytes(dataclasses.asdict(out))
    text = "x" * (conversation.PAYLOAD_LIMIT_BYTES - empty - len('{"t":""}') + over)
    out.transcript_add = [{"t": text}]
    assert conversation.json_bytes(dataclasses.asdict(out)) == (
        conversation.PAYLOAD_LIMIT_BYTES + over
    )
    problem = conversation.too_large(out)
    if over:
        assert problem is not None
        assert problem.startswith("This step's output is 1.9 MB with what it added")
        assert "Configure External Storage" in problem
    else:
        assert problem is None
    out.external_storage = True  # the store takes large payloads
    assert conversation.too_large(out) is None


def test_json_bytes_is_the_size_temporals_converter_writes() -> None:
    value = {"a": [1, 2, {"b": "c, d: e"}], "text": "مرحبا 你好 😀"}
    assert conversation.json_bytes(value) == len(CONVERTER.to_payloads([value])[0].data)


# ---- history and Continue-As-New limits ----


def test_history_limits_are_temporals_defaults() -> None:
    """The server ends a run past them; the agent stops before, with room for a step."""
    assert wf._HISTORY_EVENTS == 51_200
    assert wf._HISTORY_BYTES == 50 * MIB
    assert wf._ROOM_EVENTS == 500
    assert wf._ROOM_BYTES == 3 * conversation.PAYLOAD_LIMIT_BYTES


def test_live_output_gets_a_share_of_the_new_runs_input() -> None:
    assert wf._HANDOVER_BYTES == 1536 * KIB < conversation.PAYLOAD_LIMIT_BYTES
    assert wf._RECENT_CALLS == 256


def _stream_with(count: int, data: str) -> WorkflowStreamState:
    return WorkflowStreamState(
        log=[_WorkflowStreamWireItem(topic=events.TOPIC, data=data)] * count,
        base_offset=7,
    )


@pytest.mark.parametrize("raw", [0, 1, 3, 600])
def test_one_carried_event_costs_what_the_budget_counts(raw: int) -> None:
    """Measured with the default converter: the event's text and 34 bytes of JSON."""
    data = base64.b64encode(b"x" * raw).decode("ascii")
    item = _WorkflowStreamWireItem(topic=events.TOPIC, data=data)
    one, two = (AgentState(stream=_stream_with(n, data)) for n in (1, 2))
    assert data_bytes(two) - data_bytes(one) == wf._event_bytes(item)
    assert wf._event_bytes(item) == len(data) + len(events.TOPIC) + 34


# ---- what Claude reads, and what the hook records ----


def test_denials_claude_may_retry_say_first_that_nothing_failed() -> None:
    """Since Claude Code 2.1.281 a denial reads "PreToolUse:<tool> hook error: ..."."""
    assert hook.NOT_RUN.startswith("Not an error. ")
    assert hook.MAIN_AGENT_ONLY.startswith("Not an error, ")
    assert "Call this tool again" in hook.NOT_RUN
    assert "calling it again will not help" in hook.MAIN_AGENT_ONLY
    # A hook that failed denies the call; Claude reads first that it did not run.
    assert hook.FAILED.startswith("This call did not run: ")


def test_the_hooks_records_name_each_reason() -> None:
    """The runner reads these keys, never the reasons' text."""
    assert hook.REASON_KEYS == {
        hook.NOT_RUN: "not_run",
        hook.STOPPED: "stopped",
        hook.MAIN_AGENT_ONLY: "main_agent_only",
        hook.STEP_ONLY: "step_only",
        hook.OTHER_INPUT: "other_input",
        hook.FAILED: "failed",
    }
    assert hook.WORKER_LOCK == "worker.lock"


# ---- tool steps (the README states these) ----


def test_tool_step_settings_the_readme_states() -> None:
    """Edit and Write go back as Claude Code's own record; a step's turn ends after
    its call; a continuing segment gets 2 minutes to start its turn, 3 times; the
    attempts of a step whose call did not run stop after about 5,000 events."""
    assert runner.RECORDED_TOOLS == ("Edit", "Write")
    assert runner.STEP_MAX_TURNS == 1
    assert runner.CONTINUE_START_SECONDS == 120.0
    assert runner.CONTINUE_ATTEMPTS == 3
    assert wf._ATTEMPT_EVENTS == 5_000  # type: ignore[reportPrivateUsage]
    # Claude Code's switch for SDK mode: the segment after a record needs it.
    assert runner.RESUME == "CLAUDE_CODE_RESUME_INTERRUPTED_TURN"
    assert runner.CONTINUE_ENV[runner.RESUME] == "1"
    assert runner.STEP_ENV[runner.RESUME] == ""


def test_engine_flags_that_approve_calls_are_refused_in_extra_args() -> None:
    """A tool step replaces permission_mode and the permission prompt tool; a flag in
    extra_args would come after its own and win."""
    assert runner._APPROVING_FLAGS == {  # type: ignore[reportPrivateUsage]
        "permission-mode",
        "dangerously-skip-permissions",
        "allow-dangerously-skip-permissions",
        "permission-prompt-tool",
    }


@pytest.mark.parametrize(
    ("tool_use_id", "safe"),
    [
        ("toolu_01AbC-9", True),
        ("x" * 128, True),
        ("x" * 129, False),
        ("", False),
        ("../x", False),
        ("a b", False),
        ("ä", False),
    ],
)
def test_a_denial_record_is_named_by_its_id_when_that_is_a_safe_file_name(
    tool_use_id: str, safe: bool
) -> None:
    name = hook.denial_name(tool_use_id)
    if safe:
        assert name == tool_use_id
    else:
        assert len(name) == 64 and all(c in "0123456789abcdef" for c in name)


# ---- other limits ----


def test_one_event_keeps_32_kib_of_each_text() -> None:
    assert events.FIELD_LIMIT == 32 * KIB


def test_the_stand_in_model_refuses_bodies_over_1_gib() -> None:
    assert stand_in.MAX_BODY_BYTES == 1024 * MIB
