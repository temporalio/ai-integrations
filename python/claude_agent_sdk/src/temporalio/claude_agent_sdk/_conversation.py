"""The conversation the Workflow holds: the Query that serves it, and the step side.

Without a session store, the Claude session's transcript lives in the Workflow, like
the conversation of the other AI integrations in this repository. A segment reads the
committed transcript with a Query on its own Workflow (so the transcript is not
copied into every step's input, and the history grows with what each step adds, not
with the whole conversation), seeds an in-memory session store with it, and returns
what it changed. Every attempt starts from the committed transcript, so nothing a
failed attempt wrote can reach Claude, and any Worker can run any step.

With External Storage, a Query result over its threshold goes to the store, at every
step that reads it. So pages stay under the threshold, and a page holds nothing but
its entries: an entry too large for a page of its own is the same payload at every
read, which a content-addressed store (such as Temporal's S3 driver) keeps once.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from temporalio import activity
from temporalio.client import WorkflowQueryFailedError
from temporalio.converter import ExternalStorage

from ._models import SegmentInput, SegmentOutput, ToolStepInput

QUERY = "__claude_agent_conversation"
"""The Workflow Query that returns the conversation, a page at a time.

Arguments: the agent's key, the first entry, how many entries the asking step was
scheduled with (the Query refuses a step that is out of date), and the most bytes of
entries in the page. It returns the entries.
"""

PAGE_ROOM = 4 * 1024
"""Bytes a page leaves under a size limit for its own encoding."""

PAGE_BYTES = 512 * 1024 - PAGE_ROOM
"""Most bytes of entries in one page: under 512 KiB, where Temporal starts to log a
warning about a payload's size (on the Worker and on the server)."""

SMALLEST_PAGE = 128 * 1024
"""Fewest bytes a page may hold, so a low threshold does not mean many small Queries."""

PAYLOAD_LIMIT_BYTES = 2 * 1024 * 1024 - 64 * 1024
"""Most bytes one payload may carry without External Storage.

Temporal refuses payloads over 2 MiB by default; 64 KiB stay free for the encoding
and for a payload codec.
"""


def json_bytes(value: Any) -> int:
    r"""Size of ``value`` as Temporal's default JSON converter writes it, in bytes.

    The converter escapes non-ASCII characters (``\uXXXX``), so text in, for
    example, Arabic or Chinese takes two to three times its UTF-8 size.
    """
    return len(json.dumps(value, separators=(",", ":"), default=str))


def entry_text(entry: Any) -> str:
    """One transcript entry as JSON text, the way the Workflow keeps it.

    Its length is the entry's size in a payload. The Workflow never reads inside an
    entry, and text moves through Continue-As-New much faster than nested objects:
    Temporal's converter copies every object of a dataclass field one by one, in the
    Workflow task, before it writes the JSON.
    """
    return json.dumps(entry, separators=(",", ":"), default=str)


def entry_bytes(entry: Any) -> int:
    """Size of one transcript entry in a payload, in bytes."""
    return len(entry_text(entry))


def page(
    entries: list[Any],
    sizes: list[int],
    start: int,
    limit: int = PAGE_BYTES,
) -> list[Any]:
    """The entries from ``start`` that fit in ``limit`` bytes (always at least one).

    Args:
        entries: The whole conversation (entries, or their JSON texts).
        sizes: The size of each entry (``entry_bytes``).
        start: The first entry to return.
        limit: Most bytes in the page, as JSON (the list's brackets and commas too).

    Returns:
        The entries of the page.
    """
    end = start = max(0, start)
    used = 1  # "[" and "]", less the comma the first entry does not need
    while end < len(entries) and (end == start or used + sizes[end] + 1 <= limit):
        used += sizes[end] + 1
        end += 1
    return entries[start:end]


def page_limit() -> int:
    """Most bytes of entries in a page this step asks for.

    With External Storage, just under its threshold, so pages stay in the Query's
    response instead of going to the store at every step.
    """
    storage = _external_storage()
    if storage is None:
        return PAGE_BYTES
    room = storage.payload_size_threshold - PAGE_ROOM
    return min(PAGE_BYTES, max(SMALLEST_PAGE, room))


async def read_conversation(
    inp: SegmentInput | ToolStepInput,
) -> list[dict[str, Any]]:
    """The committed conversation a step continues: inline, or read from the Workflow.

    Args:
        inp: The segment (or tool step) input.

    Returns:
        The transcript entries, oldest first (empty for a new conversation).

    Raises:
        RuntimeError: If the Workflow did not serve the conversation it announced
            (Temporal retries the step), or a conversation the Workflow holds is read
            outside the segment Activity.
    """
    if inp.transcript is not None:
        return list(inp.transcript)
    ref = inp.conversation
    if ref is None or ref.entries == 0:
        return []
    if not activity.in_activity():
        raise RuntimeError(
            "A conversation the Workflow holds can only be read inside the segment "
            "Activity; pass SegmentInput.transcript to run a segment directly."
        )
    info = activity.info()
    if info.workflow_id is None:  # a standalone Activity has no Workflow to ask
        raise RuntimeError(
            "This segment Activity was not started by a Workflow, so no Workflow "
            "holds its conversation; pass SegmentInput.transcript instead."
        )
    handle = activity.client().get_workflow_handle(
        info.workflow_id, run_id=info.workflow_run_id
    )
    limit = page_limit()
    entries: list[dict[str, Any]] = []
    while len(entries) < ref.entries:
        try:
            got: list[dict[str, Any]] = await handle.query(
                ref.query,
                args=[ref.agent, len(entries), ref.entries, limit],
                result_type=list[dict[str, Any]],
            )
        except WorkflowQueryFailedError as err:
            raise RuntimeError(
                f"The Workflow did not serve the conversation: {err}. Retrying."
            ) from err
        if not got:
            raise RuntimeError(
                f"The Workflow served no entries at {len(entries)} of {ref.entries}. "
                "Retrying."
            )
        entries.extend(got)
    return entries


def _external_storage() -> ExternalStorage | None:
    """This Worker's External Storage settings, or None without them."""
    if not activity.in_activity():
        return None
    try:
        return activity.client().data_converter.external_storage
    except RuntimeError:  # no client, for example in an ActivityEnvironment
        return None


def external_storage_on() -> bool:
    """Whether this Worker's data converter moves large payloads to External Storage."""
    return _external_storage() is not None


def too_large(out: SegmentOutput) -> str | None:
    """Explain why ``out`` cannot be recorded as one payload, or None if it can."""
    if out.external_storage:
        return None
    size = json_bytes(dataclasses.asdict(out))
    if size <= PAYLOAD_LIMIT_BYTES:
        return None
    return (
        f"This step's output is {size / 1024 / 1024:.1f} MB with what it added to the "
        "conversation, more than one Temporal payload holds without External Storage "
        "(2 MB by default), so the step was not committed. Configure External "
        "Storage on the Client (see the README), or give the runner a session store."
    )
