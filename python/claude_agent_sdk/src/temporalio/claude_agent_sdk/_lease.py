"""Fence one Claude session so two attempts never run a CLI on it at once.

Claude Code does not lock a session transcript. A second ``--resume`` of a live
session interleaves both processes, and the next resume drops a branch. An
Activity that times out while its worker is still alive reaches that state.
"""

# Generations, never delete-then-create. Each holder claims the next generation number
# by creating that generation's record exclusively; a record is never overwritten or replaced.
# Two contenders for one generation race on one exclusive create and exactly one wins; the loser
# observes the winner. A holder renews by bumping a counter in its OWN generation's beat record,
# so a holder that was taken over can only ever write a record nobody reads any more: renewal
# cannot overwrite a newer holder, and a holder learns it lost by seeing a higher generation.
#
# No wall clock. A contender never compares timestamps written by another host. It
# reads the current record and watches it on its own monotonic clock: a record that does not
# change for a whole TTL belongs to a holder that stopped renewing. A live holder renews every
# third of the TTL and gives up (kills its CLI) when it cannot confirm a renewal for two thirds
# of it, so it is gone before any contender can judge it stale. Hosts must agree on the RATE of
# time (any real clock drifts far less than a third), never on its value: clock skew between
# workers is irrelevant.
#
# Temporal can supply that TTL of silence instead. A holder's Activity heartbeats the beat it
# last saw land, and the server times the Activity out only after a whole heartbeat timeout (the
# TTL) without receiving one. A retry that finds the record still at the beat of that last
# heartbeat knows the counter has not moved for at least that long, so it claims at once. Any
# renewal since moves the counter, and the record is watched as before.

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from temporalio.claude_agent_sdk._native import settle_answered_deferrals
from temporalio.claude_agent_sdk._sdk import load_sdk

logger = logging.getLogger(__name__)

LEASE_DIR_NAME = ".session-leases"


class SessionLeased(RuntimeError):
    """Another attempt held the session's lease, and kept renewing it, for the whole wait."""


class SessionLeaseLost(RuntimeError):
    """This attempt lost the session's lease while its CLI ran (taken over, or renewal could not
    be confirmed in time). The CLI was killed; what it did before that is unknown.
    """


@dataclass(frozen=True)
class LeaseRecord:
    """The newest generation of one session's lease, as one read of the store saw it."""

    generation: int
    owner: str
    beat: int
    """The holder's renewal counter; it changes on every renewal, so a reader can tell a live
    holder from a dead one by watching it, without trusting anyone's clock."""
    released: bool


class LeaseStore(Protocol):
    """Where session leases live. Three primitives; ``session_lease`` builds the lease on them.

    ``shared`` says every worker that can run a turn on these sessions sees the same store. A
    store that is not shared fences only the CLIs on its own host, so the Activity refuses to
    resume a lost attempt through one when a ``SessionStore`` lets other workers reach the
    session (see ``ClaudeTurnActivity``).
    """

    shared: bool

    def latest(self, session_id: str) -> LeaseRecord | None:
        """The session's newest generation with its latest beat, or None if never claimed."""
        ...

    def claim(self, session_id: str, generation: int, owner: str) -> bool:
        """Create ``generation``'s record for ``owner`` if no one has; True only for the one
        caller whose create succeeded. Must be atomic: this is the whole lock.
        """
        ...

    def beat(
        self,
        session_id: str,
        generation: int,
        owner: str,
        beat: int,
        *,
        released: bool = False,
    ) -> None:
        """Publish the holder's renewal counter (or its release) for its own generation."""
        ...


class FileLeaseStore:
    """Leases as files: ``<root>/<session_id>/<generation>`` plus a ``.beat`` beside it.

    A claim writes the whole record to a temp file and ``os.link``s it to the generation's
    name, which fails if the name exists, so no reader sees a partial record and no claim
    replaces another. On Windows it renames the temp file instead, which never replaces an
    existing name there either. A beat is written to a temp file and ``os.replace``d over the holder's own
    beat file.

    Multi-worker use needs ``root`` on a filesystem every worker mounts with POSIX semantics for
    ``link`` and ``rename`` (local disk, NFSv4+; not SMB or object-storage FUSE mounts). On NFS,
    mount with attribute caching off for this directory (``actimeo=0`` or ``noac``): a cached
    listing hides a newer generation from its loser, and a cached beat makes a live holder look
    dead, until the cache expires. A link whose reply was lost and retransmitted reports
    ``EEXIST`` for our own record, which ``claim`` checks for.

    Args:
        root: The directory holding every session's lease. Configure it explicitly, on shared
            storage, for more than one worker.
        shared: Whether ``root`` is the same directory for every worker. True for an explicitly
            configured root; the Activity's default per-host root passes False.
    """

    def __init__(self, root: str | Path, *, shared: bool = True) -> None:
        """Store leases under ``root``. See the class docstring for parameters."""
        self.root = Path(root)
        self.shared = shared

    def _dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def latest(self, session_id: str) -> LeaseRecord | None:
        """Return the newest generation and its beat."""
        path = self._dir(session_id)
        for _ in range(
            3
        ):  # a newer claim may prune the generation we listed; list again
            generations = [int(n) for n in os.listdir(path) if n.isdigit()]
            if not generations:
                return None
            generation = max(generations)
            try:
                record = json.loads((path / _name(generation)).read_text())
            except FileNotFoundError:
                continue
            beat = _read(path / f"{_name(generation)}.beat")
            if beat and (
                beat.get("owner") != record["owner"]
                or type(beat.get("beat")) is not int
                or not isinstance(beat.get("released"), bool)
            ):
                raise OSError(f"invalid lease heartbeat for {session_id}")
            return LeaseRecord(
                generation=generation,
                owner=record["owner"],
                beat=int(beat.get("beat", -1)),
                released=bool(beat.get("released")),
            )
        raise OSError(f"the lease of {session_id} kept changing while it was read")

    def claim(self, session_id: str, generation: int, owner: str) -> bool:
        """Create ``generation`` exclusively for ``owner``."""
        path = self._dir(session_id)
        target = path / _name(generation)
        tmp = path / f".{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps({"owner": owner, "generation": generation}))
        try:
            # Windows cannot delete the temp name of a linked record while a loser reads the
            # record (sharing is per file, not per name), so it renames instead.
            (os.rename if sys.platform == "win32" else os.link)(tmp, target)
        except FileExistsError:
            # Ours anyway if the link landed and only its reply was lost (NFS retransmission).
            try:
                existing_owner = _read(target).get("owner")
            except OSError:
                return False  # an unreadable existing claim cannot authorize ownership
            if existing_owner != owner:
                return False
        finally:
            tmp.unlink(missing_ok=True)
        names = os.listdir(path)
        # A slow contender can link a generation a newer claim already pruned; the newest
        # generation is never pruned, so seeing one above ours means we lost.
        if any(n.isdigit() and int(n) > generation for n in names):
            _discard(target)
            return False
        # Older generations are read by no one now; a stale holder still checking finds ours.
        for name in names:
            stem = name.split(".", 1)[0]
            if stem.isdigit() and int(stem) < generation:
                _discard(path / name)
        return True

    def beat(
        self,
        session_id: str,
        generation: int,
        owner: str,
        beat: int,
        *,
        released: bool = False,
    ) -> None:
        """Record a renewal on this holder's own generation."""
        path = self._dir(session_id)
        tmp = path / f".{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps({"owner": owner, "beat": beat, "released": released}))
        os.replace(tmp, path / f"{_name(generation)}.beat")


def _name(generation: int) -> str:
    return f"{generation:012d}"


def _discard(path: Path) -> None:
    # Windows refuses to delete a file another process has open. Only generations below the
    # newest are discarded; one left behind is never chosen by `latest`, and the next claim
    # prunes it.
    with suppress(PermissionError):
        path.unlink(missing_ok=True)


def _read(path: Path) -> dict[str, Any]:
    try:
        held = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise OSError(f"invalid lease record {path}") from exc
    if not isinstance(held, dict) or not held:
        raise OSError(f"invalid lease record {path}")
    return held


class Lease:
    """One held generation of a session's lease, handed to the block ``session_lease`` guards."""

    def __init__(
        self, store: LeaseStore, session_id: str, generation: int, owner: str
    ) -> None:
        """Remember the generation this attempt holds."""
        self.store = store
        self.session_id = session_id
        self.generation = generation
        self.owner = owner
        self.lost: str | None = None
        self.beat = -1
        """The holder's last renewal it saw land in the store; -1, as the store reads it, before
        the first."""

    async def check(self) -> None:
        """Raise ``SessionLeaseLost`` unless this is still the session's newest generation: the
        fence a writer calls right before a write a newer holder must not be raced by.
        """
        if self.lost is None:
            latest = await asyncio.to_thread(self.store.latest, self.session_id)
            if latest is None or latest.generation != self.generation:
                self.lost = f"generation {self.generation} was superseded"
        if self.lost is not None:
            raise SessionLeaseLost(f"session {self.session_id}: {self.lost}")


@asynccontextmanager
async def session_lease(
    store: LeaseStore,
    session_id: str | None,
    ttl: timedelta,
    *,
    wait: timedelta | None = None,
    on_lost: Callable[[], None] | None = None,
    silent: tuple[int, int] | None = None,
    on_beat: Callable[[], None] | None = None,
) -> AsyncIterator[Lease | None]:
    """Hold ``session_id``'s lease for the block, renewing it every third of ``ttl``.

    Yields None without a lease when there is no session id: a turn with no session id is never
    resumed, so no second attempt can reach its transcript.

    Acquiring waits while another holder is live: up to ``wait`` (default: a little over
    ``ttl``, long enough to see a dead holder go stale), then raises ``SessionLeased``. A
    released lease is taken at once, and so is a record still at ``silent``'s
    ``(generation, beat)``: one known to have been unchanged for ``ttl`` already.

    ``on_beat`` is called on every poll while acquiring and after every confirmed renewal.

    If the lease is lost while the block runs, ``on_lost`` is called at once (it must kill the
    CLI, synchronously), the block is cancelled, and ``SessionLeaseLost`` is raised from it.
    """
    if session_id is None:
        yield None
        return

    seconds = ttl.total_seconds()
    owner = uuid.uuid4().hex
    generation = await _acquire(
        store, session_id, owner, seconds, wait, silent, on_beat
    )
    lease = Lease(store, session_id, generation, owner)
    body = asyncio.current_task()
    assert body is not None

    def lose(reason: str) -> None:
        lease.lost = reason
        logger.error("session %s: lease lost (%s); killing the CLI", session_id, reason)
        try:
            if on_lost is not None:
                on_lost()
        finally:
            body.cancel()

    async def renew() -> None:
        beat = 0
        loop = asyncio.get_running_loop()
        confirmed = loop.time()
        delay = seconds / 3
        while True:
            deadline = confirmed + seconds * 2 / 3
            try:
                async with asyncio.timeout_at(deadline):
                    await asyncio.sleep(delay)
                    beat += 1
                    # A timed-out thread may still write, but only to its own generation.
                    await asyncio.to_thread(
                        store.beat, session_id, generation, owner, beat
                    )
                    latest = await asyncio.to_thread(store.latest, session_id)
            except TimeoutError:
                return lose(f"no renewal confirmed for {loop.time() - confirmed:.1f}s")
            except OSError:
                logger.warning(
                    "session %s: lease renewal failed", session_id, exc_info=True
                )
                # Leave room to retry a transient error inside the unchanged deadline.
                delay = seconds / 6
            else:
                if loop.time() >= deadline:
                    return lose(
                        f"no renewal confirmed for {loop.time() - confirmed:.1f}s"
                    )
                if latest is None or latest.generation != generation:
                    newer = latest.generation if latest is not None else "none"
                    return lose(f"generation {generation} was taken over (now {newer})")
                if latest.beat == beat:
                    confirmed = loop.time()
                    delay = seconds / 3
                    lease.beat = beat
                    if on_beat is not None:
                        on_beat()
            # A contender needs a whole TTL of silence before it claims; stop before that.
            if loop.time() - confirmed >= seconds * 2 / 3:
                return lose(f"no renewal confirmed for {loop.time() - confirmed:.1f}s")

    task = asyncio.create_task(renew())
    try:
        yield lease
    except BaseException as exc:
        if lease.lost is None:
            raise
        if isinstance(exc, asyncio.CancelledError):
            body.uncancel()
        raise SessionLeaseLost(f"session {session_id}: {lease.lost}") from exc
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if lease.lost is None:
            try:
                await asyncio.to_thread(
                    store.beat, session_id, generation, owner, -1, released=True
                )
            except OSError:
                logger.warning(
                    "session %s: lease release failed", session_id, exc_info=True
                )
    if lease.lost is not None:
        # The block ended on its own just as the lease went: its CLI still ran beside another.
        if body.cancelling():
            body.uncancel()
        raise SessionLeaseLost(f"session {session_id}: {lease.lost}")


async def _acquire(
    store: LeaseStore,
    session_id: str,
    owner: str,
    ttl: float,
    wait: timedelta | None,
    silent: tuple[int, int] | None = None,
    on_beat: Callable[[], None] | None = None,
) -> int:
    """Claim the session's next generation once it is free, released, or silent for ``ttl``."""
    poll = min(ttl / 6, 1.0)
    deadline = time.monotonic() + (
        wait.total_seconds() if wait is not None else ttl + 2 * poll
    )
    seen: LeaseRecord | None = None
    unchanged_since = time.monotonic()
    while True:
        try:
            record = await asyncio.to_thread(store.latest, session_id)
        except OSError as exc:
            now = time.monotonic()
            if now >= deadline:
                raise SessionLeased(
                    f"session {session_id}: could not read its lease before the wait deadline"
                ) from exc
            # Unreadable data is not evidence of silence, including a missing first beat.
            seen, unchanged_since = None, now
            if on_beat is not None:
                on_beat()
            await asyncio.sleep(min(poll, deadline - now))
            continue
        now = time.monotonic()
        if record != seen:
            seen, unchanged_since = record, now
        known_silent = record is not None and (record.generation, record.beat) == silent
        stale = now - unchanged_since >= ttl or known_silent
        if record is None or record.released or stale:
            generation = (record.generation if record is not None else 0) + 1
            if await asyncio.to_thread(store.claim, session_id, generation, owner):
                if stale and record is not None:
                    # A worker that died holding the session: a different operational event
                    # from a clean handover, and the only place it is still visible.
                    logger.warning(
                        "session %s: took over generation %d, whose holder stopped renewing "
                        "%s and did not release it",
                        session_id,
                        record.generation,
                        "before its last Activity heartbeat timed out"
                        if known_silent
                        else f"{now - unchanged_since:.0f}s ago",
                    )
                return generation
            continue  # another contender won this generation; watch it from scratch
        if now >= deadline:
            raise SessionLeased(
                f"session {session_id} is leased by another attempt (generation "
                f"{record.generation}) that kept renewing it; a second CLI on one session "
                "corrupts its transcript"
            )
        if on_beat is not None:
            on_beat()
        await asyncio.sleep(poll)


def fenced_session_store(store: Any, lease: Lease | None) -> Any:
    """``store`` with every ``append`` refused once ``lease`` is no longer the session's newest
    generation, so a CLI that was taken over cannot keep mirroring into the transcript, and every
    ``load`` passed through ``_native.settle_answered_deferrals`` (the stored copy is unchanged).
    Without a lease only the second applies.

    A check-then-append, not an atomic one: SessionStore (SDK 0.2.165) has no conditional write
    to carry the generation into, so an append that passed its check still lands if a takeover
    wins before the store commits it. Killing the superseded CLI within a renewal interval stops
    later batches; it does not withdraw this one.
    The SDK probes optional methods on the store's CLASS (``_store_implements``), so the wrapper
    class only defines the ones ``store`` really implements.
    """
    if store is None:
        return None
    base: Any = type(store)
    protocol = load_sdk().SessionStore
    methods: dict[str, Any] = {
        name: _delegate(name)
        for name in (
            "list_sessions",
            "list_session_summaries",
            "delete",
            "list_subkeys",
        )
        if getattr(base, name, None) not in (None, getattr(protocol, name, None))
    }
    return type("FencedSessionStore", (_Fenced,), methods)(store, lease)


class _Fenced:
    def __init__(self, inner: Any, lease: Lease | None) -> None:
        self._inner = inner
        self._lease = lease

    async def append(self, key: Any, entries: list[Any]) -> None:
        if self._lease is not None:
            await self._lease.check()
        await self._inner.append(key, entries)

    async def load(self, key: Any) -> Any:
        entries = await self._inner.load(key)
        return settle_answered_deferrals(entries) if entries else entries


def _delegate(name: str) -> Callable[..., Any]:
    async def method(self: _Fenced, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, name)(*args, **kwargs)

    method.__name__ = name
    return method
