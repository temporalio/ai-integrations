"""Persist and restore a Claude session's working directory.

Activity-side only. Workflow code records the returned revision and does not
import the I/O used here.
"""

# One snapshot is one gzipped tar, content-addressed by its SHA-256, plus a commit
# marker per revision. Layout under the store root (a directory every worker mounts, e.g. EFS):
#
#   <root>/<session_id>/blobs/<sha256>.tar.gz   immutable archive, named by its own hash
#   <root>/<session_id>/revisions/<n>.json      commit marker: {"revision", "sha256", "size", ...}
#   <root>/<session_id>/tmp/                    in-flight writes, never read
#
# A tar per revision rather than a per-file tree: the whole directory moves at once, which a
# network filesystem does best, and a revision is exactly two files, which keeps atomicity simple.
#
# Atomic and race-free on a shared filesystem. The archive is written under tmp/ and
# renamed to its hash name; the revision is claimed by `os.link`ing a fully written marker to
# revisions/<n>.json, which fails if another writer took <n> first (link is atomic, NFS included),
# so the writer retries with <n+1>. A crash before the link leaves only tmp/ garbage: the
# previous revision stays the latest and intact.
#
# Restore trusts nothing it reads. The archive is copied off the store while hashed and
# checked against the marker and its own name, member counts and sizes are bounded, and
# extraction uses tarfile's "data" filter (no absolute paths, no `..`, no links or writes that
# resolve outside the target, no devices, no setuid bits, no chown). It extracts into a fresh
# sibling directory and swaps it in, so a refused or corrupt snapshot leaves the target untouched.

from __future__ import annotations

import asyncio
import fnmatch
import gzip
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Protocol

logger = logging.getLogger(__name__)

DEFAULT_IGNORE: tuple[str, ...] = ()
"""``fnmatch`` patterns skipped by default. The native-tool state directory sits beside the
session directory, so it is not inside a snapshot."""

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
"""Bound on a workspace's total regular-file bytes, on persist and on restore."""

DEFAULT_MAX_ENTRIES = 100_000
"""Bound on a workspace's files, directories and symlinks, on persist and on restore."""

_CHUNK = 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION_FILE = re.compile(r"([1-9][0-9]*)\.json")


class WorkspaceSnapshotError(Exception):
    """A workspace could not be persisted or restored."""


class WorkspaceTooLarge(WorkspaceSnapshotError):
    """The workspace (or a stored snapshot) exceeds the store's size or entry bound."""


class WorkspaceIntegrityError(WorkspaceSnapshotError):
    """A stored snapshot does not match its recorded hash, or holds an entry restore refuses."""


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """One committed, immutable revision of a session's workspace."""

    session_id: str
    revision: int
    sha256: str
    size: int
    created_at: float


class WorkspaceStore(Protocol):
    """Where session workspaces are snapshotted. Revisions are per session, start at 1, and only
    grow; a revision's content never changes once it is listed.
    """

    async def persist(self, session_id: str, source_dir: Path) -> WorkspaceSnapshot:
        """Snapshot ``source_dir`` as a new revision (or return the latest one if its content is
        identical). Raises :class:`WorkspaceTooLarge` before writing anything if over bounds.
        """
        ...

    async def restore(
        self, session_id: str, revision: int, target_dir: Path
    ) -> WorkspaceSnapshot:
        """Make ``target_dir`` exactly revision ``revision``'s content. On any error the target
        is left as it was.
        """
        ...


class SharedDirWorkspaceStore:
    """A :class:`WorkspaceStore` on a directory every worker mounts at any path (NFS, EFS, or a
    local disk for one worker). See the module comment for the layout and its guarantees.

    Args:
        root: The store's directory; created on first write.
        ignore: ``fnmatch`` patterns, matched against each entry's name and its path relative
            to the workspace (``/``-separated); a matching directory is skipped whole.
        max_bytes: Bound on the workspace's total regular-file bytes.
        max_entries: Bound on the workspace's files, directories and symlinks.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ignore: Sequence[str] = DEFAULT_IGNORE,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Store snapshots under ``root``. See the class docstring for parameters."""
        self._root = Path(root)
        self._ignore = tuple(ignore)
        self._max_bytes = max_bytes
        self._max_entries = max_entries

    async def persist(self, session_id: str, source_dir: Path) -> WorkspaceSnapshot:
        """Snapshot ``source_dir``. See :meth:`WorkspaceStore.persist`."""
        return await asyncio.to_thread(self._persist, session_id, Path(source_dir))

    async def restore(
        self, session_id: str, revision: int, target_dir: Path
    ) -> WorkspaceSnapshot:
        """Replace ``target_dir`` with one revision. See :meth:`WorkspaceStore.restore`."""
        return await asyncio.to_thread(
            self._restore, session_id, revision, Path(target_dir)
        )

    async def get(self, session_id: str, revision: int) -> WorkspaceSnapshot:
        """Return one committed revision. See :meth:`WorkspaceStore.get`."""
        return await asyncio.to_thread(self._get, session_id, revision)

    async def latest(self, session_id: str) -> WorkspaceSnapshot | None:
        """Return the highest committed revision. See :meth:`WorkspaceStore.latest`."""
        return await asyncio.to_thread(self._latest, session_id)

    async def revisions(self, session_id: str) -> list[int]:
        """Return every committed revision, ascending."""
        return await asyncio.to_thread(self._revisions, session_id)

    async def delete(self, session_id: str) -> None:
        """Remove every snapshot of the session."""
        await asyncio.to_thread(self._delete, session_id)

    def _session(self, session_id: str) -> Path:
        if not _SAFE_ID.fullmatch(session_id):
            raise ValueError(
                f"invalid session id for a workspace store: {session_id!r}"
            )
        return self._root / session_id

    def _persist(self, session_id: str, source: Path) -> WorkspaceSnapshot:
        session = self._session(session_id)
        entries = scan_workspace(
            source, self._ignore, self._max_bytes, self._max_entries
        )
        tmp_dir = session / "tmp"
        for sub in (tmp_dir, session / "blobs", session / "revisions"):
            sub.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=tmp_dir, suffix=".tar.gz")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as raw:
                pack_workspace(source, entries, raw)
            digest, size = _hash_file(tmp)
            latest = self._latest(session_id)
            if latest is not None and latest.sha256 == digest:
                return latest
            os.replace(tmp, session / "blobs" / f"{digest}.tar.gz")
            return self._commit(session_id, digest, size)
        finally:
            tmp.unlink(missing_ok=True)

    def _commit(self, session_id: str, digest: str, size: int) -> WorkspaceSnapshot:
        session = self._session(session_id)
        revision = max(self._revisions(session_id), default=0) + 1
        while True:
            snapshot = WorkspaceSnapshot(
                session_id, revision, digest, size, time.time()
            )
            fd, marker = tempfile.mkstemp(dir=session / "tmp", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(asdict(snapshot), f)
                    f.flush()
                    os.fsync(f.fileno())
                os.link(marker, session / "revisions" / f"{revision}.json")
                return snapshot
            except FileExistsError:
                revision = max(self._revisions(session_id), default=revision) + 1
            finally:
                Path(marker).unlink(missing_ok=True)

    def _restore(
        self, session_id: str, revision: int, target: Path
    ) -> WorkspaceSnapshot:
        snapshot = self._get(session_id, revision)
        blob = self._session(session_id) / "blobs" / f"{snapshot.sha256}.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        # A private copy, so the bytes extracted are the bytes hashed even if the store changes.
        with tempfile.TemporaryFile() as local:
            with open(blob, "rb") as src:
                digest = _copy_hashing(src, local)
            if digest != snapshot.sha256:
                raise WorkspaceIntegrityError(
                    f"workspace snapshot {session_id} r{revision} is corrupt: sha256 {digest} "
                    f"!= recorded {snapshot.sha256}"
                )
            local.seek(0)
            staging = Path(
                tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.restore-")
            )
            try:
                unpack_workspace(local, staging, self._max_bytes, self._max_entries)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        _swap_in(staging, target)
        return snapshot

    def _get(self, session_id: str, revision: int) -> WorkspaceSnapshot:
        path = self._session(session_id) / "revisions" / f"{revision}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise KeyError(
                f"no workspace revision {revision} for session {session_id}"
            ) from None
        snapshot = WorkspaceSnapshot(**data)
        if snapshot.revision != revision or not re.fullmatch(
            r"[0-9a-f]{64}", snapshot.sha256
        ):
            raise WorkspaceIntegrityError(
                f"bad commit marker for {session_id} r{revision}"
            )
        return snapshot

    def _latest(self, session_id: str) -> WorkspaceSnapshot | None:
        revisions = self._revisions(session_id)
        return self._get(session_id, revisions[-1]) if revisions else None

    def _revisions(self, session_id: str) -> list[int]:
        try:
            names = os.listdir(self._session(session_id) / "revisions")
        except FileNotFoundError:
            return []
        return sorted(int(m[1]) for n in names if (m := _REVISION_FILE.fullmatch(n)))

    def _delete(self, session_id: str) -> None:
        session = self._session(session_id)
        trash = self._root / f".deleted-{session_id}-{uuid.uuid4().hex}"
        try:
            os.rename(session, trash)
        except FileNotFoundError:
            return
        shutil.rmtree(trash)


def scan_workspace(
    source: Path, ignore: Sequence[str], max_bytes: int, max_entries: int
) -> list[str]:
    """The workspace's entries to snapshot, as sorted relative ``/`` paths, after checking the
    bounds. Symlinks are kept as links, never followed; one that points outside the workspace
    (absolute, or up past its root) is left out with a warning, since restore would refuse it.
    """
    if not source.is_dir():
        raise WorkspaceSnapshotError(f"workspace {source} is not a directory")
    entries: list[str] = []
    total = 0
    stack = [""]
    while stack:
        rel = stack.pop()
        with os.scandir(source / rel) as it:
            children = sorted(it, key=lambda e: e.name)
        for child in children:
            path = posixpath.join(rel, child.name) if rel else child.name
            if any(
                fnmatch.fnmatchcase(child.name, p) or fnmatch.fnmatchcase(path, p)
                for p in ignore
            ):
                continue
            if child.is_symlink():
                link = os.readlink(child.path)
                resolved = posixpath.normpath(posixpath.join(rel, link))
                if (
                    posixpath.isabs(link)
                    or resolved == ".."
                    or resolved.startswith("../")
                ):
                    logger.warning(
                        "Not snapshotting %s: its symlink leaves the workspace", path
                    )
                    continue
            elif child.is_dir():
                stack.append(path)
            elif child.is_file():
                total += child.stat(follow_symlinks=False).st_size
            else:
                logger.warning(
                    "Not snapshotting %s: not a file, directory or symlink", path
                )
                continue
            entries.append(path)
            if total > max_bytes or len(entries) > max_entries:
                raise WorkspaceTooLarge(
                    f"workspace {source} exceeds the snapshot bound of {max_bytes} bytes / "
                    f"{max_entries} entries"
                )
    return sorted(entries)


def pack_workspace(source: Path, entries: Sequence[str], out: IO[bytes]) -> None:
    """Write ``entries`` of ``source`` to the binary file ``out`` as a gzipped tar. The bytes
    depend only on content, modes and mtimes, so an unchanged workspace hashes the same.
    """

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with (
        gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        tar.add(source, arcname=".", recursive=False, filter=normalize)
        for path in entries:
            tar.add(source / path, arcname=path, recursive=False, filter=normalize)


def unpack_workspace(
    archive: IO[bytes], target: Path, max_bytes: int, max_entries: int
) -> None:
    """Extract the gzipped tar ``archive`` (a binary file) into the empty directory ``target``,
    refusing anything that would land outside it.
    """
    try:
        with tarfile.open(fileobj=archive, mode="r:gz") as tar:
            members = tar.getmembers()
            if (
                len(members) > max_entries + 1
                or sum(m.size for m in members) > max_bytes
            ):
                raise WorkspaceTooLarge(
                    f"workspace snapshot exceeds the bound of {max_bytes} bytes / "
                    f"{max_entries} entries"
                )
            # The data filter would extract "/x" as "x"; a snapshot we wrote never has one.
            if bad := [
                m.name
                for m in members
                if posixpath.isabs(m.name) or ".." in m.name.split("/")
            ]:
                raise WorkspaceIntegrityError(
                    f"refused workspace snapshot: unsafe paths {bad}"
                )
            tar.extractall(target, members=members, filter="data")
            # extractall leaves the destination's own mode alone; apply the snapshot root's.
            if root := next((m for m in members if m.name in (".", "./")), None):
                target.chmod(root.mode & 0o755)
    except (tarfile.FilterError, tarfile.TarError, EOFError, gzip.BadGzipFile) as exc:
        raise WorkspaceIntegrityError(f"refused workspace snapshot: {exc}") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _copy_hashing(src: IO[bytes], dst: IO[bytes]) -> str:
    h = hashlib.sha256()
    while chunk := src.read(_CHUNK):
        h.update(chunk)
        dst.write(chunk)
    return h.hexdigest()


def _swap_in(staging: Path, target: Path) -> None:
    """Replace ``target`` with ``staging`` (same parent). Not one atomic step: a crash between
    the two renames leaves the old copy beside an absent target, which the next restore
    simply recreates.
    """
    old: Path | None = target.with_name(f".{target.name}.old-{uuid.uuid4().hex}")
    try:
        os.rename(target, old)  # type: ignore
    except FileNotFoundError:
        old = None
    os.rename(staging, target)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
