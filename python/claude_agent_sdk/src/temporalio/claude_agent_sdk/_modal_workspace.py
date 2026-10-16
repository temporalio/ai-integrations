"""A :class:`WorkspaceStore` on Modal, for workers that share no filesystem.

Activity-side only. ``modal`` (the ``modal`` extra) is imported only when a store is built with
:meth:`ModalVolumeWorkspaceStore.from_name` or a Modal error is checked, never on import.
"""

# The same snapshot format and layout as SharedDirWorkspaceStore, inside one Volume:
#
#   /<session_id>/blobs/<sha256>.tar.gz   immutable archive, named by its own hash
#   /<session_id>/revisions/<n>.json      commit marker: {"revision", "sha256", "size", ...}
#
# The store uses the Volume's client API (batch_upload, read_file, listdir, remove_file), never a
# mount, so commit()/reload() do not apply: an upload is visible once its batch closes, and a
# batch commits all its files in one call, so a file is either absent or whole.
#
# A revision is claimed in a modal.Dict, not by the Volume. Modal documents Volumes as
# last-write-wins with no locking (https://modal.com/docs/guide/volumes#concurrent-modification),
# while Dict.put(key, value, skip_if_exists=True) is a documented conditional write that reports
# whether it created the key: it plays the part os.link plays on a shared filesystem. The claim's
# value is a fresh token, so a put whose response was lost and retried (reporting False) is told
# apart from someone else's claim by reading the token back. Only the claim's owner uploads the
# marker. A writer that dies between claim and marker leaves a number that is skipped: revisions
# only grow, but may skip. Dict entries expire after 7 days idle
# (https://modal.com/docs/guide/dicts); by then the marker is the record, and new claims always
# start above the highest marker.
#
# Restore trusts nothing it reads, as in the shared-dir store: the blob is copied to a private
# temp file while hashed (refused once it outgrows the marker's size), checked against the
# marker, and unpacked into a sibling directory that is swapped in.

from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import json
import posixpath
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from temporalio.claude_agent_sdk._workspace import (
    _REVISION_FILE,
    _SAFE_ID,
    DEFAULT_IGNORE,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTRIES,
    WorkspaceIntegrityError,
    WorkspaceSnapshot,
    _hash_file,
    _swap_in,
    pack_workspace,
    scan_workspace,
    unpack_workspace,
)

_MARKER_MAX_BYTES = 4096

MODAL_INSTALL_HINT = (
    "ModalVolumeWorkspaceStore needs the `modal` package: "
    "pip install 'temporalio-claude-agent-sdk[modal]'"
)


def _modal() -> Any:
    try:
        return importlib.import_module("modal")
    except ImportError as exc:
        raise ImportError(MODAL_INSTALL_HINT) from exc


def _is_not_found(exc: BaseException) -> bool:
    """Modal raises FileNotFoundError from read_file and remove_file, and its own
    NotFoundError from a listing of a missing path.
    """
    return isinstance(exc, (FileNotFoundError, _modal().exception.NotFoundError))


class ModalVolumeWorkspaceStore:
    """A :class:`WorkspaceStore` in a Modal Volume, with revisions claimed in a Modal Dict, for
    workers that share no filesystem. Build it with :meth:`from_name`.

    Args:
        volume: A ``modal.Volume``.
        claims: A ``modal.Dict`` holding the revision claims; several stores may share one.
        ignore: ``fnmatch`` patterns, matched against each entry's name and its path relative
            to the workspace (``/``-separated); a matching directory is skipped whole.
        max_bytes: Bound on the workspace's total regular-file bytes.
        max_entries: Bound on the workspace's files, directories and symlinks.
    """

    def __init__(
        self,
        volume: Any,
        claims: Any,
        *,
        ignore: Sequence[str] = DEFAULT_IGNORE,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Bind the Volume and the claims Dict. See the class docstring for parameters."""
        self._volume = volume
        self._claims = claims
        self._ignore = tuple(ignore)
        self._max_bytes = max_bytes
        self._max_entries = max_entries

    @classmethod
    def from_name(
        cls,
        volume_name: str,
        *,
        claims_name: str | None = None,
        environment_name: str | None = None,
        create_if_missing: bool = True,
        **kwargs: Any,
    ) -> ModalVolumeWorkspaceStore:
        """The store in the named Volume, claiming revisions in the Dict ``claims_name`` (default
        ``<volume_name>-claims``). Nothing is contacted until first use; credentials come from
        ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``.
        """
        modal = _modal()
        volume = modal.Volume.from_name(
            volume_name,
            environment_name=environment_name,
            create_if_missing=create_if_missing,
        )
        claims = modal.Dict.from_name(
            claims_name or f"{volume_name}-claims",
            environment_name=environment_name,
            create_if_missing=create_if_missing,
        )
        return cls(volume, claims, **kwargs)

    async def persist(self, session_id: str, source_dir: Path) -> WorkspaceSnapshot:
        """Snapshot ``source_dir`` as a new revision, or return the latest if identical."""
        session = self._session(session_id)
        source = Path(source_dir)
        entries = await asyncio.to_thread(
            scan_workspace, source, self._ignore, self._max_bytes, self._max_entries
        )
        with tempfile.TemporaryDirectory(prefix="claude-workspace-") as tmp_dir:
            archive = Path(tmp_dir) / "snapshot.tar.gz"
            digest, size = await asyncio.to_thread(_pack_to, source, entries, archive)
            latest = await self.latest(session_id)
            if latest is not None and latest.sha256 == digest:
                return latest
            try:
                async with self._volume.batch_upload.aio(force=False) as batch:
                    batch.put_file(str(archive), f"{session}/blobs/{digest}.tar.gz")
            except FileExistsError:
                pass  # content-addressed: the same bytes, from an earlier or concurrent persist
        return await self._commit(session_id, digest, size)

    async def _commit(
        self, session_id: str, digest: str, size: int
    ) -> WorkspaceSnapshot:
        session = self._session(session_id)
        revision = max(await self.revisions(session_id), default=0) + 1
        while True:
            if not await self._claim(session_id, revision):
                revision = max([*await self.revisions(session_id), revision]) + 1
                continue
            snapshot = WorkspaceSnapshot(
                session_id, revision, digest, size, time.time()
            )
            marker = json.dumps(asdict(snapshot)).encode()
            path = f"{session}/revisions/{revision}.json"
            try:
                async with self._volume.batch_upload.aio(force=False) as batch:
                    batch.put_file(io.BytesIO(marker), path)
            except FileExistsError:
                # Our own upload, retried after a lost response, is done. Any other marker means
                # the claim was not exclusive (it expired): leave that marker and move on.
                if await self._read_small(path) != marker:
                    revision += 1
                    continue
            return snapshot

    async def _claim(self, session_id: str, revision: int) -> bool:
        key = f"{session_id}/{revision}"
        token = uuid.uuid4().hex
        if await self._claims.put.aio(key, token, skip_if_exists=True):
            return True
        return await self._claims.get.aio(key) == token

    async def restore(
        self, session_id: str, revision: int, target_dir: Path
    ) -> WorkspaceSnapshot:
        """Make ``target_dir`` exactly revision ``revision``; on error it is left as it was."""
        target = Path(target_dir)
        snapshot = await self.get(session_id, revision)
        blob = f"{self._session(session_id)}/blobs/{snapshot.sha256}.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        # A private copy, so the bytes extracted are the bytes hashed.
        with tempfile.TemporaryFile() as local:
            h = hashlib.sha256()
            received = 0
            try:
                async for chunk in self._volume.read_file.aio(blob):
                    received += len(chunk)
                    if received > snapshot.size:
                        raise WorkspaceIntegrityError(
                            f"workspace snapshot {session_id} r{revision} is corrupt: larger "
                            f"than its recorded {snapshot.size} bytes"
                        )
                    h.update(chunk)
                    local.write(chunk)
            except FileNotFoundError as exc:
                raise WorkspaceIntegrityError(
                    f"workspace snapshot {session_id} r{revision} is missing its archive"
                ) from exc
            if h.hexdigest() != snapshot.sha256:
                raise WorkspaceIntegrityError(
                    f"workspace snapshot {session_id} r{revision} is corrupt: sha256 "
                    f"{h.hexdigest()} != recorded {snapshot.sha256}"
                )
            local.seek(0)
            staging = Path(
                tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.restore-")
            )
            try:
                await asyncio.to_thread(
                    unpack_workspace, local, staging, self._max_bytes, self._max_entries
                )
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        await asyncio.to_thread(_swap_in, staging, target)
        return snapshot

    async def get(self, session_id: str, revision: int) -> WorkspaceSnapshot:
        """The commit marker of one revision; ``KeyError`` when there is none."""
        path = f"{self._session(session_id)}/revisions/{revision}.json"
        try:
            raw = await self._read_small(path)
        except FileNotFoundError:
            raise KeyError(
                f"no workspace revision {revision} for session {session_id}"
            ) from None
        try:
            snapshot = WorkspaceSnapshot(**json.loads(raw))
        except (ValueError, TypeError) as exc:
            raise WorkspaceIntegrityError(
                f"bad commit marker for {session_id} r{revision}"
            ) from exc
        if snapshot.revision != revision or not re.fullmatch(
            r"[0-9a-f]{64}", snapshot.sha256
        ):
            raise WorkspaceIntegrityError(
                f"bad commit marker for {session_id} r{revision}"
            )
        return snapshot

    async def latest(self, session_id: str) -> WorkspaceSnapshot | None:
        """The newest revision's marker, or None."""
        revisions = await self.revisions(session_id)
        return await self.get(session_id, revisions[-1]) if revisions else None

    async def revisions(self, session_id: str) -> list[int]:
        """The session's committed revisions, ascending."""
        try:
            entries = await self._volume.listdir.aio(
                f"{self._session(session_id)}/revisions"
            )
        except Exception as exc:
            if not _is_not_found(exc):
                raise
            return []
        names = (posixpath.basename(e.path) for e in entries)
        return sorted(int(m[1]) for n in names if (m := _REVISION_FILE.fullmatch(n)))

    async def delete(self, session_id: str) -> None:
        """Remove every revision of the session and its claims."""
        session = self._session(session_id)
        revisions = await self.revisions(session_id)
        try:
            await self._volume.remove_file.aio(session, recursive=True)
        except Exception as exc:
            if not _is_not_found(exc):
                raise
        # Claims up to the last marker, skipped numbers included, so the id starts over at 1.
        for revision in range(1, max(revisions, default=0) + 1):
            await self._claims.pop.aio(f"{session_id}/{revision}", None)

    def _session(self, session_id: str) -> str:
        if not _SAFE_ID.fullmatch(session_id):
            raise ValueError(
                f"invalid session id for a workspace store: {session_id!r}"
            )
        return f"/{session_id}"

    async def _read_small(self, path: str) -> bytes:
        data = b""
        async for chunk in self._volume.read_file.aio(path):
            data += chunk
            if len(data) > _MARKER_MAX_BYTES:
                raise WorkspaceIntegrityError(f"commit marker {path} is too large")
        return data


def _pack_to(source: Path, entries: list[str], archive: Path) -> tuple[str, int]:
    with open(archive, "wb") as out:
        pack_workspace(source, entries, out)
    return _hash_file(archive)
