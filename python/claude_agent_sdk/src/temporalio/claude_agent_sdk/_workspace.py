"""Capture and restore a disposable workspace without following filesystem links."""

from __future__ import annotations

import base64
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


def snapshot(root: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Capture regular files and directories, including modes and modification times.

    Links and special files fail the snapshot. The caller must fence workspace
    writers while taking it; it is not a filesystem sandbox.

    Args:
        root: The managed workspace directory.
        max_bytes: Maximum total file contents to capture.

    Returns:
        A JSON-serializable tree, with paths relative to ``root``.

    Raises:
        RuntimeError: If the tree contains links, special files, or exceeds the limit.
    """
    files: dict[str, Any] = {}
    total = 0

    def visit(directory: Path) -> None:
        nonlocal total
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            name = path.relative_to(root).as_posix()
            entry: dict[str, Any] = {
                "mode": stat.S_IMODE(info.st_mode),
                "mtime_ns": info.st_mtime_ns,
            }
            if stat.S_ISDIR(info.st_mode):
                entry["kind"] = "directory"
                files[name] = entry
                visit(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                total += info.st_size
                if total > max_bytes:
                    raise RuntimeError("workspace snapshot exceeds its byte limit")
                # Do not follow a link swapped into place after lstat.
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as handle:
                    opened = os.fstat(handle.fileno())
                    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        raise RuntimeError("workspace changed while taking snapshot")
                    data = handle.read(max_bytes - (total - info.st_size) + 1)
                    after = os.fstat(handle.fileno())
                if len(data) != info.st_size or after.st_mtime_ns != info.st_mtime_ns:
                    raise RuntimeError("workspace changed while taking snapshot")
                entry["data"] = base64.b64encode(data).decode("ascii")
                files[name] = entry
            else:
                raise RuntimeError(
                    "workspace artifact is not a regular file or directory"
                )

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace root is not a regular directory")
    visit(root)
    return files


def restore(root: Path, files: dict[str, Any]) -> None:
    """Replace a disposable workspace with a validated stored tree.

    Files absent from the snapshot are removed. Validation and materialization
    finish before replacing the old directory, so invalid stored paths leave it
    untouched. The caller must exclude live executors during replacement.

    Args:
        root: The disposable workspace directory.
        files: A tree produced by ``snapshot``.

    Raises:
        RuntimeError: If a stored path escapes the workspace or has an invalid parent.
    """
    if root.is_symlink():
        raise RuntimeError("workspace root is a symlink")
    for name, entry in files.items():
        relative = PurePosixPath(name)
        if (
            not name
            or not relative.parts
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(
                part in {".", ".."} or "\\" in part or ":" in part
                for part in relative.parts
            )
            or entry.get("kind", "file") not in {"file", "directory"}
        ):
            raise RuntimeError("invalid stored workspace path")
        for parent in relative.parents:
            if (
                str(parent) != "."
                and files.get(str(parent), {}).get("kind") != "directory"
            ):
                raise RuntimeError("invalid stored workspace parent")
    root.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".workspace-", dir=root.parent))
    try:
        for name, entry in sorted(
            files.items(), key=lambda item: (len(PurePosixPath(item[0]).parts), item[0])
        ):
            path = staged / name
            if entry.get("kind") == "directory":
                path.mkdir()
            else:
                path.write_bytes(base64.b64decode(entry["data"], validate=True))
        # Restore directory metadata last: adding children changes its mtime.
        for name, entry in sorted(files.items(), reverse=True):
            path = staged / name
            path.chmod(entry["mode"])
            os.utime(path, ns=(entry["mtime_ns"], entry["mtime_ns"]))
        if root.exists():
            shutil.rmtree(root)
        os.replace(staged, root)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
