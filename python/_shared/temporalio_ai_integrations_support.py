"""Shared metadata and installation-provenance support for plugin tests."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: tomllib arrived in 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class PluginMeta:
    """Metadata needed by the shared test support."""

    coordinate: str
    root_api: str
    allow_final: bool

    @property
    def package_relpath(self) -> str:
        """Return the root API as a path relative to site-packages."""
        return self.root_api.replace(".", "/")


def load_plugin_meta(plugin_root: Path) -> PluginMeta:
    """Load the metadata needed by the shared test support from ``plugin.toml``."""
    data = tomllib.loads((plugin_root / "plugin.toml").read_text(encoding="utf-8"))
    plugin = data["plugin"]
    release = data.get("release", {})
    return PluginMeta(
        coordinate=plugin["coordinate"],
        root_api=plugin["root-api"],
        allow_final=bool(release.get("allow-final", False)),
    )


class ProvenanceError(RuntimeError):
    """The installed distribution is not the one this checkout built."""


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_provenance(
    dist_name: str,
    pkg_rel: str,
    *,
    allow_overlap: bool = False,
    warn: Callable[[str], None] | None = None,
) -> Path:
    """Verify the installed distribution and return its package directory."""
    emit = warn or (lambda message: print(f"WARNING: {message}"))
    try:
        dist = importlib_metadata.distribution(dist_name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ProvenanceError(f"{dist_name} is not installed; run `make sync`") from exc

    direct_url = dist.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise ProvenanceError(
            f"{dist_name} is installed editable, so temporalio.contrib.* cannot resolve to src/; "
            "run `make sync` (it exports UV_NO_EDITABLE=1)"
        )

    files = dist.files
    if files is None:
        raise ProvenanceError(f"{dist_name} has no RECORD; reinstall with `make sync`")

    owned: set[str] = set()
    for record_path in files:
        if record_path.hash is None or str(record_path).endswith(".pyc"):
            continue
        path = Path(str(record_path.locate()))
        owned.add(os.path.normpath(str(path)))
        if not path.is_file():
            raise ProvenanceError(
                f"{record_path} is in RECORD but missing on disk; run `make sync`"
            )
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
            .rstrip(b"=")
            .decode()
        )
        if record_path.hash.mode != "sha256" or digest != record_path.hash.value:
            raise ProvenanceError(
                f"{record_path} differs from RECORD (overwritten by another distribution?); "
                "run `make sync`"
            )

    pkg_dir = Path(str(dist.locate_file(pkg_rel)))
    if not pkg_dir.is_dir():
        raise ProvenanceError(f"{pkg_dir} does not exist; run `make sync`")
    extras = {
        os.path.normpath(str(path))
        for path in pkg_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    } - owned
    if allow_overlap:
        # temporalio<=1.32 ships this extra file until the SDK cutover.
        extras.discard(os.path.normpath(str(pkg_dir / "README.md")))
    if extras:
        raise ProvenanceError(
            f"files under {pkg_dir} are not owned by {dist_name}: {sorted(extras)}; "
            "run `make sync`"
        )

    prefix = pkg_rel.rstrip("/") + "/"
    others = sorted(
        {
            other.metadata["Name"]
            for other in importlib_metadata.distributions()
            if _normalize(other.metadata["Name"]) != _normalize(dist_name)
            and any(
                str(file).replace(os.sep, "/").startswith(prefix)
                for file in (other.files or [])
            )
        }
    )
    if others:
        message = f"{prefix} is also shipped by {others}"
        if allow_overlap:
            emit(
                f"{message}; tolerated until the SDK cutover "
                "(plugin.toml [release] allow-final = false)"
            )
        else:
            raise ProvenanceError(message)
    return pkg_dir
