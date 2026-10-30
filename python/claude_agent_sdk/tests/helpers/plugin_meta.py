"""Read the plugin's ``plugin.toml`` (the machine-readable metadata every plugin carries)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Each branch is unreachable under one interpreter version; basedpyright fails on that warning.
if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[reportUnreachable]
else:  # Python 3.10: tomllib arrived in 3.11
    import tomli as tomllib  # type: ignore[reportUnreachable]


@dataclass(frozen=True)
class PluginMeta:
    name: str
    coordinate: str
    root_api: str

    @property
    def package_relpath(self) -> str:
        """Root API as a path relative to site-packages, e.g. ``temporalio/openai_agents``."""
        return self.root_api.replace(".", "/")


def load_plugin_meta(plugin_root: Path) -> PluginMeta:
    data = tomllib.loads((plugin_root / "plugin.toml").read_text(encoding="utf-8"))
    plugin = data["plugin"]
    return PluginMeta(
        name=plugin["name"],
        coordinate=plugin["coordinate"],
        root_api=plugin["root-api"],
    )
