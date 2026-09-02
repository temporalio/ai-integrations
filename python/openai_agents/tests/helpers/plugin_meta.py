"""Read the plugin's ``plugin.toml`` (the machine-readable metadata every plugin carries)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class PluginMeta:
    name: str
    coordinate: str
    root_api: str
    allow_final: bool

    @property
    def package_relpath(self) -> str:
        """Root API as a path relative to site-packages, e.g. ``temporalio/contrib/openai_agents``."""
        return self.root_api.replace(".", "/")


def load_plugin_meta(plugin_root: Path) -> PluginMeta:
    data = tomllib.loads((plugin_root / "plugin.toml").read_text(encoding="utf-8"))
    plugin = data["plugin"]
    release = data.get("release", {})
    return PluginMeta(
        name=plugin["name"],
        coordinate=plugin["coordinate"],
        root_api=plugin["root-api"],
        allow_final=bool(release.get("allow-final", False)),
    )
