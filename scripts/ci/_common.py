"""Shared helpers for the CI scripts (plugin discovery, TOML, GITHUB_OUTPUT).

Stdlib only. Imported by the sibling scripts via their own directory on sys.path.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Language root -> manifest file names that mark a plugin directory.
LANGUAGE_MANIFESTS: dict[str, tuple[str, ...]] = {
    "python": ("pyproject.toml",),
    "typescript": ("package.json",),
    "java": ("build.gradle", "build.gradle.kts"),
    "go": ("go.mod",),
}
LANGUAGES: tuple[str, ...] = tuple(LANGUAGE_MANIFESTS)

# Directories inside a language root that are never plugins.
IGNORED_PREFIXES = ("_", ".")


@dataclass(frozen=True)
class Plugin:
    language: str
    name: str
    path: Path

    @property
    def rel(self) -> str:
        return f"{self.language}/{self.name}"


def repo_root(start: Path | None = None) -> Path:
    """Locate the repository root (the directory containing the language roots)."""
    if start is None:
        env = os.environ.get("GITHUB_WORKSPACE")
        start = Path(env) if env else Path(__file__).resolve().parents[2]
    return start.resolve()


def is_plugin_dir(language: str, path: Path) -> bool:
    if not path.is_dir() or path.name.startswith(IGNORED_PREFIXES):
        return False
    return any((path / manifest).is_file() for manifest in LANGUAGE_MANIFESTS[language])


def discover_plugins(root: Path) -> dict[str, list[Plugin]]:
    """Return {language: [Plugin, ...]} for every language root present under `root`."""
    found: dict[str, list[Plugin]] = {}
    for language in LANGUAGES:
        lang_dir = root / language
        plugins: list[Plugin] = []
        if lang_dir.is_dir():
            for child in sorted(lang_dir.iterdir(), key=lambda p: p.name):
                if is_plugin_dir(language, child):
                    plugins.append(Plugin(language, child.name, child))
        found[language] = plugins
    return found


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def write_github_output(path: str | os.PathLike[str] | None, values: dict[str, Any]) -> None:
    """Append key=value lines to a GITHUB_OUTPUT file (values are JSON-encoded unless str)."""
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
            if "\n" in text:
                raise ValueError(f"multi-line output not supported for {key}")
            fh.write(f"{key}={text}\n")


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)
