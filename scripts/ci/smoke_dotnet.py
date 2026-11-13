#!/usr/bin/env python3
"""Restore a .NET plugin's package in a clean consumer and load its public types.

Default (`--dist`): the consumer's only source for the plugin coordinate is the tested `dist/`
folder, dependencies come from nuget.org, and the restored `.nupkg` must be byte-identical to the
tested file. With `--source URL` the consumer restores the plugin from that feed instead (the
release pipeline passes nuget.org). nuget.org appends a repository signature to every upload, so the
restored package must then equal the tested one in every entry except a trailing stored
`.signature.p7s`; scripts/release/nuget_release.py applies the same rule to the flat container.

The consumer is rendered from dotnet/_shared/smoke/ into a temporary directory with its own
NUGET_PACKAGES, NUGET_HTTP_CACHE_PATH and NUGET_PLUGINS_CACHE_PATH, so nothing cached on the machine
can satisfy the restore, and with a global.json that pins the plugin's SDK channel so the newest
SDK on the machine is not the one under test. Program.cs loads every `[smoke] imports` type with
Type.GetType, requires the loaded assembly to be byte-identical to a lib/<tfm>/ assembly in the
restored package and its informational version to be the version under test. Same role as
smoke_java.py and typescript/_shared/smoke.mjs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

TEMPLATES = Path(__file__).resolve().parents[2] / "dotnet" / "_shared" / "smoke"
NUGET_ORG = "https://api.nuget.org/v3/index.json"
SIGNATURE = ".signature.p7s"


def unsigned_entries(payload: bytes) -> list[tuple[str, int, int, bytes]]:
    """Zip entries as (name, size, crc, bytes) without the trailing stored repository signature.

    nuget.org appends `.signature.p7s` as the last stored entry of every upload and rewrites nothing
    else, so this is the part of a served package that must equal the tested package. A signature
    anywhere else is not that case and is an error.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        if infos and infos[-1].filename == SIGNATURE and infos[-1].compress_type == zipfile.ZIP_STORED:
            infos = infos[:-1]
        if any(info.filename == SIGNATURE for info in infos):
            raise ValueError("package has a signature entry that is not the trailing stored repository signature")
        return [(info.filename, info.file_size, info.CRC, archive.read(info)) for info in infos]


def _attribute(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def render(consumer: Path, coordinate: str, version: str, channel: str, imports: list[str], dist: Path, source: str | None) -> None:
    """Write global.json, Consumer.csproj, Program.cs and nuget.config from the shared templates."""
    tfm = f"net{channel}"
    def template(name: str) -> str:
        return (TEMPLATES / name).read_text(encoding="utf-8")

    if source is None:
        sources = (f'    <add key="tested" value="{_attribute(str(dist))}" />\n'
                   f'    <add key="nuget.org" value="{NUGET_ORG}" protocolVersion="3" />')
        mapping = (f'    <packageSource key="tested">\n      <package pattern="{_attribute(coordinate)}" />\n    </packageSource>\n'
                   '    <packageSource key="nuget.org">\n      <package pattern="*" />\n    </packageSource>')
    else:
        sources = f'    <add key="under-test" value="{_attribute(source)}" protocolVersion="3" />'
        mapping = '    <packageSource key="under-test">\n      <package pattern="*" />\n    </packageSource>'
    (consumer / "nuget.config").write_text(
        template("nuget.config.tmpl").replace("__PACKAGE_SOURCES__", sources).replace("__PACKAGE_SOURCE_MAPPING__", mapping),
        encoding="utf-8")
    (consumer / "global.json").write_text(template("global.json.tmpl").replace("__CHANNEL__", channel), encoding="utf-8")
    (consumer / "Consumer.csproj").write_text(
        template("Consumer.csproj.tmpl").replace("__TFM__", tfm).replace("__COORDINATE__", coordinate).replace("__VERSION__", version),
        encoding="utf-8")
    types = ", ".join(json.dumps(name) for name in imports)
    (consumer / "Program.cs").write_text(
        template("Program.cs.tmpl").replace("__COORDINATE__", coordinate).replace("__VERSION__", version).replace("__TYPES__", types),
        encoding="utf-8")


def smoke(plugin_dir: Path, version: str, dist: Path | None = None, source: str | None = None,
          attempts: int = 1, delay: float = 30) -> None:
    plugin_dir = plugin_dir.resolve()
    meta = tomllib.loads((plugin_dir / "plugin.toml").read_text(encoding="utf-8"))
    coordinate = meta["plugin"]["coordinate"]
    imports = [str(name) for name in meta.get("smoke", {}).get("imports", [])]
    if not imports:
        raise ValueError("plugin.toml [smoke] imports must name at least one type")
    channel = str(meta["ci"]["runtime-versions"][-1])
    dist = (dist or plugin_dir / "dist").resolve()
    tested = dist / f"{coordinate}.{version}.nupkg"
    if not tested.is_file():
        raise ValueError(f"tested package {tested} is missing; run `make build`")
    tested_bytes = tested.read_bytes()
    if attempts < 1 or delay < 0:
        raise ValueError("smoke needs attempts >= 1 and delay >= 0")
    with tempfile.TemporaryDirectory(prefix="dotnet-consumer-") as directory:
        root = Path(directory)
        consumer = root / "consumer"
        consumer.mkdir()
        render(consumer, coordinate, version, channel, imports, dist, source)
        packages = root / "packages"
        http_cache = root / "http-cache"
        # The plugin's make targets export locked-mode restores; the consumer has no lock of its own.
        env = {key: value for key, value in os.environ.items() if key not in {"RestoreLockedMode", "RestorePackagesWithLockFile"}}
        env.update(NUGET_PACKAGES=str(packages), NUGET_HTTP_CACHE_PATH=str(http_cache),
                   NUGET_PLUGINS_CACHE_PATH=str(root / "plugins-cache"), DOTNET_CLI_TELEMETRY_OPTOUT="1",
                   DOTNET_NOLOGO="1", DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1", DOTNET_GENERATE_ASPNET_CERTIFICATE="false",
                   MSBUILDDISABLENODEREUSE="1")
        command = ["dotnet", "run", "--project", "Consumer.csproj", "-c", "Release"]
        for attempt in range(1, attempts + 1):
            if subprocess.run(command, cwd=consumer, env=env).returncode == 0:
                break
            if attempt == attempts:
                raise ValueError(f"clean consumer failed to restore, build or run after {attempts} attempt(s)")
            print(f"::warning::clean consumer failed (attempt {attempt}/{attempts}); retrying in {delay:.0f}s", flush=True)
            # A restore that ran before the version propagated caches the miss; start the next attempt clean.
            for stale in (packages, http_cache, consumer / "obj", consumer / "bin"):
                shutil.rmtree(stale, ignore_errors=True)
            time.sleep(delay)
        restored = packages / coordinate.lower() / version.lower() / f"{coordinate.lower()}.{version.lower()}.nupkg"
        if not restored.is_file():
            raise ValueError(f"NuGet did not restore {coordinate} {version} into the isolated packages folder")
        restored_bytes = restored.read_bytes()
        if source is None:
            if restored_bytes != tested_bytes:
                raise ValueError("the restored package differs from the tested dist/ file")
        elif unsigned_entries(restored_bytes) != unsigned_entries(tested_bytes):
            raise ValueError(f"{source} serves different package entries than the tested file; "
                             "only a trailing repository signature may differ, so fix forward with the next version")
    print(f"OK: clean consumer restored {coordinate} {version} from {source or dist} and loaded {len(imports)} type(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, help="tested packages (default: <plugin-dir>/dist)")
    parser.add_argument("--source", help="V3 feed to restore the plugin from instead of the tested dist/ folder")
    parser.add_argument("--attempts", type=int, default=1, help="restore attempts while a feed propagates a new version")
    parser.add_argument("--delay", type=float, default=30, help="seconds between attempts")
    args = parser.parse_args()
    try:
        smoke(args.plugin_dir, args.version, args.dist, args.source, args.attempts, args.delay)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
