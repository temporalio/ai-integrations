#!/usr/bin/env python3
"""Narrow .NET analyzer/editorconfig configuration to the entries with an observable effect.

Method (remove-and-measure): for every entry of the given config files, remove it with all other
entries present, run the build (B1) and the format verification (B2) exactly as the plugin's
`make test`/`make build` and `make lint` run them, and record the effect relative to the baseline
measured on the unmodified tree. Entries are then removed together (the ones with no effect) and
the combined removal is re-measured; if it has an effect, a delta-debugging pass finds the minimal
interacting subset, restores one member per interaction (first in scope order) and repeats.

Standard library only; re-runnable on another target framework or SDK.

    uv run --project scripts --locked python scripts/dev/narrow_dotnet_analyzers.py measure \
        --tree /tmp/narrow --plugin-dir dotnet/microsoft-agent-framework \
        --config dotnet/_shared/analyzers.globalconfig \
        --config dotnet/_shared/analyzers.tests.globalconfig \
        --config dotnet/_template/.editorconfig \
        --mirror dotnet/_template/.editorconfig=dotnet/microsoft-agent-framework/.editorconfig \
        --also-key global_level --auto-combos --workers 4 --output results.json
    uv run --project scripts --locked python scripts/dev/narrow_dotnet_analyzers.py report \
        --results results.json --output tables.md

`--tree` is a scratch checkout of the commit to measure (a `git worktree add --detach` of HEAD, or a
clone); the script adds worker worktrees next to it. Re-run it when the SDK channel, the target
framework or an analyzer package moves, and keep an entry only with a comment stating what fails
without it (AGENTS.md, ".NET conventions").

An entry is one `dotnet_diagnostic.X.severity = S` line (global configs: only those lines; other
keys such as is_global/global_level are structural and are measured only when named with
--also-key), one `key = value` option line of an .editorconfig (except `root`), or one whole naming
rule group (`dotnet_naming_rule.<r>.*` plus the `dotnet_naming_symbols`/`dotnet_naming_style`
groups it references and no other rule shares). Section headers and comments are not entries.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------------------------

SECTION_RE = re.compile(r"^\s*\[(?P<glob>.+)\]\s*$")
KV_RE = re.compile(r"^\s*(?P<key>[^#;=\s][^=]*?)\s*=\s*(?P<value>.*?)\s*$")
COMMENT_RE = re.compile(r"^\s*[#;]")
STRUCTURAL_KEYS = {"root", "is_global", "global_level"}
EDITOR_BASICS = {"charset", "end_of_line", "insert_final_newline", "trim_trailing_whitespace",
                 "indent_style", "indent_size", "tab_width", "max_line_length"}
FORMATTING_PREFIXES = ("csharp_indent_", "csharp_new_line_", "csharp_space_", "csharp_preserve_",
                       "dotnet_sort_system_directives_first", "dotnet_separate_import_directive_groups")
NAMING_RE = re.compile(r"^dotnet_naming_(?P<part>rule|symbols|style)\.(?P<name>[^.]+)\.(?P<prop>.+)$")
DIAG_RE = re.compile(r"^dotnet_diagnostic\.(?P<id>[^.]+)\.severity$")


@dataclasses.dataclass
class Entry:
    id: str
    file: str
    section: str
    key: str
    value: str
    lines: List[int]            # 0-based line numbers removed when the entry is removed
    first_line: int             # 1-based line of the first removed line (for reporting)
    comment: str
    kind: str
    structural: bool = False    # measured only through --also-key; never dropped by the rule

    def label(self) -> str:
        scope = f"[{self.section}]" if self.section else ""
        return f"{Path(self.file).name}{scope} {self.key}"


def classify(key: str, value: str) -> str:
    m = DIAG_RE.match(key)
    if m:
        sev = value.strip().lower()
        if sev in ("none", "silent", "suggestion", "hidden"):
            return "suppression"
        if sev in ("warning", "error"):
            return "enablement"
        return "severity-" + sev
    if key in EDITOR_BASICS:
        return "editor-basic"
    if key.startswith(FORMATTING_PREFIXES):
        return "formatting"
    if key.startswith(("csharp_style_", "dotnet_style_", "csharp_prefer_", "csharp_using_directive_placement",
                       "dotnet_code_quality", "dotnet_remove_unnecessary", "dotnet_analyzer_diagnostic",
                       "file_header_template", "dotnet_diagnostic")):
        return "style"
    return "other"


def _comment_above(lines: Sequence[str], index: int) -> str:
    """The contiguous comment block above line `index`, skipping entry lines of the same block."""
    i = index - 1
    while i >= 0 and (KV_RE.match(lines[i]) and not COMMENT_RE.match(lines[i])):
        i -= 1
    block: List[str] = []
    while i >= 0 and COMMENT_RE.match(lines[i]):
        block.append(lines[i].strip().lstrip("#;").strip())
        i -= 1
    return " ".join(reversed(block))


def parse_entries(file: str, text: str, also_keys: Set[str]) -> List[Entry]:
    lines = text.split("\n")
    is_global = any(re.match(r"^\s*is_global\s*=\s*true\s*$", l) for l in lines)
    entries: List[Entry] = []
    section = ""
    naming: Dict[Tuple[str, str, str], List[Tuple[int, str, str]]] = {}  # (section, part, name) -> [(line, prop, value)]
    for n, line in enumerate(lines):
        sm = SECTION_RE.match(line)
        if sm:
            section = sm.group("glob").strip()
            continue
        if COMMENT_RE.match(line) or not line.strip():
            continue
        km = KV_RE.match(line)
        if not km:
            continue
        key, value = km.group("key"), km.group("value")
        nm = NAMING_RE.match(key)
        if nm:
            naming.setdefault((section, nm.group("part"), nm.group("name")), []).append((n, nm.group("prop"), value))
            continue
        structural = key in STRUCTURAL_KEYS
        if structural and key not in also_keys:
            continue
        if is_global and not DIAG_RE.match(key) and key not in also_keys:
            continue
        scope = f"[{section}]" if section else ""
        eid = f"{file}:{scope}:{key}"
        if any(e.id == eid for e in entries):
            eid = f"{eid}@{n + 1}"
        entries.append(Entry(id=eid, file=file, section=section, key=key, value=value, lines=[n],
                             first_line=n + 1, comment=_comment_above(lines, n),
                             kind="structural" if structural else classify(key, value), structural=structural))
    # Naming rule groups: a rule plus the symbols/style groups only it references.
    rules = [(sec, name, props) for (sec, part, name), props in naming.items() if part == "rule"]
    referenced: Dict[Tuple[str, str, str], Set[str]] = {}
    for sec, name, props in rules:
        for _, prop, value in props:
            if prop in ("symbols", "style"):
                referenced.setdefault((sec, prop, value.strip()), set()).add(name)
    claimed: Set[Tuple[str, str, str]] = set()
    for sec, name, props in rules:
        group_lines = [n for n, _, _ in props]
        for _, prop, value in props:
            if prop in ("symbols", "style"):
                target = (sec, prop, value.strip())
                if referenced.get(target) == {name}:
                    group_lines += [n for n, _, _ in naming.get(target, [])]
                    claimed.add(target)
        group_lines.sort()
        scope = f"[{sec}]" if sec else ""
        entries.append(Entry(id=f"{file}:{scope}:dotnet_naming_rule.{name}", file=file, section=sec,
                             key=f"dotnet_naming_rule.{name}", value="(group)", lines=group_lines,
                             first_line=group_lines[0] + 1, comment=_comment_above(lines, group_lines[0]), kind="naming"))
    for (sec, part, name), props in naming.items():
        if part == "rule" or (sec, part, name) in claimed or (sec, part, name) in referenced:
            continue
        group_lines = sorted(n for n, _, _ in props)
        scope = f"[{sec}]" if sec else ""
        entries.append(Entry(id=f"{file}:{scope}:dotnet_naming_{part}.{name}", file=file, section=sec,
                             key=f"dotnet_naming_{part}.{name}", value="(orphan group)", lines=group_lines,
                             first_line=group_lines[0] + 1, comment=_comment_above(lines, group_lines[0]), kind="naming"))
    entries.sort(key=lambda e: e.lines[0])
    return entries


def remove_lines(text: str, lines_to_drop: Iterable[int]) -> str:
    drop = set(lines_to_drop)
    return "\n".join(l for n, l in enumerate(text.split("\n")) if n not in drop)


# --------------------------------------------------------------------------------------------
# Running the build and the format check
# --------------------------------------------------------------------------------------------

BUILD_DIAG_RE = re.compile(
    r"^(?:(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\)|(?P<origin>[^\s(][^(]*?))\s?: "
    r"(?P<kind>warning|error) (?P<id>[A-Za-z]+\d+): (?P<msg>.*?)(?: \[(?P<proj>[^\]]+)\])?$")
INFRA_MARKERS = ("MSB4166", "MSB4025", "MSB1025", "NU1301", "Could not connect", "exited prematurely",
                 "The build stopped unexpectedly", "Unhandled exception")
NON_DETERMINISTIC_IDS = ("NU1900", "NU1901", "NU1902", "NU1903", "NU1904")


@dataclasses.dataclass
class Tree:
    name: str
    root: Path
    created: bool = False


class Runner:
    def __init__(self, dotnet: str, plugin_rel: str, slnx: str, timestamp: str, env: Dict[str, str],
                 retries: int = 3, timeout: int = 1800) -> None:
        self.dotnet = dotnet
        self.plugin_rel = plugin_rel
        self.slnx = slnx
        self.timestamp = timestamp
        self.env = env
        self.retries = retries
        self.timeout = timeout

    def build_args(self, survey: bool = False) -> List[str]:
        args = [self.dotnet, "build", self.slnx, "-c", "Release", "--no-incremental",
                "-p:Version=0.0.0", "-p:ContinuousIntegrationBuild=false"]
        if self.timestamp:
            args.append(f"-p:DeterministicTimestamp={self.timestamp}")
        if survey:
            args.append("-p:TreatWarningsAsErrors=false")
        return args

    def format_args(self, report_dir: str) -> List[str]:
        return [self.dotnet, "format", self.slnx, "--verify-no-changes", "--exclude-diagnostics", "CA2007",
                "--report", report_dir]

    def _run(self, args: List[str], cwd: Path) -> Tuple[int, str, float]:
        start = time.monotonic()
        try:
            proc = subprocess.run(args, cwd=str(cwd), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", timeout=self.timeout)
            code, out = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired as exc:
            code, out = -999, (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            out += "\n[timeout]"
        return code, out, time.monotonic() - start

    @staticmethod
    def parse_build(out: str, plugin_dir: Path) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, int, int, str, str]] = set()
        diags: List[Dict[str, Any]] = []
        for raw in out.splitlines():
            m = BUILD_DIAG_RE.match(raw.strip())
            if not m or m.group("id").startswith(NON_DETERMINISTIC_IDS):
                continue
            file = m.group("file") or m.group("origin") or ""
            line = int(m.group("line") or 0)
            col = int(m.group("col") or 0)
            # A multi-targeted project reports once per framework as "X.csproj::TargetFramework=net8.0";
            # a site is counted once, so the framework suffix is dropped from the key.
            proj = (m.group("proj") or "").split("::", 1)[0]
            rel = _relpath(file, plugin_dir)
            key = (rel, line, col, m.group("id"), Path(proj).stem if proj else "")
            if key in seen:
                continue
            seen.add(key)
            diags.append({"file": rel, "line": line, "col": col, "id": m.group("id"), "kind": m.group("kind"),
                          "project": Path(proj).stem if proj else "", "message": m.group("msg")})
        diags.sort(key=lambda d: (d["file"], d["line"], d["col"], d["id"]))
        return diags

    @staticmethod
    def parse_format(out: str, report_dir: Path, plugin_dir: Path) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, int, int, str]] = set()
        changes: List[Dict[str, Any]] = []
        report = report_dir / "format-report.json"
        if report.is_file():
            try:
                data = json.loads(report.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                data = []
            for item in data:
                rel = _relpath(item.get("FilePath", ""), plugin_dir)
                for change in item.get("FileChanges", []):
                    key = (rel, int(change.get("LineNumber", 0)), int(change.get("CharNumber", 0)), change.get("DiagnosticId", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    changes.append({"file": rel, "line": key[1], "col": key[2], "id": key[3],
                                    "message": change.get("FormatDescription", "")})
        for raw in out.splitlines():  # fallback for runs where the report was not written
            m = BUILD_DIAG_RE.match(raw.strip())
            if not m or not m.group("file"):
                continue
            key = (_relpath(m.group("file"), plugin_dir), int(m.group("line")), int(m.group("col")), m.group("id"))
            if key in seen:
                continue
            seen.add(key)
            changes.append({"file": key[0], "line": key[1], "col": key[2], "id": key[3], "message": m.group("msg")})
        changes.sort(key=lambda d: (d["file"], d["line"], d["col"], d["id"]))
        return changes

    def measure(self, tree: Tree) -> Dict[str, Any]:
        """B1 (build, warnings as errors), survey build when B1 fails, and B2 (format verification)."""
        plugin_dir = tree.root / self.plugin_rel
        result: Dict[str, Any] = {}
        for attempt in range(1, self.retries + 1):
            code, out, secs = self._run(self.build_args(), plugin_dir)
            diags = self.parse_build(out, plugin_dir)
            infra = code != 0 and not diags and (code < 0 or code > 1 or any(mk in out for mk in INFRA_MARKERS))
            if not infra:
                break
            sys.stderr.write(f"[{tree.name}] build infrastructure failure (exit {code}), retry {attempt}/{self.retries}\n")
            time.sleep(10 * attempt)
        result["build"] = {"exit": code, "seconds": round(secs, 2), "diagnostics": diags, "tail": _tail(out)}
        if code != 0 or diags:
            scode, sout, ssecs = self._run(self.build_args(survey=True), plugin_dir)
            result["survey"] = {"exit": scode, "seconds": round(ssecs, 2), "diagnostics": self.parse_build(sout, plugin_dir),
                                "tail": _tail(sout)}
        with tempfile.TemporaryDirectory(prefix="fmt-report-") as tmp:
            for attempt in range(1, self.retries + 1):
                code, out, secs = self._run(self.format_args(tmp), plugin_dir)
                changes = self.parse_format(out, Path(tmp), plugin_dir)
                infra = code not in (0, 2) or (code == 2 and not changes) or any(mk in out for mk in INFRA_MARKERS)
                if not infra:
                    break
                sys.stderr.write(f"[{tree.name}] format infrastructure failure (exit {code}), retry {attempt}/{self.retries}\n")
                time.sleep(10 * attempt)
            result["format"] = {"exit": code, "seconds": round(secs, 2), "changes": changes, "tail": _tail(out)}
        return result


def _relpath(path: str, base: Path) -> str:
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.is_absolute():
            return path
        return str(p.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return path


def _tail(out: str, n: int = 12) -> str:
    lines = [l for l in out.splitlines() if l.strip()]
    return "\n".join(lines[-n:])


# --------------------------------------------------------------------------------------------
# Comparison with the baseline
# --------------------------------------------------------------------------------------------

def _keys(items: Iterable[Dict[str, Any]]) -> Set[Tuple[str, int, int, str]]:
    return {(d["file"], d["line"], d["col"], d["id"]) for d in items}


def effect(baseline: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    base_build = _keys(baseline["build"]["diagnostics"])
    base_format = _keys(baseline["format"]["changes"])
    new_build = [d for d in run["build"]["diagnostics"] if (d["file"], d["line"], d["col"], d["id"]) not in base_build]
    gone_build = sorted(base_build - _keys(run["build"]["diagnostics"]))
    survey_src = run.get("survey", {}).get("diagnostics", [])
    new_survey = [d for d in survey_src if (d["file"], d["line"], d["col"], d["id"]) not in base_build]
    new_format = [d for d in run["format"]["changes"] if (d["file"], d["line"], d["col"], d["id"]) not in base_format]
    gone_format = sorted(base_format - _keys(run["format"]["changes"]))
    build_effect = run["build"]["exit"] != baseline["build"]["exit"] or bool(new_build) or bool(gone_build)
    format_effect = run["format"]["exit"] != baseline["format"]["exit"] or bool(new_format) or bool(gone_format)
    kinds = [k for k, flag in (("build", build_effect), ("format", format_effect)) if flag]
    return {"summary": "+".join(kinds) if kinds else "none",
            "build_exit": run["build"]["exit"], "format_exit": run["format"]["exit"],
            "new_build": new_build, "gone_build": gone_build,
            "new_survey": new_survey if new_survey else new_build,
            "new_format": new_format, "gone_format": gone_format}


def summarize_diags(diags: List[Dict[str, Any]]) -> str:
    """'CA2007 x39 (src: ModelActivities.cs x4, ...)' style summary grouped by id and top-level folder."""
    by_id: Dict[str, Dict[str, Dict[str, int]]] = {}
    for d in diags:
        folder = d["file"].split("/", 1)[0] if "/" in d["file"] else "(project)"
        name = Path(d["file"]).name if d["file"] else d.get("project", "")
        by_id.setdefault(d["id"], {}).setdefault(folder, {})
        by_id[d["id"]][folder][name] = by_id[d["id"]][folder].get(name, 0) + 1
    parts = []
    for did in sorted(by_id):
        total = sum(sum(v.values()) for v in by_id[did].values())
        folders = []
        for folder in sorted(by_id[did]):
            files = by_id[did][folder]
            fsum = sum(files.values())
            names = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in sorted(files.items())[:6])
            more = f", +{len(files) - 6} files" if len(files) > 6 else ""
            folders.append(f"{folder} x{fsum}: {names}{more}")
        parts.append(f"{did} x{total} ({'; '.join(folders)})")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------

class Workspace:
    def __init__(self, args: argparse.Namespace) -> None:
        self.tree = Path(args.tree).resolve()
        self.plugin_rel = args.plugin_dir
        self.configs: List[str] = args.config
        self.mirrors: Dict[str, List[str]] = {}
        for spec in args.mirror or []:
            src, dst = spec.split("=", 1)
            self.mirrors.setdefault(src, []).append(dst)
        self.pristine: Dict[str, str] = {c: (self.tree / c).read_text(encoding="utf-8") for c in self.configs}
        for src, dsts in self.mirrors.items():
            for dst in dsts:
                if (self.tree / dst).read_bytes() != (self.tree / src).read_bytes():
                    raise SystemExit(f"{dst} is not byte-identical to {src}; refusing to measure")
        self.also_keys = set(args.also_key or [])
        self.entries: List[Entry] = []
        for c in self.configs:
            self.entries += parse_entries(c, self.pristine[c], self.also_keys)
        self.lock = threading.Lock()

    def all_targets(self, config: str) -> List[str]:
        return [config] + self.mirrors.get(config, [])

    def write_removed(self, tree: Tree, removed: Sequence[Entry]) -> None:
        by_file: Dict[str, Set[int]] = {}
        for e in removed:
            by_file.setdefault(e.file, set()).update(e.lines)
        for c in self.configs:
            text = remove_lines(self.pristine[c], by_file.get(c, set())) if c in by_file else self.pristine[c]
            for target in self.all_targets(c):
                (tree.root / target).write_text(text, encoding="utf-8", newline="\n")

    def restore(self, tree: Tree) -> None:
        self.write_removed(tree, [])


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def prepare_trees(ws: Workspace, args: argparse.Namespace, runner: Runner) -> List[Tree]:
    head = git(ws.tree, "rev-parse", "HEAD")
    trees = [Tree("main", ws.tree)]
    wanted = args.worker_tree or [f"{ws.tree}-w{i}" for i in range(1, args.workers + 1)]
    for i, path in enumerate(wanted[: args.workers], start=1):
        root = Path(path).resolve()
        created = False
        if not root.exists():
            sys.stderr.write(f"creating worktree {root} at {head[:8]}\n")
            git(ws.tree, "worktree", "add", "--detach", str(root), head)
            created = True
        if git(root, "rev-parse", "HEAD") != head:
            raise SystemExit(f"{root} is not at {head[:8]}")
        trees.append(Tree(f"w{i}", root, created))
    # Workers replicate the main tree's working state: every tracked file modified in the main tree
    # (config files and, for example, source fixes that go with a dropped suppression) is copied over.
    modified = [l for l in git(ws.tree, "diff", "--name-only", "HEAD").splitlines() if (ws.tree / l).is_file()]
    for t in trees[1:]:
        for rel in modified:
            (t.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ws.tree / rel, t.root / rel)
        ws.restore(t)   # worker configs = main tree's current (pristine) content
        allowed = set(modified) | set(sum((ws.all_targets(c) for c in ws.configs), []))
        dirty = [l for l in git(t.root, "diff", "--name-only", "HEAD").splitlines() if l not in allowed]
        if dirty:
            raise SystemExit(f"{t.root} has unrelated modifications: {dirty}")
        plugin_dir = t.root / ws.plugin_rel
        sys.stderr.write(f"[{t.name}] restore\n")
        subprocess.run([runner.dotnet, "restore", runner.slnx, "--locked-mode"], cwd=str(plugin_dir), env=runner.env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return trees


def measure_set(ws: Workspace, runner: Runner, tree: Tree, removed: Sequence[Entry]) -> Dict[str, Any]:
    try:
        ws.write_removed(tree, removed)
        return runner.measure(tree)
    finally:
        ws.restore(tree)


def run_parallel(ws: Workspace, runner: Runner, workers: List[Tree], jobs: List[Tuple[str, List[Entry]]],
                 baseline: Dict[str, Any], log_prefix: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    pool: List[Tree] = list(workers)
    pool_lock = threading.Lock()
    total = len(jobs)
    done = [0]

    def work(job: Tuple[str, List[Entry]]) -> Tuple[str, Dict[str, Any]]:
        name, removed = job
        with pool_lock:
            tree = pool.pop()
        try:
            start = time.monotonic()
            run = measure_set(ws, runner, tree, removed)
            eff = effect(baseline, run)
            secs = time.monotonic() - start
            with ws.lock:
                done[0] += 1
                sys.stderr.write(f"[{tree.name}] {log_prefix} {done[0]}/{total} {name} -> {eff['summary']}"
                                 f"{': ' + summarize_diags(eff['new_survey']) if eff['new_survey'] else ''}"
                                 f"{' | format: ' + summarize_diags(eff['new_format']) if eff['new_format'] else ''} ({secs:.1f}s)\n")
            return name, {"run": run, "effect": eff, "seconds": round(secs, 1), "worker": tree.name}
        finally:
            with pool_lock:
                pool.append(tree)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as ex:
        for name, res in ex.map(work, jobs):
            results[name] = res
    return results


def ddmin(test_fails, items: List[Entry]) -> List[Entry]:
    """Zeller's ddmin: a minimal subset of `items` whose joint removal still has an effect."""
    n = 2
    current = list(items)
    while len(current) >= 2:
        chunk = max(1, len(current) // n)
        subsets = [current[i:i + chunk] for i in range(0, len(current), chunk)]
        reduced = False
        for subset in subsets:
            if test_fails(subset):
                current, n, reduced = subset, 2, True
                break
        if not reduced:
            for subset in subsets:
                complement = [e for e in current if e not in subset]
                if complement and test_fails(complement):
                    current, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(current):
                break
            n = min(n * 2, len(current))
    return current


def cmd_measure(args: argparse.Namespace) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    ws = Workspace(args)
    plugin_dir = ws.tree / ws.plugin_rel
    slnx = [p.name for p in plugin_dir.glob("*.slnx")]
    if len(slnx) != 1:
        raise SystemExit(f"expected exactly one .slnx in {plugin_dir}, found {slnx}")
    dotnet = args.dotnet
    env = dict(os.environ)
    env.update({"DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
                "DOTNET_GENERATE_ASPNET_CERTIFICATE": "false", "RestoreLockedMode": "true",
                "DOTNET_CLI_UI_LANGUAGE": "en", "MSBUILDDISABLENODEREUSE": "1"})
    if os.path.isabs(dotnet):
        env["PATH"] = str(Path(dotnet).parent) + os.pathsep + env.get("PATH", "")
    timestamp = git(ws.tree, "log", "-1", "--pretty=%ct")
    runner = Runner(dotnet, ws.plugin_rel, slnx[0], timestamp, env, retries=args.retries)
    sdk = subprocess.run([dotnet, "--version"], cwd=str(plugin_dir), env=env, stdout=subprocess.PIPE, text=True).stdout.strip()

    entries = ws.entries
    if args.only:
        pats = [re.compile(p) for p in args.only]
        entries = [e for e in entries if any(p.search(e.id) for p in pats)]
    sys.stderr.write(f"{len(entries)} entries from {len(ws.configs)} config files; SDK {sdk}; dotnet={dotnet}\n")
    trees = prepare_trees(ws, args, runner)
    workers = trees[1:] if len(trees) > 1 else trees

    # Baseline on the unmodified tree (main) and a consistency check on every worker.
    sys.stderr.write("[main] baseline\n")
    t0 = time.monotonic()
    baseline = runner.measure(trees[0])
    baseline_seconds = time.monotonic() - t0
    sys.stderr.write(f"[main] baseline: build exit {baseline['build']['exit']} ({len(baseline['build']['diagnostics'])} diagnostics), "
                     f"format exit {baseline['format']['exit']} ({len(baseline['format']['changes'])} changes) ({baseline_seconds:.1f}s)\n")
    worker_baselines = run_parallel(ws, runner, workers, [(f"baseline@{t.name}", []) for t in workers], baseline, "baseline")
    for name, res in worker_baselines.items():
        if res["effect"]["summary"] != "none":
            raise SystemExit(f"worker baseline differs from main baseline: {name}: {res['effect']}")

    # Singles.
    jobs = [(e.id, [e]) for e in entries]
    single_results = run_parallel(ws, runner, workers, jobs, baseline, "entry")

    # Duplicate keys measured as a group (same key in more than one scope).
    combos: Dict[str, Dict[str, Any]] = {}
    combo_jobs: List[Tuple[str, List[Entry]]] = []
    if args.auto_combos:
        groups: Dict[str, List[Entry]] = {}
        for e in entries:
            if not e.structural:
                groups.setdefault(e.key, []).append(e)
        for key, members in groups.items():
            if len(members) > 1:
                combo_jobs.append((f"all:{key}", members))
    for spec in args.combo or []:
        ids = spec.split(",")
        members = [e for e in entries if e.id in ids]
        if len(members) != len(ids):
            raise SystemExit(f"unknown entry id in combo {spec}")
        combo_jobs.append((f"combo:{spec}", members))
    if combo_jobs:
        combos = run_parallel(ws, runner, workers, combo_jobs, baseline, "combo")
        for name, res in combos.items():
            res["members"] = [e.id for e in dict(combo_jobs)[name]]

    # Combined removal of every no-effect entry, with delta debugging on interactions.
    combined: Dict[str, Any] = {"rounds": [], "interactions": [], "restored": []}
    if not args.no_combined:
        drop = [e for e in entries if not e.structural and single_results[e.id]["effect"]["summary"] == "none"]
        tests = [0]

        def test_fails(subset: List[Entry]) -> bool:
            tests[0] += 1
            run = measure_set(ws, runner, workers[0], subset)
            eff = effect(baseline, run)
            sys.stderr.write(f"[{workers[0].name}] ddmin test {tests[0]}: {len(subset)} entries -> {eff['summary']}\n")
            return eff["summary"] != "none"

        while True:
            sys.stderr.write(f"[{workers[0].name}] combined removal of {len(drop)} entries\n")
            run = measure_set(ws, runner, workers[0], drop)
            eff = effect(baseline, run)
            combined["rounds"].append({"dropped": [e.id for e in drop], "effect": eff, "run": run})
            sys.stderr.write(f"[{workers[0].name}] combined -> {eff['summary']}\n")
            if eff["summary"] == "none" or not drop:
                break
            minimal = ddmin(test_fails, drop)
            keep = minimal[0]   # first in scope order stays; the maintainer may swap it for another member
            combined["interactions"].append({"members": [e.id for e in minimal], "restored": keep.id,
                                             "effect_when_all_removed": effect(baseline, measure_set(ws, runner, workers[0], minimal))})
            combined["restored"].append(keep.id)
            drop = [e for e in drop if e.id != keep.id]
        combined["final_drop"] = [e.id for e in drop]
        combined["ddmin_tests"] = tests[0]

    finished = dt.datetime.now(dt.timezone.utc)
    out = {
        "meta": {
            "tree": str(ws.tree), "head": git(ws.tree, "rev-parse", "HEAD"), "plugin_dir": ws.plugin_rel,
            "configs": ws.configs, "mirrors": ws.mirrors, "also_keys": sorted(ws.also_keys),
            "dotnet": dotnet, "sdk": sdk, "workers": [str(t.root) for t in workers],
            "started": started.isoformat(), "finished": finished.isoformat(),
            "wall_seconds": round((finished - started).total_seconds(), 1),
            "baseline_seconds": round(baseline_seconds, 1),
            "build_command": " ".join(runner.build_args()), "survey_command": " ".join(runner.build_args(survey=True)),
            "format_command": " ".join(runner.format_args("<report-dir>")),
            "env": {k: env[k] for k in ("DOTNET_CLI_TELEMETRY_OPTOUT", "DOTNET_NOLOGO", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE",
                                        "DOTNET_GENERATE_ASPNET_CERTIFICATE", "RestoreLockedMode", "MSBUILDDISABLENODEREUSE")},
        },
        "baseline": baseline,
        "entries": [dict(dataclasses.asdict(e), **single_results[e.id]) for e in entries],
        "combos": combos,
        "combined": combined,
    }
    Path(args.output).write_text(json.dumps(out, indent=1, sort_keys=False), encoding="utf-8")
    sys.stderr.write(f"wrote {args.output} ({out['meta']['wall_seconds']}s)\n")
    if args.remove_workers:
        for t in trees[1:]:
            if t.created:
                git(ws.tree, "worktree", "remove", "--force", str(t.root))
    return 0


# --------------------------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------------------------

def _md(s: str) -> str:
    return s.replace("|", "\\|")


def cmd_report(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    entries = data["entries"]
    restored = set(data.get("combined", {}).get("restored", []))
    interactions = data.get("combined", {}).get("interactions", [])
    lines: List[str] = []
    meta = data["meta"]
    b = data["baseline"]
    lines.append(f"Baseline: build exit {b['build']['exit']} with {len(b['build']['diagnostics'])} diagnostics; "
                 f"format exit {b['format']['exit']} with {len(b['format']['changes'])} changes. "
                 f"SDK {meta['sdk']}, {len(entries)} entries, wall {meta['wall_seconds']}s.")
    lines.append("")
    kept = [e for e in entries if e["effect"]["summary"] != "none" and not e["structural"]]
    kept_inter = [e for e in entries if e["id"] in restored]
    lines.append("## Table 1: kept entries with evidence")
    lines.append("")
    lines.append("| # | file | scope | entry | effect | evidence |")
    lines.append("|---|---|---|---|---|---|")
    for i, e in enumerate(kept + kept_inter, 1):
        eff = e["effect"]
        ev = []
        if eff["new_survey"]:
            ev.append("build: " + summarize_diags(eff["new_survey"]))
        if eff["new_format"]:
            ev.append("format: " + summarize_diags(eff["new_format"]))
        if e["id"] in restored:
            inter = next(x for x in interactions if x["restored"] == e["id"])
            ev.append("interaction: no effect alone; removing it together with " +
                      ", ".join(m for m in inter["members"] if m != e["id"]) + " -> " +
                      summarize_diags(inter["effect_when_all_removed"]["new_survey"]) + " " +
                      summarize_diags(inter["effect_when_all_removed"]["new_format"]))
        lines.append(f"| {i} | {Path(e['file']).name} | {_md(e['section']) or '-'} | `{_md(e['key'])} = {_md(e['value'])}` | "
                     f"{eff['summary']} | {_md('; '.join(ev))} |")
    lines.append("")
    dropped = [e for e in entries if e["effect"]["summary"] == "none" and not e["structural"] and e["id"] not in restored]
    lines.append("## Table 2: dropped suppressions whose rule never fires")
    lines.append("")
    lines.append("| # | file | scope | entry | comment in the file |")
    lines.append("|---|---|---|---|---|")
    for i, e in enumerate([e for e in dropped if e["kind"] == "suppression"], 1):
        lines.append(f"| {i} | {Path(e['file']).name} | {_md(e['section']) or '-'} | `{_md(e['key'])} = {_md(e['value'])}` | {_md(e['comment'])} |")
    lines.append("")
    lines.append("## Table 3: dropped enablements and style/formatting preferences with no effect")
    lines.append("")
    lines.append("| # | file | scope | entry | kind | comment in the file |")
    lines.append("|---|---|---|---|---|---|")
    for i, e in enumerate([e for e in dropped if e["kind"] != "suppression"], 1):
        lines.append(f"| {i} | {Path(e['file']).name} | {_md(e['section']) or '-'} | `{_md(e['key'])} = {_md(e['value'])}` | {e['kind']} | {_md(e['comment'])} |")
    lines.append("")
    structural = [e for e in entries if e["structural"]]
    if structural:
        lines.append("## Structural keys measured (not entries; never dropped by the rule)")
        lines.append("")
        for e in structural:
            lines.append(f"- `{e['key']} = {e['value']}` in {Path(e['file']).name}: {e['effect']['summary']}")
        lines.append("")
    if data.get("combos"):
        lines.append("## Duplicate keys removed as a group")
        lines.append("")
        for name, c in data["combos"].items():
            eff = c["effect"]
            lines.append(f"- {name} ({', '.join(c['members'])}): {eff['summary']}"
                         + (f"; build: {summarize_diags(eff['new_survey'])}" if eff["new_survey"] else "")
                         + (f"; format: {summarize_diags(eff['new_format'])}" if eff["new_format"] else ""))
        lines.append("")
    comb = data.get("combined", {})
    if comb.get("rounds"):
        lines.append("## Combined removal")
        lines.append("")
        for i, r in enumerate(comb["rounds"], 1):
            lines.append(f"- round {i}: {len(r['dropped'])} entries removed together -> {r['effect']['summary']}")
        lines.append(f"- ddmin tests: {comb.get('ddmin_tests', 0)}; restored by interaction: {', '.join(comb.get('restored', [])) or 'none'}")
        lines.append("")
    text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    ws = Workspace(args)
    for e in ws.entries:
        print(f"{e.first_line:4d} {e.kind:12s} {e.id}  = {e.value}")
    print(f"{len(ws.entries)} entries", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tree_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tree", required=True, help="scratch checkout root (repository root)")
        p.add_argument("--plugin-dir", required=True, help="plugin directory relative to the tree, holding the .slnx and Makefile")
        p.add_argument("--config", action="append", required=True, help="config file relative to the tree (repeatable, scope order)")
        p.add_argument("--mirror", action="append", help="SRC=DST: keep DST byte-identical to SRC while measuring (repeatable)")
        p.add_argument("--also-key", action="append", help="structural key to measure as well (e.g. global_level)")

    m = sub.add_parser("measure", help="run the remove-and-measure loop")
    tree_args(m)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--worker-tree", action="append", help="worker worktree path (default <tree>-wN; created if missing)")
    m.add_argument("--output", required=True, help="results JSON path")
    m.add_argument("--dotnet", default="dotnet", help="dotnet executable (its directory is prepended to PATH)")
    m.add_argument("--retries", type=int, default=3)
    m.add_argument("--only", action="append", help="regex on entry ids to restrict the run")
    m.add_argument("--combo", action="append", help="comma-separated entry ids to remove together (repeatable)")
    m.add_argument("--auto-combos", action="store_true", help="also remove every key that appears in more than one scope as a group")
    m.add_argument("--no-combined", action="store_true", help="skip the combined removal and the ddmin pass")
    m.add_argument("--remove-workers", action="store_true", help="remove worker worktrees this run created")
    m.set_defaults(func=cmd_measure)

    r = sub.add_parser("report", help="render markdown tables from a results JSON")
    r.add_argument("--results", required=True)
    r.add_argument("--output")
    r.set_defaults(func=cmd_report)

    l = sub.add_parser("list", help="list the entries that would be measured")
    tree_args(l)
    l.set_defaults(func=cmd_list)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
