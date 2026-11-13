"""check_dotnet_deps.py enforces the nuget.config cooldown on the committed locks without network."""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import check_dotnet_deps
import pytest
from conftest import dotnet_lock

NUGET_CONFIG = '''<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" minPublishAgeHours="336" />
  </packageSources>
  <minPublishAgeExceptions>
    <package pattern="Temporalio*" />
  </minPublishAgeExceptions>
</configuration>
'''


def plugin_with_lock(tmp_path: Path, packages: dict[str, str], config: str = NUGET_CONFIG) -> Path:
    plugin = tmp_path / "plugin"
    (plugin / "src" / "Temporalio.Fakedotnet").mkdir(parents=True)
    (plugin / "nuget.config").write_text(config)
    (plugin / "src" / "Temporalio.Fakedotnet" / "packages.lock.json").write_text(dotnet_lock(packages))
    return plugin


def registry(tmp_path: Path, leaves: dict[tuple[str, str], dict]) -> Path:
    root = tmp_path / "registry"
    for (package_id, version), leaf in leaves.items():
        path = root / package_id.lower() / f"{version.lower()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(leaf))
    return root


def published(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.1234567+00:00")


@pytest.mark.parametrize("age_days,ok", [(3, False), (13.9, False), (15, True), (400, True)])
def test_third_party_versions_must_be_older_than_the_cooldown(tmp_path: Path, age_days: float, ok: bool) -> None:
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3"})
    leaves = registry(tmp_path, {("xunit.v3", "4.0.3"): {"listed": True, "published": published(age_days)}})
    failures = check_dotnet_deps.check(plugin, leaves)
    assert (failures == []) is ok, failures
    if not ok:
        assert "xunit.v3 4.0.3 was published" in failures[0] and "336 hours" in failures[0]


def test_first_party_exception_skips_the_age_check_but_not_the_listing(tmp_path: Path) -> None:
    plugin = plugin_with_lock(tmp_path, {"Temporalio": "1.22.0", "Temporalio.Mcp": "0.1.0"})
    leaves = registry(tmp_path, {
        ("Temporalio", "1.22.0"): {"listed": True, "published": published(1)},
        ("Temporalio.Mcp", "0.1.0"): {"published": published(0.5)},  # absent `listed` means listed
    })
    assert check_dotnet_deps.check(plugin, leaves) == []
    leaves = registry(tmp_path, {
        ("Temporalio", "1.22.0"): {"listed": False, "published": "1900-01-01T00:00:00+00:00"},
        ("Temporalio.Mcp", "0.1.0"): {"listed": True, "published": published(1)},
    })
    failures = check_dotnet_deps.check(plugin, leaves)
    assert len(failures) == 1 and "Temporalio 1.22.0 is unlisted" in failures[0]


def test_unlisted_third_party_versions_fail_even_when_old(tmp_path: Path) -> None:
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3"})
    leaves = registry(tmp_path, {("xunit.v3", "4.0.3"): {"listed": False, "published": "1900-01-01T00:00:00+00:00"}})
    failures = check_dotnet_deps.check(plugin, leaves)
    assert failures == ["xunit.v3 4.0.3 is unlisted on nuget.org (the yanked analogue); move to a listed version"]


def test_missing_registration_leaf_fails_closed(tmp_path: Path) -> None:
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3", "Temporalio": "1.22.0"})
    leaves = registry(tmp_path, {("xunit.v3", "4.0.3"): {"listed": True, "published": published(30)}})
    failures = check_dotnet_deps.check(plugin, leaves)
    assert failures == ["Temporalio 1.22.0: registry answered HTTP 404; refusing to guess"]


@pytest.mark.parametrize("status", [401, 429, 503])
def test_registry_errors_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "failed", {}, None)

    monkeypatch.setattr(check_dotnet_deps.urllib.request, "urlopen", fail)
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3"})
    failures = check_dotnet_deps.check(plugin)
    assert failures == [f"xunit.v3 4.0.3: registry answered HTTP {status}; refusing to guess"]


def test_gzip_registration_leaves_are_decoded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fetch(request, timeout):
        requests.append((request.full_url, request.get_header("Accept-encoding")))
        return Response(gzip.compress(json.dumps({"listed": True, "published": published(20)}).encode()))

    monkeypatch.setattr(check_dotnet_deps.urllib.request, "urlopen", fetch)
    plugin = plugin_with_lock(tmp_path, {"Xunit.V3": "4.0.3"})
    assert check_dotnet_deps.check(plugin) == []
    assert requests == [("https://api.nuget.org/v3/registration5-gz-semver2/xunit.v3/4.0.3.json", "gzip")]


def test_project_references_and_duplicate_versions_are_fetched_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3"})
    tests = plugin / "tests" / "Temporalio.Fakedotnet.Tests"
    tests.mkdir(parents=True)
    lock = json.loads(dotnet_lock({"xunit.v3": "4.0.3"}))
    lock["dependencies"]["net10.0"]["Temporalio.Fakedotnet"] = {"type": "Project"}
    (tests / "packages.lock.json").write_text(json.dumps(lock))
    calls = []
    real = check_dotnet_deps.registration_leaf
    monkeypatch.setattr(check_dotnet_deps, "registration_leaf", lambda *args: calls.append(args[:2]) or real(*args))
    leaves = registry(tmp_path, {("xunit.v3", "4.0.3"): {"listed": True, "published": published(30)}})
    assert check_dotnet_deps.check(plugin, leaves) == []
    assert calls == [("xunit.v3", "4.0.3")]


@pytest.mark.parametrize("config,message", [
    (NUGET_CONFIG.replace(' minPublishAgeHours="336"', ""), "must set minPublishAgeHours"),
    (NUGET_CONFIG.replace('minPublishAgeHours="336"', 'minPublishAgeHours="0"'), "must set minPublishAgeHours"),
    (NUGET_CONFIG.replace("https://api.nuget.org/v3/index.json", "https://mirror.example/v3/index.json"), "nuget.org source"),
])
def test_nuget_config_must_declare_the_cooldown(tmp_path: Path, config: str, message: str, capsys: pytest.CaptureFixture[str]) -> None:
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3"}, config)
    leaves = registry(tmp_path, {("xunit.v3", "4.0.3"): {"listed": True, "published": published(30)}})
    assert check_dotnet_deps.main(["--plugin-dir", str(plugin), "--registry-dir", str(leaves)]) == 1
    assert message in capsys.readouterr().out


def test_cli_exit_codes_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plugin = plugin_with_lock(tmp_path, {"xunit.v3": "4.0.3", "Temporalio": "1.22.0"})
    leaves = registry(tmp_path, {
        ("xunit.v3", "4.0.3"): {"listed": True, "published": published(30)},
        ("Temporalio", "1.22.0"): {"listed": True, "published": published(1)},
    })
    assert check_dotnet_deps.main(["--plugin-dir", str(plugin), "--registry-dir", str(leaves)]) == 0
    out = capsys.readouterr().out
    assert "cooldown 336 hours; exempt ['temporalio*']; 2 locked versions" in out
    assert "OK: Temporalio 1.22.0 is listed and exempt from the cooldown" in out
    assert out.rstrip().endswith("OK: every locked version is listed and older than the cooldown")
    (plugin / "src" / "Temporalio.Fakedotnet" / "packages.lock.json").write_text(dotnet_lock({"xunit.v3": "4.0.3", "Fresh": "1.0.0"}))
    leaves = registry(tmp_path, {("Fresh", "1.0.0"): {"listed": True, "published": published(2)}})
    assert check_dotnet_deps.main(["--plugin-dir", str(plugin), "--registry-dir", str(leaves)]) == 1
    out = capsys.readouterr().out
    assert "FAIL: 1 locked version(s) violate the dependency cooldown" in out and "Fresh 1.0.0 was published" in out


def test_missing_lock_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "nuget.config").write_text(NUGET_CONFIG)
    assert check_dotnet_deps.main(["--plugin-dir", str(plugin), "--registry-dir", str(tmp_path)]) == 1
    assert "no committed packages.lock.json" in capsys.readouterr().out
