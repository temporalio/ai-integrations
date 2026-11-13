"""smoke_dotnet.py renders an isolated consumer and proves the restored package is the tested one."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
import smoke_dotnet

COORDINATE = "Temporalio.Fakedotnet"
VERSION = "0.0.1-rc.1"


def nupkg_bytes(*, dll: bytes = b"MZ-dll", signature: bytes | None = None, stored: bool = True,
                signature_first: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        def sign() -> None:
            info = zipfile.ZipInfo(".signature.p7s")
            info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            archive.writestr(info, signature or b"")

        if signature is not None and signature_first:
            sign()
        archive.writestr(f"{COORDINATE}.nuspec", "<package />")
        archive.writestr(f"lib/net10.0/{COORDINATE}.dll", dll)
        if signature is not None and not signature_first:
            sign()
    return buffer.getvalue()


def plugin(tmp_path: Path) -> Path:
    directory = tmp_path / "dotnet" / "fakedotnet"
    (directory / "dist").mkdir(parents=True)
    (directory / "plugin.toml").write_text(
        f'[plugin]\ncoordinate = "{COORDINATE}"\n[ci]\nruntime-versions = ["10.0"]\n'
        f'[smoke]\nimports = ["{COORDINATE}.FakedotnetPlugin", "{COORDINATE}.Other"]\n'
    )
    (directory / "dist" / f"{COORDINATE}.{VERSION}.nupkg").write_bytes(nupkg_bytes())
    return directory


class FakeDotnet:
    """Stands in for `dotnet run`: records the rendered consumer and restores the given package bytes."""

    def __init__(self, restored: bytes | None, returncodes: tuple[int, ...] = (0,)) -> None:
        self.restored = restored
        self.returncodes = returncodes
        self.calls: list[list[str]] = []
        self.rendered: dict[str, str] = {}
        self.env: dict[str, str] = {}

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str], **_kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append(command)
        self.env = env
        consumer = Path(cwd)
        self.rendered = {name: (consumer / name).read_text()
                         for name in ("Consumer.csproj", "Program.cs", "nuget.config", "global.json")}
        code = self.returncodes[min(len(self.calls), len(self.returncodes)) - 1]
        if code == 0 and self.restored is not None:
            target = (Path(env["NUGET_PACKAGES"]) / COORDINATE.lower() / VERSION.lower()
                      / f"{COORDINATE.lower()}.{VERSION.lower()}.nupkg")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.restored)
        return subprocess.CompletedProcess(command, code)


def test_local_feed_consumer_is_isolated_and_restores_the_tested_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    directory = plugin(tmp_path)
    fake = FakeDotnet(nupkg_bytes())
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", fake)
    monkeypatch.setenv("RestoreLockedMode", "true")
    smoke_dotnet.smoke(directory, VERSION)
    assert fake.calls == [["dotnet", "run", "--project", "Consumer.csproj", "-c", "Release"]]
    assert "RestoreLockedMode" not in fake.env
    assert fake.env["NUGET_PACKAGES"] != fake.env["NUGET_HTTP_CACHE_PATH"]
    assert all(fake.env[key] for key in ("NUGET_PACKAGES", "NUGET_HTTP_CACHE_PATH", "NUGET_PLUGINS_CACHE_PATH"))
    csproj = fake.rendered["Consumer.csproj"]
    assert f'<PackageReference Include="{COORDINATE}" Version="[{VERSION}]" />' in csproj
    assert "<TargetFramework>net10.0</TargetFramework>" in csproj
    assert "<ImportDirectoryBuildProps>false</ImportDirectoryBuildProps>" in csproj
    # Advisories on the plugin's closure are warnings in the consumer, as in dotnet.props.
    assert "<WarningsNotAsErrors>NU1900;NU1901;NU1902;NU1903;NU1904</WarningsNotAsErrors>" in csproj
    # The consumer builds with the plugin's SDK channel, not the newest SDK on the machine.
    assert json.loads(fake.rendered["global.json"]) == {
        "sdk": {"version": "10.0.100", "rollForward": "latestFeature", "allowPrerelease": False},
    }
    program = fake.rendered["Program.cs"]
    assert f'"{COORDINATE}.FakedotnetPlugin", "{COORDINATE}.Other"' in program
    assert f'var version = "{VERSION}";' in program
    # The DLL match searches every lib/<tfm>/ folder, so a package whose best asset is not the consumer TFM passes.
    assert 'Directory.GetDirectories(Path.Combine(packageRoot, "lib"))' in program and "net10.0" not in program
    config = fake.rendered["nuget.config"]
    assert f'<add key="tested" value="{directory / "dist"}" />' in config
    assert '<add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />' in config
    assert f'<packageSource key="tested">\n      <package pattern="{COORDINATE}" />' in config
    assert '<packageSource key="nuget.org">\n      <package pattern="*" />' in config
    assert capsys.readouterr().out.strip().endswith(f"and loaded 2 type(s)")


def test_local_feed_restore_must_be_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = plugin(tmp_path)
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", FakeDotnet(nupkg_bytes(dll=b"MZ-other")))
    with pytest.raises(ValueError, match="differs from the tested dist/ file"):
        smoke_dotnet.smoke(directory, VERSION)


def test_registry_restore_may_differ_only_by_the_trailing_repository_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = plugin(tmp_path)
    fake = FakeDotnet(nupkg_bytes(signature=b"nuget.org"))
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", fake)
    smoke_dotnet.smoke(directory, VERSION, source="https://api.nuget.org/v3/index.json")
    config = fake.rendered["nuget.config"]
    assert '<add key="under-test" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />' in config
    assert 'key="tested"' not in config
    assert '<packageSource key="under-test">\n      <package pattern="*" />' in config


@pytest.mark.parametrize("restored,message", [
    pytest.param(nupkg_bytes(dll=b"MZ-other", signature=b"sig"), "different package entries", id="changed-assembly"),
    pytest.param(nupkg_bytes(signature=b"sig", stored=False), "not the trailing stored repository signature", id="compressed-signature"),
    pytest.param(nupkg_bytes(signature=b"sig", signature_first=True), "not the trailing stored repository signature", id="signature-first"),
    pytest.param(None, "did not restore", id="not-restored"),
])
def test_registry_restore_rejects_changed_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored: bytes | None, message: str) -> None:
    directory = plugin(tmp_path)
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", FakeDotnet(restored))
    with pytest.raises(ValueError, match=message):
        smoke_dotnet.smoke(directory, VERSION, source="https://api.nuget.org/v3/index.json")


def test_unsigned_entries_strip_only_a_trailing_stored_signature() -> None:
    plain = smoke_dotnet.unsigned_entries(nupkg_bytes())
    assert [name for name, *_ in plain] == [f"{COORDINATE}.nuspec", f"lib/net10.0/{COORDINATE}.dll"]
    assert smoke_dotnet.unsigned_entries(nupkg_bytes(signature=b"sig")) == plain
    assert smoke_dotnet.unsigned_entries(nupkg_bytes(dll=b"MZ-other", signature=b"sig")) != plain
    for payload in (nupkg_bytes(signature=b"sig", stored=False), nupkg_bytes(signature=b"sig", signature_first=True)):
        with pytest.raises(ValueError, match="trailing stored repository signature"):
            smoke_dotnet.unsigned_entries(payload)


def test_feed_propagation_is_retried_from_a_clean_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    directory = plugin(tmp_path)
    fake = FakeDotnet(nupkg_bytes(signature=b"sig"), returncodes=(1, 0))
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", fake)
    monkeypatch.setattr(smoke_dotnet.time, "sleep", lambda seconds: None)
    smoke_dotnet.smoke(directory, VERSION, source="https://api.nuget.org/v3/index.json", attempts=2, delay=0)
    assert len(fake.calls) == 2
    assert "::warning::clean consumer failed (attempt 1/2)" in capsys.readouterr().out
    fake = FakeDotnet(nupkg_bytes(), returncodes=(1,))
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", fake)
    with pytest.raises(ValueError, match="after 1 attempt"):
        smoke_dotnet.smoke(directory, VERSION)


def test_missing_tested_package_and_imports_are_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = plugin(tmp_path)
    monkeypatch.setattr(smoke_dotnet.subprocess, "run", lambda *args, **kwargs: pytest.fail("no consumer without a package"))
    with pytest.raises(ValueError, match="is missing"):
        smoke_dotnet.smoke(directory, "9.9.9")
    (directory / "plugin.toml").write_text(f'[plugin]\ncoordinate = "{COORDINATE}"\n[ci]\nruntime-versions = ["10.0"]\n')
    with pytest.raises(ValueError, match="smoke. imports"):
        smoke_dotnet.smoke(directory, VERSION)
