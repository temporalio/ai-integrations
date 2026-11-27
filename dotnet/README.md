# .NET integrations

Plugins that connect .NET AI frameworks and SDKs to Temporal. Each plugin lives in `dotnet/<name>/`
as an independent NuGet package with its own `src/<Package>/<Package>.csproj`,
`tests/<Package>.Tests/`, per-project `packages.lock.json`, `plugin.toml`, `Directory.Build.props`,
`global.json`, `nuget.config`, `.editorconfig`, `<Package>.slnx`, `Makefile`, `README.md` and `LICENSE`.

Each plugin's README describes its API and installation. Its csproj sets the target framework and
dependencies, `global.json` pins the .NET SDK channel, and `plugin.toml` records its integration
metadata. Package IDs are `Temporalio.Extensions.<Name>` (the SDK's pattern for libraries that build
on `Temporalio`) and the root namespace is the package ID.

Install the .NET 10 SDK, the .NET 8 runtime (the test suite runs on both; `make test
TEST_FRAMEWORK=net10.0` runs one) and GNU make, then run the shared make targets from the plugin
directory:

```bash
cd dotnet/<name>
make sync
make lint
make test
make build
```

Plugin Makefiles set `PACKAGE` and include [`_shared/dotnet.mk`](_shared/dotnet.mk). `make help`
lists the targets. `make sync` restores the committed locks; every other target restores in locked
mode too, so after a csproj edit without `make lock`, `make build` and `make test` fail with NU1004
and `make lint` fails to restore. `make lint` runs
`dotnet format --verify-no-changes` against the solution; analyzers run as part of the build with
warnings as errors. `make test` builds in Release and runs the suite under Microsoft.Testing.Platform
(`TestResults/*.trx`). `make build` packs the tested build into `dist/` (`.nupkg` and `.snupkg`).

CI reads the runtimes from `plugin.toml` `[ci] runtime-versions` (oldest first; the last is the SDK
channel `global.json` pins) and tests the oldest on Ubuntu and the newest on Ubuntu, macOS and
Windows. The newest runtime's Ubuntu cell also checks the dependency cooldown, packs, verifies the package
contents and restores it in a clean consumer. Nightly checks use `make sync-latest` to update every
direct dependency in the CI checkout; failures are tracked in a `nightly` issue keyed `dotnet/<name>`.

Tests must not require provider credentials. Use deterministic fakes (an `IChatClient` that replays
scripted responses, in-process MCP servers) and the Temporal dev server that
`make test` downloads and caches on first run.

Repository conventions are in [`AGENTS.md`](../AGENTS.md).
