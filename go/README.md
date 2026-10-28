# Go integrations

Plugins that connect Go AI frameworks and SDKs to Temporal. Each plugin lives
in `go/<name>/` as an independent Go module with its own source, tests,
`go.mod`, `go.sum`, `plugin.toml`, `Makefile`, `README.md` and `LICENSE`.

Each plugin's README describes its API and installation. Its `go.mod` sets
the minimum Go version, and `plugin.toml` records its integration metadata.

Run the shared make targets from the plugin directory, replacing `<name>`
with the plugin's folder name:

```bash
cd go/<name>
make sync
make lint
make test
make build
```

Plugin Makefiles include [`_shared/go.mk`](_shared/go.mk). `make help` lists
the targets; dependency checks use the committed module versions.

CI reads each plugin's Go versions from `plugin.toml`, testing the minimum
and maximum on Ubuntu and the maximum on macOS and Windows. The Ubuntu
maximum-version job also runs race detection. Nightly checks use
`make sync-latest` to upgrade compatible dependencies in the CI checkout;
failures are tracked in a `nightly` issue for the plugin.

Tests must not require provider credentials. Use deterministic local models,
mock transports or in-process servers. Tests that need Temporal use
`testClient(t)` to lazily share a local dev server and its default client within
the package's test process. Initial startup may download the Temporal CLI and
gets up to two 90-second attempts; cancellation and readiness failures propagate
immediately. Each test owns its workers and uses unique task queues.
`startTestWorkflow` registers cleanup for unfinished workflows, and
`TestMain` closes the shared client and stops the server after all tests finish.

Repository conventions are in
[`AGENTS.md`](../AGENTS.md).
