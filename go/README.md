# Go integrations

Plugins that connect Go AI frameworks and SDKs to Temporal. Each plugin lives
in `go/<name>/` as an independent Go module with its own source, tests,
`go.mod`, `go.sum`, `plugin.toml`, `Makefile`, `README.md` and `LICENSE`.

Each plugin's README describes its API and installation. Its `go.mod` sets
the minimum Go version, and `plugin.toml` records its maturity, upstream
and release status.

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

Tests must not require provider credentials. Use deterministic local models,
mock transports or in-process servers. Tests that need Temporal start a local
dev server and may download the Temporal CLI on first use.

Repository conventions and Go publishing requirements are in
[`AGENTS.md`](../AGENTS.md). History imports and re-syncs follow
[`scripts/migrate/README.md`](../scripts/migrate/README.md), with provenance
recorded in [`IMPORTS.md`](../scripts/migrate/IMPORTS.md).
