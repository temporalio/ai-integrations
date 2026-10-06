# Go integrations

[`googleadk`](googleadk) contains the Google ADK integration imported with its
history from `temporalio/sdk-go`. See [`scripts/migrate/IMPORTS.md`](../scripts/migrate/IMPORTS.md)
for the pinned upstream commit and import merge.

```bash
cd go/googleadk
make sync
make lint
make test
make build
```

Go 1.26.5 or later is required. Tests use local models, mock transports and
in-process MCP servers; integration tests start a local Temporal dev server
and download the Temporal CLI on first use. Provider credentials are not required.

This is a history snapshot, with `sdk-go` still the published upstream. The
imported module path remains `go.temporal.io/sdk/contrib/googleadk`, and final
releases here are disabled. Before publishing from this repository, resolve
the Go vanity-path hosting decision in [`AGENTS.md`](../AGENTS.md).

The imported source, tests, dependency files and README remain unchanged.
The upstream changelog is preserved in Git history; this repository derives
release notes from commits and does not maintain a changelog. The repository
adds metadata, make targets and a copy of the root license in a separate commit.
Go CI and release workflows must be added before the publishing cutover.
