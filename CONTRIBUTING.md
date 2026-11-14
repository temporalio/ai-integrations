# Contributing

Read [`AGENTS.md`](AGENTS.md) first; it is the normative guide and this page only summarizes the
workflow.

## Set up

- Python plugins: install [uv](https://docs.astral.sh/uv/) and GNU make (`brew install make` on
  macOS if needed; `choco install make` on Windows, or use WSL).
- `cd python/<plugin> && make sync && make lint && make test`.
- .NET plugins: install the .NET 10 SDK and GNU make; `cd dotnet/<plugin> && make sync && make lint && make test`.

## Pull requests

- PRs are squash-merged. Issue and PR templates are the organization defaults from
  `temporalio/.github`.
- CI runs only the plugins whose files changed (`.github/workflows/ci.yml`); `ci-status`,
  `Check for CODEOWNERS` and `opengrep/scan` are the required checks, plus one code-owner
  approval. No secrets are needed: provider behavior uses deterministic local test doubles.
- Plugins match their framework's behavior (see "Plugin behavior" in `AGENTS.md`). Say in the PR
  when the framework's docs contradict its code, and when a change can alter recorded histories.

## Adding a Python plugin

```bash
uv run --project scripts --locked python scripts/new_python_plugin.py <name> --description "..." --maturity pre-release
```

The scaffolder copies `python/_template`, prints the remaining manual steps, and no workflow
edits are needed: CI discovers plugins from `plugin.toml`.

## Adding a .NET plugin

There is no scaffolder yet. Create `dotnet/<name>/` with the layout in [`dotnet/README.md`](dotnet/README.md)
and the ".NET conventions" in `AGENTS.md`, run `make lock` to write the `packages.lock.json` files, and
commit them. CI discovers the plugin from `Directory.Build.props`;
`uv run --project scripts --locked python scripts/ci/check_conventions.py` lists anything still missing.

## Contributor License Agreement

All contributors must complete the Temporal Contributor License Agreement (CLA)
before changes can be merged. A link to the CLA will be posted in the pull request.
