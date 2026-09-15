# Contributing

Read [`AGENTS.md`](AGENTS.md) first; it is the normative guide and this page only summarizes the
workflow.

## Set up

- Python plugins: install [uv](https://docs.astral.sh/uv/) and GNU make (`brew install make` on
  macOS if needed; `choco install make` on Windows, or use WSL).
- `cd python/<plugin> && make sync && make lint && make test`.

## Pull requests

- Ordinary PRs are squash-merged. Issue and PR templates are the organization defaults from
  `temporalio/.github`.
- PRs that import or re-sync history from an SDK repository are labelled `history-import` and are
  merged with **"Create a merge commit"**, never squash or rebase. Adaptation changes go in
  separate commits after the merge commit and never touch imported files.
- CI runs only the plugins whose files changed (`.github/workflows/ci.yml`); `ci-status`,
  `Check for CODEOWNERS` and `opengrep/scan` are the required checks, plus one code-owner
  approval. No secrets are needed: provider behavior uses deterministic local test doubles.
- During the SDK cutover transition, bug fixes for a migrated plugin land in the SDK repository
  first and reach this repository through a re-sync (AGENTS.md, "Transition rules").

## Adding a Python plugin

```bash
python3 scripts/new_python_plugin.py <name> --description "..." --maturity experimental
```

The scaffolder copies `python/_template`, prints the remaining manual steps, and no workflow
edits are needed: CI discovers plugins from `plugin.toml`. To migrate an existing plugin with its
history, follow [`scripts/migrate/README.md`](scripts/migrate/README.md).

## Contributor License Agreement

All contributors must complete the Temporal Contributor License Agreement (CLA)
before changes can be merged. A link to the CLA will be posted in the pull request.

## Releasing

Release tags are authoritative: keep the committed Python version at `0.0.0`, dry-run the intended
`python/<name>/v<version>` tag from `main`, then create the protected tag without a version-only PR.
Follow the runbook in AGENTS.md ("Releases"). Release notes are generated from commit messages, so
write commit subjects you would want users to read.
