# Migrating a plugin with its history

`extract-sdk-python.sh` rewrites a fresh clone of `temporalio/sdk-python` so that its `main`
contains only one plugin's files, already at their paths in this repository, with every commit's
author, date and message intact. The result is merged into this repository with an
unrelated-histories merge commit. `git log --follow`, `git blame` and `git shortlog` then work on
the imported files exactly as they did upstream.

Imported history means commits reachable from the upstream repository's default branch. Never use
this procedure, the `history-import` label or `IMPORTS.md` for work that exists only on an unmerged
or closed upstream PR, feature branch or fork; port that work as ordinary local commits instead.

## Procedure

```bash
git checkout -b import/python-openai-agents main
PLUGIN=openai_agents scripts/migrate/extract-sdk-python.sh     # prints SRC_SHA, WORK and the merge commands
git remote add sdk-python-filtered "$WORK/src"
git fetch --no-tags sdk-python-filtered main
git merge --allow-unrelated-histories --no-ff -m "Import temporalio.contrib.openai_agents from temporalio/sdk-python@<SRC_SHA> (history preserved; git-filter-repo 2.47.0)" sdk-python-filtered/main
git remote remove sdk-python-filtered
```

Then add the files the import does not bring, as **separate commits on top of the merge**
(never amend the merge and never edit an imported file in the same PR):

```bash
python3 scripts/new_python_plugin.py openai_agents --existing --maturity ga \
  --description "Temporal integration for the OpenAI Agents SDK"
```

Record the import in `IMPORTS.md`, open a PR labelled `history-import`, and merge it with
**"Create a merge commit"**. Squash or rebase merging destroys the imported history.

## What the script does

- Clones with `--no-tags --single-branch` so no upstream release tags are imported.
- Runs `git-filter-repo` pinned to 2.47.0 (`uvx`), with:
  - four `--path` filters covering every location the plugin's files ever had upstream
    (`temporalio/contrib/openai_agents/`, `tests/contrib/openai_agents/`,
    `tests/contrib/test_openai.py`, `tests/contrib/research_agents/`);
  - `--path-rename` rules into `python/openai_agents/...`; the package README is renamed to
    the plugin root before the directory rename, and all filters precede all renames because
    filter-repo evaluates later filters against already-renamed paths;
  - `--replace-message` rewriting `#123` to `temporalio/sdk-python#123` so issue and PR links
    keep pointing at the SDK repository;
  - `--preserve-commit-hashes` so SHAs mentioned in messages keep referring to sdk-python;
  - `--prune-empty always` so upstream's originally-empty commits (the 2022 "Initial commit",
    a dependabot bump) do not survive as unrelated root commits;
  - a commit callback appending `Migrated-From: temporalio/sdk-python@<original sha>` as a
    trailer, placed contiguously with existing trailers such as `Co-authored-by:`;
  - `--force`, needed only because the optional `SRC_REF` pin fails filter-repo's fresh-clone
    check; the clone is a throwaway directory.

## Re-sync (bringing new upstream commits)

Use this procedure only while the plugin's `plugin.toml` names `sdk-python` as its `upstream`.
Removing that field makes this repository the source of truth; do not re-sync the plugin after
that point. While the field is present, pick up upstream changes as follows:

1. Run the script unchanged, fetch, and `git merge sdk-python-filtered/main` on a new branch.
   Only commits newer than the previous import arrive because the rewrite is byte-identical.
2. Resolve conflicts only where adaptation commits or documented local-only divergences touched
   imported files. For the `openai_agents` MCP v2 adapter files listed in AGENTS.md, "Transition
   rules", preserve both the new upstream changes and the local adapter.
3. Diff the vendored test scaffolding against upstream and port relevant changes by hand:
   `tests/conftest.py`, `tests/__init__.py`, `tests/helpers/__init__.py`, `tests/helpers/nexus.py`.
4. If upstream added provider-network tests, add deterministic local coverage without credentials.
5. Append a row to `IMPORTS.md`; open a `history-import` PR; merge with a merge commit.

## Determinism breakers

The rewrite stays byte-identical only if none of these change: the `--path`/`--path-rename`
arguments, `replace-message.txt`, the commit callback, `--prune-empty`/`--preserve-commit-hashes`,
the pinned git-filter-repo version, and upstream default-branch history itself. `SRC_REF` must be
both reachable from the upstream default branch and a descendant of the previous import. Stop if
either condition is false; a PR or feature-branch ref is not a history-import input and its changes
must be ported as ordinary local commits. If a rewrite rule changes or upstream rewrites its default
branch, the plugin needs a one-time full re-import PR (delete `python/<plugin>`, import again,
re-apply adaptation and documented local-only commits) and a note in `IMPORTS.md`.

## Adding another plugin

Find every historical path first:

```bash
cd /path/to/sdk-python
git log --all --diff-filter=R --name-status --format='--%h %ad' -- temporalio/contrib/<name> tests/contrib/<name>
git log --all --name-only --format= -- temporalio/contrib/<name> tests/contrib/<name> | sort -u
git log --all --oneline --follow -- temporalio/contrib/<name>/__init__.py
```

Add a `case` entry to `extract-sdk-python.sh` with every path found (filters first, renames
second, README rename before directory rename). Missing a historical path cannot be fixed later
without rewriting every imported SHA.
