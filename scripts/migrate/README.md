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
python3 scripts/new_python_plugin.py openai_agents --existing --maturity generally-available \
  --upstream temporalio/sdk-python:temporalio/contrib/openai_agents \
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

## Remaining Python plugin imports (2026-10-05)

Five initial imports are pinned to sdk-python `main` at
`6adc0d84290a79952dee3ef02c36f6ed9334874a`. Google ADK is pinned to
`d61b3f3ad3dcfd9187fa6012b5d77de2d5b4cb9f`, the parent of #1854. That merged PR
requires the named `workflow.new_random(name)` API, absent from the latest published SDK
(1.34.0); port that fix after an SDK release contains it. Both SHAs are reachable from
upstream `main`. The history-import merges preserve source and plugin-specific tests
unchanged. A separate local cutover commit moves all six to `temporalio.<folder_name>`,
including `temporalio.google_adk` and `temporalio.strands_agents`, updates imports and
examples, and flattens the test trees. This API cutover was explicitly requested with
the initial import. Active `upstream` metadata is removed because this repository now
owns the code. Path archaeology found no earlier locations outside their current package and test
trees. Expected filtered history counts are:

| Folder | Upstream module | Commits | Identities |
|---|---|---|---|
| deepagents | deepagents | 6 | 4 |
| google_adk | google_adk_agents | 22 | 11 |
| google_genai | google_genai | 8 | 4 |
| langgraph | langgraph | 7 | 2 |
| langsmith | langsmith | 12 | 5 |
| strands_agents | strands | 6 | 2 |

The initial live GitHub audit inspected all 64 open issues and the changed files of all 16 open
PRs. None then touched these plugin source or test trees. The related open PRs
[sdk-python#1805](https://github.com/temporalio/sdk-python/pull/1805),
[sdk-python#1811](https://github.com/temporalio/sdk-python/pull/1811), and
[sdk-python#1891](https://github.com/temporalio/sdk-python/pull/1891) concern SDK-owned Workflow
Streams or OpenTelemetry and are not imported. Closed DeepAgents alternatives #1902 and #1904
are unmerged; the merged fix #1901 is included instead. Recent merged plugin fixes, including
#1865, #1873, #1887, #1899 and #1908, are reachable from the pinned SHAs. #1854 is merged
on `main` but deferred for the SDK compatibility reason above.

Each plugin has its own canonical provenance support and local server fixtures. Shared upstream
helpers are copied only where used: `new_worker` for Google GenAI and LangSmith, Nexus/trace
helpers for LangSmith, and the span formatter and provider-reset fixtures for Google ADK.
DeepAgents sets the low history-count threshold required by its server-suggested continue-as-new
test. Its README tests point at the standalone distribution README. Provider tests use
local models, mock transports or in-process MCP servers. Final releases are enabled
after the API cutover; the committed development version remains `0.0.0`.

Google GenAI and Strands README links resolve SDK-owned dependencies to the pinned SDK
tree. The conventions check validates the README actually published.

The shared make targets use the committed lockfile for regular checks and re-lock to
the newest allowed dependencies on nightly runs. Lint and tests use `uv run --locked`
to preserve the selected versions. Dependency floors are LangGraph 1.2.0
(the imported adapter uses `Runtime.execution_info`), Google ADK OpenTelemetry
1.40.0 (its MCP semantic-convention imports), and pytest-asyncio 0.21.2
(the compatible pytest 9 fixture implementation). Google ADK test setup preloads OpenAI
types before workflow tasks to avoid cold-import deadlock detection on Python 3.10.

At the final audit, 63 issues and 16 PRs were open. A new draft,
[sdk-python#1923](https://github.com/temporalio/sdk-python/pull/1923), removes the bundled
implementations and forwards to the planned final APIs. It depends on publishing these
packages after cutover and is not an import input. The newer SDK main commit
`3f29fd7935f9a03fbe86f24624dfe6458872098e` does not change the five plugins pinned to
`6adc0d84290a79952dee3ef02c36f6ed9334874a`.

## Java Spring AI extraction

Run `scripts/migrate/extract-sdk-java.sh` with a default-branch-reachable `SRC_REF`.
The frozen mapping covers `temporal-spring-ai/` and `contrib/temporal-spring-ai/`.
It retains the historical destination `java/temporal-spring-ai/`; a separate
adaptation commit renames that directory to `java/spring-ai/` and the published
coordinate to `io.temporal:spring-ai`. Do not change the frozen extractor.
The frozen `replace-message-java.txt` rule qualifies issue and PR references as
`temporalio/sdk-java#NNN` so imported messages retain their upstream links.
At be01e60acc1e2ccfb20e783a9770bad745ed85c1 there are 12 rewritten commits
(13 upstream path commits; the directory-only rename becomes empty), one author
identity, 33 main Java files, one resource, and 14 test files. Source and tests
are unchanged; the upstream README and build are archived under `_upstream/`.
Merge the extracted branch with unrelated histories allowed, record the merge in
IMPORTS.md, and label only the import PR `history-import`. The GitHub PR must use
Create a merge commit. Gradle and MIT licensing adaptations are separate commits.
