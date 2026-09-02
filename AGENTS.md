# AGENTS.md — temporalio/ai-integrations

Normative guide for humans and coding agents working in this repository. Other documents
(`README.md`, `CONTRIBUTING.md`, `python/README.md`, `scripts/migrate/README.md`) are short and
link here. Design source: the
[ai-integrations design doc](https://app.notion.com/p/temporalio/ai-integrations-3ca8fc5677388120b5c3cf5567f0327c).

## Do not

1. Do not create `__init__.py` in `python/<plugin>/src/temporalio/` or `src/temporalio/contrib/`. The SDK wheel owns those packages; `scripts/ci/check_wheel.py` fails the build if they appear.
2. Do not edit imported files (`src/...`, `tests/contrib/<plugin>/...`) while the plugin's `plugin.toml` still names an `upstream`. Fix upstream, then re-sync.
3. Do not squash or rebase a PR labelled `history-import`. Merge it with "Create a merge commit".
4. Do not install with plain `uv sync` or run tests with a bare `uv run`. Use `make sync`, `make test`, and the other targets; they export `UV_NO_EDITABLE=1`.
5. Do not add a workflow file, job, or secret for one plugin. Plugin variation lives in `plugin.toml`, `pyproject.toml`, and the shared make targets.
6. Do not add `exclude-newer` without `exclude-newer-package = { temporalio = false }`, and do not replace `--locked` with `--frozen`.
7. Do not add `[tool.uv.sources]` path or workspace entries between plugins. Cross-plugin dependencies use published registry coordinates only.
8. Do not bump a version outside a release PR, and never move or delete a tag. A failed pre-release gets the next `rcN`.
9. Do not write `CHANGELOG.md` files. Release notes are generated from commit messages.
10. Do not modify `samples-python`, `documentation`, `sdk-python`, `auto-aie`, or `cicd-terraform` from this repository. Those edits belong to the cutover checklist below.

## Purpose and layout

AI plugins for Temporal SDKs, factored out of the SDK repositories so each plugin is
independently installable, dependency-isolated, tested, versioned and released. Layout is
`<language>/<integration>/`; directories under a language root that start with `_` are shared
resources (`python/_shared/`, `python/_template/`) and are ignored by CI discovery.

| Folder | Coordinate | First version here | Maturity | Root API |
|---|---|---|---|---|
| `python/mcp` | `temporalio-mcp` | 0.1.0 | experimental | `temporalio.contrib.mcp` |
| `python/deepagents` | `temporalio-deepagents` | 0.1.0 | experimental | `temporalio.contrib.deepagents` |
| `python/google_adk` | `temporalio-google-adk` | 0.1.0 | preview | `temporalio.contrib.google_adk` |
| `python/google_genai` | `temporalio-google-genai` | 0.1.0 | experimental | `temporalio.contrib.google_genai` |
| `python/langgraph` | `temporalio-langgraph` | 0.1.0 | experimental | `temporalio.contrib.langgraph` |
| `python/langsmith` | `temporalio-langsmith` | 0.1.0 | experimental | `temporalio.contrib.langsmith` |
| `python/openai_agents` | `temporalio-openai-agents` | 1.0.0 | ga | `temporalio.contrib.openai_agents` |
| `python/strands_agents` | `temporalio-strands-agents` | 0.1.0 | experimental | `temporalio.contrib.strands_agents` |
| `typescript/vercel-ai-sdk` | `@temporalio/vercel-ai-sdk` | 1.0.0 | ga | `@temporalio/vercel-ai-sdk` |
| `typescript/google-adk` | `@temporalio/google-adk` | 0.1.0 | preview | `@temporalio/google-adk` |
| `typescript/langsmith` | `@temporalio/langsmith` | continues (1.24.0 next) | experimental | `@temporalio/langsmith` |
| `typescript/openai-agents` | `@temporalio/openai-agents` | continues (1.24.0 next) | ga | `@temporalio/openai-agents` |
| `typescript/strands-agents` | `@temporalio/strands-agents` | continues (1.24.0 next) | experimental | `@temporalio/strands-agents` |
| `java/temporal-spring-ai` | `io.temporal:temporal-spring-ai` | continues (1.39.0 next) | preview | `io.temporal.springai` |
| `go/googleadk` | `go.temporal.io/sdk/contrib/googleadk` | continues (v0.3.0 next) | preview | `googleadk` |

"First version here" values are informational; the registry is the source of truth for the
version policy (below). `python/mcp` does not exist upstream yet (sdk-python PR #1793). The Go row
has an unresolved problem: a module served by the static vanity site cannot live in a monorepo
subdirectory under an unchanged import path; decide (split mirror repo, new import path, or
staying in sdk-go) before that migration.

Naming derivation, enforced by `scripts/ci/check_conventions.py`: folder name = `plugin.toml`
`name`; Python coordinate = `temporalio-` + name with `_` replaced by `-`; Python root API =
`temporalio.contrib.<name>`; release tag = `<language>/<name>/v<version>`. Folders never end in
`-plugin` or `_plugin`.

Maturity mapping (`plugin.toml` `maturity` and the Python classifier must agree): `ga` =
`Development Status :: 5 - Production/Stable`; `preview` = `4 - Beta`; `experimental` = `3 - Alpha`.

## Repository invariants

- Each plugin owns its manifest and lockfile (`pyproject.toml` + `uv.lock`); no lockfile at a language root, no root Python project. `scripts/` is a separate tooling project, not a root project.
- Each plugin carries `plugin.toml` (metadata only: name, language, coordinate, registry, root API, maturity, status, docs, upstream, `[release] allow-final`, `[ci] runtime-versions`, `[smoke] imports`). It holds no secrets and no owners; `.github/CODEOWNERS` is `* @temporalio/ai-sdk`.
- One `LICENSE` at the root and nothing committed in plugin directories: `make` copies the root file into the plugin as a gitignored `LICENSE` before any uv command that may build (uv_build fails if a declared license file is missing), `pyproject.toml` declares `license = "MIT"` and `license-files = ["LICENSE"]`, and `check_wheel.py` verifies the packaged text equals the root file. A plain `uv sync` in a fresh clone fails until `make sync` has run once.
- The manifest version is static and is the only version source. A tag is valid only if `uv version --short` in the plugin directory equals the tag's version.
- No changelog files. `scripts/release/release_tool.py release-notes` derives notes from commit subjects touching the plugin directory since its previous tag.
- Every GitHub Action is pinned to a full commit SHA with a `# vN` comment (org opengrep rule).
- Dependabot mirrors sdk-python: security advisories only (`open-pull-requests-limit: 0`) because its uv support is not yet mature enough for routine bumps; the nightly newest/lowest lanes are the dependency-drift signal.

## Python conventions

- Layout: `python/<name>/{pyproject.toml, uv.lock, plugin.toml, Makefile, README.md, LICENSE -> ../../LICENSE, src/temporalio/contrib/<name>/, tests/}`. Migrated plugins keep the upstream test tree (`tests/contrib/<name>/...`, `tests/helpers/`, `tests/conftest.py`) so re-syncs never conflict; flatten only after cutover.
- Build backend `uv_build` with `module-name = "temporalio.contrib.<name>"`. `py.typed` ships in the leaf package (redundant with the SDK's marker, kept on purpose).
- Installs are non-editable. `temporalio` is a regular package owned by the SDK wheel, so an editable install of a plugin makes `temporalio.contrib.<name>` resolve to whatever the SDK ships (silently wrong before cutover, `ImportError` after). `python/_shared/python.mk` exports `UV_NO_EDITABLE=1`; `[tool.uv] cache-keys` includes `src/**/*` so edits trigger a rebuild; `link-mode = "copy"` keeps overwrites deterministic during the transition.
- Provenance guard (`tests/helpers/provenance.py`, mirrored by `scripts/ci/smoke.py`) runs at every pytest session start and fails loudly if the install is editable, any file differs from the distribution's RECORD, files under the package directory are not owned by the distribution, or another distribution ships the same paths. While `plugin.toml` `[release] allow-final = false`, the SDK's overlap (`temporalio<=1.32` ships `temporalio/contrib/openai_agents/*`) is tolerated with a warning. `tests/test_installed_matches_source.py` additionally byte-compares the installed package with `src/`.
- Dependency cooldown: `exclude-newer = "2 weeks"` (org supply-chain policy) with `exclude-newer-package = { temporalio = false }` so a new SDK release is adoptable the day it ships. Declare exactly what the package imports at module level; `openinference` and similar lazy imports go in an extra.
- Tooling: ruff, pyright, basedpyright, mypy (`mypy_path = "src"`, `explicit_package_bases`), pydocstyle (google), pytest + xdist (`-n auto --dist=worksteal`; the `os._exit(0)` hook is xdist-aware). All invoked through `make` targets; see `make help`.
- Tests self-provision the Temporal dev server (`WorkflowEnvironment.start_local`, version pinned in `tests/__init__.py`) with its default configuration; add a `--dynamic-config-value` flag in the template conftest only when a test needs a server feature that is off by default. Upstream MCP tests currently spawn `npx`, so Node must be present (it is on GitHub-hosted runners) until those tests are rewritten upstream.
- Offline tests: `tests/conftest.py` marks every test with pytest-recording's `vcr` marker; HTTP traffic is replayed from `tests/contrib/<name>/cassettes/<module>/<test>.yaml` with body-inclusive matching and `allow_playback_repeats=True`. Policy: no real provider key anywhere, not even for recording; upstream tests that call a provider API are skipped through `plugin.toml` `[offline] skips` (function name or parametrized id, with a reason) while their offline twins run, and cassette recording (`make record`, local only) stays available as an opt-in for real-response regression coverage. Plugin-specific offline settings are data in `plugin.toml` `[offline]`, so the conftest stays identical across plugins: `dummy-env` holds placeholder values (never real secrets, e.g. `OPENAI_API_KEY = "sk-cassette-replay"`) exported during replay so upstream skip guards do not skip. For OpenAI Agents SDK plugins the conftest removes the SDK's default backend trace exporter with `set_trace_processors([])` (do not use `OPENAI_AGENTS_DISABLE_TRACING`, it breaks the tracing tests). A `CannotOverwriteExistingCassetteException` means a test made a request with no cassette: run `OPENAI_API_KEY=<real key> make record` locally (never in CI; `make record` refuses when `CI` is set) and commit the cassette. Cassettes must contain no secrets; the conventions check scans for them.

## CI

One entry workflow, one reusable workflow per language, plugin as a parameter, no secrets.

- `.github/workflows/ci.yml` (`pull_request`, `merge_group`, push to `main`, nightly, dispatch). Job `changes` runs `scripts/ci/detect_changes.py`: plugins are discovered from `<language>/*/<manifest>` (ignoring `_*`); a changed file under a plugin selects that plugin; a non-plugin file under a language root selects every plugin of that language; `.github/**` and `scripts/ci/**` select everything; `scripts/release/**` and `scripts/migrate/**` select only the script tests; push to `main`, nightly and dispatch select everything. Job `conventions` checks repository invariants and runs the script tests. Job `python` calls `_python-plugin.yml` once per selected plugin. Job `ci-status` fans in and is the only required check (skipped upstream jobs count as success).
- `.github/workflows/_python-plugin.yml`: job `matrix` reads `plugin.toml` `runtime-versions` and emits the same matrix for every run, pull requests included (ubuntu at the min and max versions, macOS and Windows at max); job `test` runs `make sync` (or `sync-latest` / `sync-lowest`), `make lint`, `make test`, then, on the ubuntu/max cell only, the `python-build-check` composite action (`make build`, `check_wheel.py`, isolated `smoke.py` on wheel and sdist). Windows runners install GNU make with choco.
- Dependency lanes: nightly runs every plugin with the newest allowed dependencies (`sync-latest`) and with the lowest allowed direct dependencies (`sync-lowest`), opening or updating one issue per failing plugin. The lowest-direct lane also runs, and blocks, on pull requests that change a plugin's `pyproject.toml` or `uv.lock`, because that is when floors change.
- Required checks on `main`: `ci-status`, plus the org-enforced required workflows that run automatically on every PR (`Check for CODEOWNERS`, `Opengrep SAST`), and `license/cla` once the CLA app is installed. Do not add a local opengrep caller; the org one already runs.

## Releases

Trusted publishing by ecosystem: PyPI uses OIDC trusted publishing (`pypa/gh-action-pypi-publish`, no stored token; PyPI cannot bind a reusable workflow, so publish jobs live inline in `release-python.yml`). npm supports OIDC trusted publishing (GitHub-hosted runners, npm >= 11.5.1, one publisher per package, register the calling workflow's filename; provenance is automatic for a public repo and package). Maven Central has no OIDC: Central Portal user token plus GPG signing, kept as environment-scoped secrets. Go has nothing to upload: an immutable tag plus `sum.golang.org` is the release.

Version policy (`release_tool.py check-version-policy`, evaluated against pypi.org only): a coordinate with no published release must start at exactly `1.0.0` (`ga`) or `0.1.0` (otherwise), pre-releases of that version allowed; an existing coordinate must be strictly greater than its highest published version, yanked releases included. Final versions additionally require `plugin.toml` `[release] allow-final = true` and no `TRANSITION(sdk-cutover)` marker in the plugin.

Runbook for `python/<name>`:
1. Open a release PR that sets `version` in `pyproject.toml` (re-sync from upstream first while the transition rules apply). Merge it.
2. `git tag -a python/<name>/v<version> -m "python/<name> v<version>"` on the merged `main` commit and push the tag. Tags must match `<language>/<name>/v<version>` and are protected by a tag ruleset.
3. `release-python.yml` validates the tag, runs the full test matrix, builds once, publishes to TestPyPI (environment `testpypi`), smoke-installs from TestPyPI in a clean project, and for final versions publishes to PyPI (environment `pypi`, required reviewers confirm the tag SHA is on `main`) and smoke-installs again.
4. A draft GitHub Release is created idempotently with generated notes and the artifacts. Edit the notes and publish it by hand.
5. If anything fails after upload, fix forward with the next `rcN`; uploaded files are immutable and tags are never moved.

## Migration and re-sync

`scripts/migrate/extract-sdk-python.sh` plus `scripts/migrate/README.md` are the procedure. Rules: find every historical path first; the script is frozen after a plugin's first import; import PRs are labelled `history-import` and merged with a merge commit; adaptation files are separate commits on top; every import and re-sync gets a row in `scripts/migrate/IMPORTS.md`. Expected verification numbers are in that README.

## Transition rules (until the SDK cutover PR merges)

- `sdk-python` is the source of truth for a migrated plugin's code, and its CI remains the authoritative signal. Bug fix procedure: fix upstream, merge, re-sync here with a `history-import` PR. Never cherry-pick or patch imported files here.
- Adaptation-only changes (manifest, `plugin.toml`, `Makefile`, conftest, workflows) are the only local commits; temporary ones carry `TRANSITION(sdk-cutover):` in the code comment or commit message.
- `plugin.toml` `[release] allow-final = false` blocks final tags and makes the provenance guard tolerate the SDK overlap. `scripts/ci/check_conventions.py --nightly` asserts the coordinate is still absent from PyPI.
- Re-sync before every tag.
- Pre-cutover hazard for users: installing `temporalio-openai-agents` next to `temporalio<=1.32` overlaps on files, and uninstalling the plugin deletes files it shares with the SDK (the SDK's `openai_agents/__init__.py` vanishes); repair by reinstalling `temporalio`. This is why only pre-releases to TestPyPI are allowed before cutover.

## Cutover checklist

SDK side: remove the module, tests, the `openai-agents` extra, CODEOWNERS lines, README news line and CI step; add a `:boom: Breaking Changes` changelog entry pointing at `uv add temporalio-openai-agents`; consider a helpful `ImportError` via `temporalio.contrib.__getattr__`, `pkgutil.extend_path` in `temporalio/__init__.py` and `temporalio/contrib/__init__.py` (enables editable installs), and a public export of `TemporalIdGenerator`. Plugin side: raise the `temporalio` floor to the cutover release; delete the second `uv sync` in `make sync`, `link-mode`, the README exception in the provenance guard and every other `TRANSITION(sdk-cutover)` marker; set `allow-final = true`; drop `upstream` from `plugin.toml`; flatten `tests/`. Docs: repoint `openai-agents.mdx` and the install text; decide where the plugin API reference is hosted (the SDK's pydoctor site loses these pages). Downstream: `samples-python` dependency groups, `auto-aie` path assumptions, `cicd-terraform` onboarding (merge commits allowed, required check `ci-status`, CLA app installed first).

## Verification commands

```bash
cd python/<name> && make sync && make lint && make test && make build
uv run --project scripts pytest                          # tooling script tests
uv run --project scripts python scripts/ci/check_conventions.py
uv run --project scripts python scripts/ci/detect_changes.py --dry-run --base origin/main
uv run --project scripts python scripts/ci/check_wheel.py --plugin-dir python/<name> --dist python/<name>/dist
```
