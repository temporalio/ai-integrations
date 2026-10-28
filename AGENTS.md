# AGENTS.md — temporalio/ai-integrations

Normative guide for humans and coding agents working in this repository. Other documents
(`README.md`, `CONTRIBUTING.md`, `python/README.md`) are short and
link here. Design source: the
[ai-integrations design doc](https://app.notion.com/p/temporalio/ai-integrations-3ca8fc5677388120b5c3cf5567f0327c).

## Do not

1. Do not create `__init__.py` in `python/<plugin>/src/temporalio/` or `src/temporalio/contrib/`. The SDK wheel owns those packages; `scripts/ci/check_wheel.py` fails the build if they appear.
2. Do not install with plain `uv sync` or run tests with a bare `uv run`. Use `make sync`, `make test`, and the other targets; they export `UV_NO_EDITABLE=1`.
3. Do not add a workflow file, job, or secret for one plugin. Plugin variation lives in `plugin.toml`, `pyproject.toml`, and the shared make targets.
4. Do not add `exclude-newer` without `exclude-newer-package = { temporalio = false }`, and do not replace `--locked` with `--frozen`.
5. Do not add `[tool.uv.sources]` path or workspace entries between plugins. Cross-plugin dependencies use published registry coordinates only.
6. Do not hand-edit the committed `0.0.0` development version. Protected release tags are the published version source; never move or delete a tag. A failed pre-release gets the next `rcN`.
7. Do not write `CHANGELOG.md` files. Release notes are generated from commit messages.
8. Do not put Python source or stub files in `python/_shared/`. Python test support is duplicated into each plugin (and its source template) so every file resolves against that plugin's environment. Non-Python make logic and tool configuration may be shared there when the format supports composition.

## Purpose and layout

AI plugins for Temporal SDKs. Each plugin is
independently installable, dependency-isolated, tested, versioned and released. Layout is
`<language>/<integration>/`; directories under a language root that start with `_` are shared
resources (`python/_shared/`, `python/_template/`) and are ignored by CI discovery.

| Folder | Coordinate | First version here | Maturity | Root API |
|---|---|---|---|---|
| `python/mcp` | `temporalio-mcp` | 0.1.0 | Pre-release | `temporalio.mcp` |
| `python/deepagents` | `temporalio-deepagents` | 0.0.1 | Pre-release | `temporalio.deepagents` |
| `python/google_adk` | `temporalio-google-adk` | 0.0.1 | Pre-release | `temporalio.google_adk` |
| `python/google_genai` | `temporalio-google-genai` | 0.1.0 | Public Preview | `temporalio.google_genai` |
| `python/langgraph` | `temporalio-langgraph` | 0.1.0 | Public Preview | `temporalio.langgraph` |
| `python/langsmith` | `temporalio-langsmith` | 0.1.0 | Public Preview | `temporalio.langsmith` |
| `python/openai_agents` | `temporalio-openai-agents` | 1.0.0 | Generally Available | `temporalio.openai_agents` |
| `python/strands_agents` | `temporalio-strands-agents` | 0.1.0 | Public Preview | `temporalio.strands_agents` |
| `python/typesafe` | `temporalio-typesafe` | 0.0.1 | Pre-release | `temporalio.typesafe` |
| `typescript/vercel-ai-sdk` | `@temporalio/vercel-ai-sdk` | 1.0.0 | Generally Available | `@temporalio/vercel-ai-sdk` |
| `typescript/google-adk` | `@temporalio/google-adk` | 0.1.0 | Public Preview | `@temporalio/google-adk` |
| `typescript/langsmith` | `@temporalio/langsmith` | continues (1.24.0 next) | Public Preview | `@temporalio/langsmith` |
| `typescript/openai-agents` | `@temporalio/openai-agents` | continues (1.24.0 next) | Generally Available | `@temporalio/openai-agents` |
| `typescript/strands-agents` | `@temporalio/strands-agents` | continues (1.24.0 next) | Pre-release | `@temporalio/strands-agents` |
| `java/spring-ai` | `io.temporal:spring-ai` | 0.1.0 | Public Preview | `io.temporal.springai` |
| `java/google-adk` | `io.temporal:google-adk` | 0.1.0 | Public Preview | `io.temporal.googleadk` |
| `go/googleadk` | `go.temporal.io/googleadk` | 0.1.0 (new coordinate) | Public Preview | `googleadk` |

"First version here" values are informational; the registry is the source of truth for the
version policy (below). Go plugins use `go.temporal.io/<name>` module paths and live under
`go/<name>/`. Before publishing, the vanity site must route each module to this repository's
subdirectory. The [Go module reference](https://go.dev/ref/mod#vcs-find) documents the fourth
`go-import` field for subdirectories, supported since Go 1.25. Google ADK requires Go 1.26.6,
so its mapping can be:

```html
<meta name="go-import" content="go.temporal.io/googleadk git https://github.com/temporalio/ai-integrations go/googleadk">
```

Its tags are `go/googleadk/v<version>`. `go/googleadk` is now owned here under the new module
path. UUID and span random-stream
identifiers retain their original strings for replay compatibility. The shared Go release
workflow verifies vanity routing before accepting a published module.

Naming derivation, enforced by `scripts/ci/check_conventions.py`: folder name = `plugin.toml`
`name`; Python coordinate = `temporalio-` + name with `_` replaced by `-`; Python root API =
`temporalio.<name>`; release tag = `<language>/<name>/v<version>`. Folders never end in `-plugin`
or `_plugin`.

Maturity mapping (`plugin.toml` `maturity` and the Python classifier must agree):
`pre-release` = `Development Status :: 3 - Alpha`; `public-preview` =
`Development Status :: 4 - Beta`; `generally-available` =
`Development Status :: 5 - Production/Stable`.
Use each plugin's public Temporal documentation for its release stage. In READMEs and
this table's Maturity column, use the three release-stage labels: Pre-release maps to
`pre-release`, Public Preview to `public-preview`, and Generally Available to
`generally-available`. Feature-specific stages do not change a plugin's
overall maturity (for example, OpenAI Agents is Generally Available with preview or experimental features).

## Repository invariants

- Each plugin owns its manifest and lockfile (`pyproject.toml` + `uv.lock`); no lockfile at a language root, no root Python project. `scripts/` is a separate tooling project, not a root project.
- Each plugin carries `plugin.toml` (metadata only: name, language, coordinate, registry, root API, maturity, docs, `[ci] runtime-versions`, `[smoke] imports`). It holds no secrets and no owners; `.github/CODEOWNERS` is `* @temporalio/ai-sdk`.
- The root `LICENSE` is the source of truth and every plugin directory carries a committed regular-file copy of it, because each wheel and sdist must ship the license text. The conventions check fails unless the plugin copy is committed and byte-identical to the root file (`cp LICENSE python/<name>/LICENSE` to refresh); `pyproject.toml` declares `license = "MIT"` and `license-files = ["LICENSE"]`; `check_wheel.py` verifies the packaged text equals the root file.
- Python manifests and lockfiles carry the static `0.0.0` development placeholder. A protected release tag is the published version source; the release matrix runs `uv version <tag-version>` before sync, test and build, and only those tested artifacts are published. Ordinary CI and local development keep `0.0.0`.
- No changelog files. `scripts/release/release_tool.py release-notes` derives notes from commit subjects touching the plugin directory since its previous tag.
- Every GitHub Action is pinned to a full commit SHA with a `# vN` comment (org opengrep rule).

## Python conventions

- Layout: `python/<name>/{pyproject.toml, uv.lock, plugin.toml, Makefile, README.md, LICENSE, src/temporalio/<name>/, tests/}`.
- Build backend `uv_build` with `module-name = "temporalio.<name>"`. `py.typed` ships in the leaf package (redundant with the SDK's marker, kept on purpose).
- Installs are non-editable. `temporalio` is a regular package owned by the SDK wheel, so an editable install of a plugin can resolve incorrectly unless the SDK extends its package path. `python/_shared/python.mk` exports `UV_NO_EDITABLE=1`; `[tool.uv] cache-keys` includes `src/**/*` so edits trigger a rebuild; `link-mode = "copy"` keeps overwrites deterministic.
- Provenance guard (`tests/helpers/provenance.py`, mirrored by `scripts/ci/smoke.py`) runs at every pytest session start and fails loudly if the install is editable, any file differs from the distribution's RECORD, files under the package directory are not owned by the distribution, or another distribution ships the same paths. `tests/test_installed_matches_source.py` additionally byte-compares the installed package with `src/`.
- Dependency cooldown: `exclude-newer = "2 weeks"` (org supply-chain policy) with `exclude-newer-package = { temporalio = false }` so a new SDK release is adoptable the day it ships; a plugin that depends on another plugin adds that coordinate too (`temporalio-mcp = false` in `openai_agents`), otherwise a fresh first-party release is invisible to `uv lock` for two weeks. Declare exactly what the package imports at module level; `openinference` and similar lazy imports go in an extra.
- Tooling: ruff, pyright, basedpyright, mypy (`mypy_path = "src"`, `explicit_package_bases`), pydocstyle (google), pytest + xdist (`-n auto --dist=worksteal`; the `os._exit(0)` hook is xdist-aware). `python/_shared/` owns the make recipes and composable Pyright/Ruff defaults; mypy and pydocstyle settings stay in each `pyproject.toml` because those tools cannot extend another config while preserving plugin overrides. All tools are invoked through `make` targets; see `make help`.
- Tests self-provision the Temporal dev server (`WorkflowEnvironment.start_local`, version pinned in `tests/__init__.py`) with its default configuration; add a `--dynamic-config-value` flag in a plugin's conftest only when one of its tests needs a server feature that is off by default. Bind those options in a startup callback passed to `tests/helpers/environment.py`'s `start_local_with_retry`. This plugin-agnostic policy allows three startup attempts with a one-second delay for the SDK's explicit startup-deadline message; other errors and cancellation propagate immediately.
- Test scaffolding: each plugin owns its provenance guard, local dev-server fixtures and pytest hooks. New plugins copy the standard implementations from `python/_template/tests/`; the conventions check keeps `environment.py`, `plugin_meta.py`, `provenance.py`, `test_env.py`, and `test_installed_matches_source.py` byte-identical to their canonical templates. Plugin-specific fixtures stay in that plugin's conftest. Tests must not require real provider credentials; use deterministic local models, mock transports, or in-process servers to exercise provider behavior in CI.

## Java conventions

- Each Java plugin owns a Gradle wrapper with distribution checksum, `build.gradle`,
  `settings.gradle`, `plugin.toml`, and committed dependency locks. Plugins with a
  `spring-boot-versions` matrix lock once per Boot version
  (`gradle/dependency-locks/spring-boot-<version>.lockfile`); the others keep one
  `gradle/dependency-locks/dependencies.lockfile`. Shared build conventions live in
  `java/_shared/java.gradle`.
- Every Java plugin starts `settings.gradle` with a `pluginManagement` block applying
  `../_shared/settings.gradle`. The shared script checks Maven Central before the
  Gradle Plugin Portal; the conventions check enforces this entry point for every plugin.
- Java development versions are `0.0.0`; a release tag supplies `-PreleaseVersion`
  before tests and builds. Do not publish the placeholder or add snapshot automation.
- Java packages and publication metadata use the root MIT license.
- CI uses metadata to test Ubuntu on Java 17/25 and macOS/Windows on Java 25.
  All compatibility variants run on PRs. The primary Ubuntu/max variant alone
  produces the tested Maven distributions and runs a clean consumer smoke test.
- `./gradlew spotlessCheck test stageDist` checks without rewriting source. Updating
  dependency locks is intentional: `./gradlew resolveAndLockAll --write-locks`.
- Nightly and `latest-deps=true` runs use `-PdependencyMode=latest`, selecting stable
  releases within the plugin's declared dependency families and each configured
  Spring Boot major series. Run `resolveAndLockAll --write-locks --refresh-dependencies`
  first; subsequent test and build commands reuse a separate ignored latest lock.
  Ordinary CI and releases default to `-PdependencyMode=locked` and committed locks.

## TypeScript conventions

- Each `typescript/<name>/` is an independent pnpm project with `package.json`,
  `pnpm-lock.yaml`, `pnpm-workspace.yaml`, `plugin.toml`, `tsconfig.json`, `Makefile`,
  `README.md` and a regular-file copy of the root MIT `LICENSE`. No language-level
  workspace or lockfile. Shared make, TypeScript and ESLint configuration lives
  in `typescript/_shared/`.
- Coordinates and root APIs are `@temporalio/<name>`. The development version is
  `0.0.0`. Node 22.12.0 is the repository minimum (provider packages require Node 22
  and unflagged CommonJS loading of ES modules); a plugin whose dependencies require
  more declares its own floor in `engines` (`google-adk`: 22.17.0, which
  `@mikro-orm/core` 7 requires through `@google/adk` 2.1 and later). CI tests the
  latest Node 22.x and 24.x using major selectors with `check-latest: true`; a plugin
  may name its floor's `major.minor` as the first `[ci] runtime-versions` entry so the
  minimum cell runs that line's latest patch instead of the latest 22.x, which such a
  plugin then does not test (`google-adk` tests 22.17 and 24). Use pnpm 10.27.0 and
  `make sync`, `make lint`, `make test`, `make build`.
- SDK and Workflow Streams dependencies use published npm coordinates. The
  unpublished SDK test helpers are copied into each consuming plugin's
  `tests/helpers/`, with a private development-only link. Never use a local
  SDK checkout or another plugin's checkout as a dependency.
- Local server startup uses `startLocalWithRetry`, with three attempts and a
  one-second delay only for the SDK's explicit startup-deadline error. Bind each
  plugin's server configuration in the startup callback. Copy the helper and its
  regression tests from `typescript/_template/src/__tests__/`; the conventions
  check keeps each plugin's copies byte-identical to the templates.
- `make sync` preserves the committed lock. `make sync-latest` updates compatible
  ranges under the two-week pnpm release-age policy. Preserve first-party and
  exact AI SDK dependency-family cooldown exceptions inherited from the SDK.
- Activity names remain stable for workflow replay compatibility.
- All five TypeScript integrations are enabled for release
  through the shared `release-typescript.yml` workflow. New coordinates require one
  interactive publication of the tested tarball before npm permits configuring their trusted
  publisher, as described below. No per-plugin workflows or secrets. CI checks
  tarball contents and a clean consumer import.

## CI

One entry workflow, one reusable workflow per language, plugin as a parameter, no secrets.

- `.github/workflows/ci.yml` (`pull_request`, `merge_group`, push to `main`, nightly, dispatch). Job `changes` runs `scripts/ci/detect_changes.py`: plugins are discovered from `<language>/*/<manifest>` (ignoring `_*`); a changed file under a plugin selects that plugin; a non-plugin file under a language root selects every plugin of that language; `.github/**` and `scripts/ci/**` select everything; `scripts/release/**` select only the script tests; push to `main`, nightly and dispatch select everything. Job `conventions` checks repository invariants and runs the script tests. Jobs `python`, `typescript`, `java`, and `go` call their per-language reusable workflows once per selected plugin. Job `ci-status` fans in and is the only required check (skipped upstream jobs count as success).
- `.github/workflows/_typescript-plugin.yml`: uses the shared runtime/platform matrix and independent pnpm lockfile for each plugin. Every cell runs sync, lint and tests; ubuntu/max also packs, verifies contents and smoke-installs the tarball in a clean consumer. Nightly selects compatible latest dependencies. TypeScript nightly issues use `typescript/<plugin>` keys to avoid collisions with Python.
- `.github/workflows/_python-plugin.yml`: job `matrix` reads `plugin.toml` `runtime-versions` and emits the same matrix for every run, pull requests included (ubuntu at the min and max versions, macOS and Windows at max); job `test` runs `make sync` (or `sync-latest` on nightly runs), `make lint`, `make test`, then, on the ubuntu/max cell only, the `python-build-check` composite action (`make build`, `check_wheel.py`, isolated `smoke.py` on wheel and sdist). Windows runners install GNU make with choco.
- `.github/workflows/_go-plugin.yml`: uses the same matrix generator and platform layout with each plugin's Go `runtime-versions`. Every cell runs `make sync` (or `sync-latest`), `make lint`, `make test`, and `make build`; the ubuntu/max cell enables `-race`. `GOTOOLCHAIN=local` prevents dependencies from silently replacing the runtime under test. Windows runners install GNU make with choco. Go action caching keys include the plugin's `go.mod` and `go.sum`.
- Dependencies: ordinary CI uses the committed lockfile (`go.mod`/`go.sum` for Go); local `make sync` uses the existing versions. Nightly runs every plugin with the newest allowed dependencies (`sync-latest`) without committing the updated lock; manual dispatch can select that mode too. Python make targets use `uv run --locked`; Go lint, test and build use `-mod=readonly` to preserve the versions selected during sync. Go `sync-latest` runs `go get -u -t ./...`, tidies the module and verifies its checksums.
- Required checks on `main`: `ci-status`, `Check for CODEOWNERS` and `opengrep/scan` (the last two are org-enforced workflows that run automatically on every PR), plus one approving review from a code owner; `license/cla` joins once the CLA app is installed. Do not add a local opengrep caller; the org one already runs. Repository setup: branch protection, the `testpypi`/`pypi` environments (tag policy `python/*/v*`) and the release-tag ruleset were configured by hand on 2026-09-09. The `pypi` required-reviewer gate was removed on 2026-10-07.
- Nightly failures: `scripts/ci/nightly_report.py` opens or updates one `nightly` issue per failing plugin from the job names `Python (<plugin>) / ...`, `Java (<plugin>) / ...`, and `Go (<plugin>) / ...`; `scripts/tests/test_nightly_report.py` fails if `ci.yml` renames those jobs. Existing Python and Java issue titles stay unchanged; Go keys use `go/<plugin>` so plugins in different languages cannot share an issue.

## Releases

Release procedures and publication policy belong only in this file. READMEs and
contributing guides cover installation, usage, and local development.

Pre-release and Public Preview plugins publish without a required-reviewer approval
gate in every language. Use the plugin's `maturity`, independently of whether its
version is a prerelease. Keep automated validation, immutable tags, tested-artifact
verification, and final-release eligibility checks. GA plugins retain any existing
approval policy.

Trusted publishing by ecosystem: PyPI uses OIDC trusted publishing (`pypa/gh-action-pypi-publish`, no stored token; PyPI cannot bind a reusable workflow, so publish jobs live inline in `release-python.yml`). npm supports OIDC trusted publishing (GitHub-hosted runners, npm >= 11.5.1, one publisher per package, register the calling workflow's filename; provenance is automatic for a public repo and package). Maven Central has no OIDC: Central Portal user token plus GPG signing, kept as environment-scoped secrets. Go has nothing to upload: an immutable tag plus `sum.golang.org` is the release.

After a successful dry run, use its full commit SHA as `<tested-sha>` when generating
notes with `release-notes --ref` and creating the draft with `gh release create
--target`. This pins both to the tested source even if `main` advances. GitHub creates
the new tag when the draft is published.
Use `--prerelease` when the version is a prerelease or the plugin's maturity is
Pre-release. Release tags are immutable once created.

### Go releases

`.github/workflows/release-go.yml` handles every Go plugin. It runs on pushes of
`go/*/v*` tags, published Go GitHub Releases, and manual dispatch. Go versions use
canonical SemVer (`0.1.0`, `0.1.0-rc.1`), with no build metadata. The shared release
tool checks the module path, maturity's first version, and forward ordering against
`proxy.golang.org`, including retracted releases. A re-run of the newest version is
allowed.
The currently supported vanity coordinates have no `/vN` suffix; v2+ requires a
module-path change before the release tooling can support it.

The full `_go-plugin.yml` matrix tests locked dependencies at one pinned commit,
including race detection on Ubuntu/max. Tagged runs additionally verify the
four-field vanity mapping, download the exact public version with a clean module
cache and `sum.golang.org` verification, compare the complete module zip and
`go.mod` bytes with that tested checkout, and build a clean consumer importing the
root package. Temporary proxy propagation failures are retried. The workflow uses
read-only repository permissions and creates no tags or GitHub Releases.

**Creating a Go tag makes the version discoverable before CI finishes.** A GitHub
draft does not hide an existing tag from Go consumers. Complete the dry run before
publishing the release:

1. Merge the release changes to `main`. Configure the vanity mapping above and
   include Go tags in the protected release-tag ruleset. Do not alter `go.mod` dependency versions for a release.
2. Dispatch `gh workflow run release-go.yml --ref main -f tag=go/<name>/v<version>`.
   Branch dispatches always validate, test and build without consuming a tag or
   fetching the proposed module. An explicit `-f dry-run=true` does the same for
   an existing tag ref. Use `0.1.0-rc.1` for
   the Google ADK dry run.
3. After the full matrix succeeds, generate notes with `uv run --project scripts --locked python
   scripts/release/release_tool.py release-notes --plugin-dir go/<name>
   --tag go/<name>/v<version> --ref <tested-sha> --output /tmp/release-notes.md`.
   Create the draft with `gh release create go/<name>/v<version>
   --target <tested-sha> --draft --notes-file /tmp/release-notes.md`, adding
   `--prerelease` when required.
4. Review the draft and publish it with
   `gh release edit go/<name>/v<version> --draft=false`. GitHub creates the tag on
   the tested commit. The release workflow repeats the matrix and verifies the
   public module, including the release's prerelease classification. The GitHub
   Release and Go version are visible before this verification finishes. A failed
   release requiring changed source gets the next version, such as `0.1.0-rc.2`.
5. For a proxy or consumer verification failure, re-run failed jobs or dispatch
   `release-go.yml` on the same tag ref with no tag input. Never move or delete
   the tag. A missing vanity mapping must be repaired before retrying.

### TypeScript releases

`.github/workflows/release-typescript.yml` handles every npm plugin. Published
GitHub Releases trigger publication; tag pushes alone do not. Branch dispatches
always run a dry run, even when `skip-publish` is false. npm versions are canonical
SemVer without build metadata, such as `0.1.0` or `0.1.0-rc.1`. Registry ordering
includes deprecated releases; the newest exact version may be verified again.
Keep the
committed development version `0.0.0`.

1. Configure the shared GitHub environment `npm` to accept only
   `typescript/*/v*` tags and include that pattern in the immutable release-tag
   ruleset. Existing packages register a trusted publisher for GitHub repository
   `temporalio/ai-integrations`, workflow `release-typescript.yml`, environment
   `npm`, with direct `npm publish` allowed. No stored npm tokens are used.
   The `npm` environment and TypeScript tag protections were configured on
   2026-10-15; a new coordinate's npm-side publisher still follows step 4.
2. Merge release changes to main. Dry-run with
   `gh workflow run release-typescript.yml --ref main -f tag=typescript/<name>/v<version>`.
   The full Node/platform matrix tests the injected version at one pinned commit;
   Ubuntu/max packs, checks and smoke-installs the tarball and saves
   `npm-dist-<name>` as an Actions artifact.
3. Generate notes with `uv run --project scripts --locked python
   scripts/release/release_tool.py release-notes --plugin-dir typescript/<name>
   --tag typescript/<name>/v<version> --ref <tested-sha> --output /tmp/release-notes.md`.
   Create the draft with `gh release create typescript/<name>/v<version>
   --target <tested-sha> --draft --notes-file /tmp/release-notes.md`, adding
   `--prerelease` when required.
4. For a new npm coordinate only, npm requires the package to exist before a
   trusted publisher can be registered. Download the successful dry run's tested
   `npm-dist-<name>` tarball and record its run ID. Authenticate interactively with
   an authorized npm account and publish
   those exact bytes using `npm publish <tarball> --access public --tag latest
   --ignore-scripts` (use `next` for prerelease versions). Then register the trusted
   publisher with `npm trust github @temporalio/<name> --repo temporalio/ai-integrations
   --file release-typescript.yml --env npm --allow-publish`. Complete any npm 2FA
   challenge interactively. Do not upload a placeholder package or add a token.
5. Publish the reviewed draft with
   `gh release edit typescript/<name>/v<version> --draft=false`. GitHub creates the
   tag on the tested commit and starts the workflow. A later dispatch on that tag
   ref can retry publication.
   After the full matrix succeeds, the inline OIDC job publishes the tested
   tarball to `latest` for final versions or `next` for prereleases, verifies npm's
   SHA-512 integrity against the tested tarball, and installs that public version
   in a clean consumer. npm verifies the downloaded bytes against that integrity.
   It leaves the reviewed GitHub Release unchanged.
   After the bootstrap upload in step 4, dispatch
   `gh workflow run release-typescript.yml --ref typescript/<name>/v<version> -f recover-run=<dry-run-id>`.
   This verifies the exact tarball already uploaded, even if the automatic release
   run rebuilt different bytes. Recovery requires that the source dry run or
   release run completed at the tag's exact main-reachable commit, passed version
   validation and every matrix job, and retains one unexpired `npm-dist-<name>`
   artifact. It downloads that artifact without rebuilding. The downloaded
   package must have the release's name and version and be public before any upload.
6. Re-run failed jobs after an upload. An existing version is skipped only when
   its registry integrity matches the tested tarball. For a fresh run,
   dispatch on the same tag with `-f recover-run=<original-run-id>` to reuse those
   tested bytes. The published GitHub Release and normal registry policy are still
   required. Recovery is accepted only on a tag ref; branch dispatches remain dry
   runs. A full rerun can replace its saved artifact, so preserve a downloaded copy
   after manually publishing it and prefer re-running failed jobs. If the original
   artifact is unavailable or its bytes differ from npm, fix forward with a new
   version; never move a tag or replace an uploaded artifact.

### Python releases

Every Python plugin uses the `testpypi` and `pypi` GitHub environments without required reviewers, regardless of maturity. Both environments allow the `python/*/v*` tag pattern and `main` for validated publication recovery. Publishing the GitHub Release starts the release pipeline; eligible versions upload to PyPI automatically after the full test matrix and TestPyPI verification succeed. Trusted publishers remain bound to their existing environment names.

Version policy (`release_tool.py check-version-policy`; version ordering is evaluated against pypi.org, test.pypi.org, or Maven Central's metadata): a coordinate with no published release must start at exactly `1.0.0` (`generally-available`), `0.1.0` (`public-preview`), or `0.0.1` (`pre-release`), pre-releases of that version allowed; an existing coordinate must be strictly greater than its highest published version, yanked releases included. Maturity is independent of PEP 440 version status: a Pre-release plugin can publish a final version such as `0.0.1` to PyPI. TestPyPI versions also move forward. A version already staged on TestPyPI, or already the newest release on pypi.org, only produces a warning (a re-run after an upload is the normal recovery path); an older staged version is rejected. Each Python smoke job proves the index serves exactly the artifacts this run built, none yanked (`verify-index-files`). Java smoke jobs compare staged and public Maven artifacts byte for byte with the tested files.

Runbook for `python/<name>`:
1. Merge every code and dependency change intended for the release. Do not change the committed `0.0.0` development version.
2. Choose a canonical PEP 440 version and dry-run it on `main`: `gh workflow run release-python.yml --ref main -f tag=python/<name>/v<version>`. The workflow injects the tag version into each checkout, runs policy validation, the full test matrix and the artifact build, but uploads nothing and consumes no tag. `-f skip-publish=true` does the same for a dispatch on an existing tag ref.
3. Generate notes with `uv run --project scripts --locked python scripts/release/release_tool.py release-notes --plugin-dir python/<name> --tag python/<name>/v<version> --ref <tested-sha> --output /tmp/release-notes.md`. Create the draft with `gh release create python/<name>/v<version> --target <tested-sha> --draft --notes-file /tmp/release-notes.md`, adding `--prerelease` when required. If GitHub release assets are wanted, attach the tested distributions downloaded from the successful dry run before publication; the workflow retains its distributions as Actions artifacts.
4. Publish the reviewed GitHub Release using the GitHub UI or `gh release edit python/<name>/v<version> --draft=false`. GitHub creates the protected tag on the tested commit. Its `release: published` event starts `release-python.yml` for Python tags; pushing a tag alone does not start publication. Publication with a workflow's `GITHUB_TOKEN` does not trigger this event workflow; use a human's session or the manual dispatch fallback on the tag. The GitHub Release is now visible even if package publication later fails.
5. `release-python.yml` validates the main-reachable tag, version policy, published release and GitHub pre-release classification, injects the tag's version, and runs the full test matrix (its ubuntu dist cell builds, checks and smoke-tests the wheel and sdist). It publishes those tested artifacts to TestPyPI (environment `testpypi`), proves TestPyPI serves exactly those files, and smoke-installs from TestPyPI in a clean project. Eligible versions then publish automatically to PyPI (environment `pypi`), with the published release linked from the deployment; the workflow repeats the artifact proof and smoke there. PEP 440 pre-releases go to TestPyPI only unless a tag dispatch uses `publish-prerelease-to-pypi`; GitHub's pre-release flag does not control index routing. The workflow reads the GitHub Release without changing its notes or assets.
6. If a job fails after an upload, use "Re-run failed jobs" on that run: `prepare`'s outputs and the tested artifact survive, the upload is skipped, and the smoke jobs verify the served files. If the tagged workflow or tooling itself needs a repair, merge the repair first, then dispatch on `main` with `-f tag=python/<name>/v<version> -f recover-run=<original-release-run-id>`. Recovery validates that the completed source run is a release-event run (or a legacy tag-push run) for the exact main-reachable tag, passed version validation and every test-matrix job, and retains an unexpired distribution artifact. A published GitHub Release for the tag must exist, and package manifests must still match the tag. It downloads those tested bytes without rebuilding; the normal ref restrictions, registry ordering and file-hash checks still apply. Recovery requires a `main` branch deployment policy in the `testpypi` and `pypi` environments. A fresh dispatch on the tag also passes the version policy (the newest published version is treated as a re-run, with a warning) but rebuilds the artifacts, and `verify-index-files` fails if the rebuild is not byte-identical (a different uv version stamps its `Generator` into the wheel). If the artifacts themselves must change, fix forward with the next `rcN`; uploaded files are immutable and tags are never moved.

Runbook for `java/<name>`:

The shared workflow starts when the reviewed GitHub Release is published. Tag
pushes alone do not start staging. The workflow verifies the published release
before signing or uploading and retains tested distributions and deployment state
as Actions artifacts.

1. Merge the changes intended for the release. Keep the committed `0.0.0` development version.
2. Configure shared environments `maven-central-staging` and `maven-central`, both
   accepting tags `java/*/v*` only, with no required reviewers. Every plugin uses
   `maven-central-staging` for private staging and `maven-central` for public
   publication, so their deployment histories track the two phases separately.
   The shared `maven-central-approval` environment accepts the same tag pattern,
   requires an `@temporalio/ai-sdk` reviewer and prevents self-review; it holds no
   secrets. GA publication must pass this approval job first. Only recognized
   Pre-release and Public Preview maturities skip it; missing or unknown maturity
   requires approval. Extend the immutable release-tag ruleset to Java without
   relaxing its Python rules.
   Credential installation remains a maintainer setup step.
3. In staging, install environment secrets `CENTRAL_USERNAME`, `CENTRAL_PASSWORD`
   (a Central Portal user-token pair), `GPG_PRIVATE_KEY` (armored private key or
   base64-encoded secret keyring), and
   `GPG_PASSPHRASE`; set environment variable `GPG_FINGERPRINT` to the full public
   fingerprint. Production needs only the Central token pair, with access to the
   staging deployment. Use credentials from the same publishing account in both.
   GitHub cannot export existing secret values;
   an authorized maintainer must install them from the credential source. Publish
   the public key to a supported keyserver before Central validation; see
   [Sonatype's GPG requirements](https://central.sonatype.org/publish/requirements/gpg/).
4. Dry-run the candidate on main:
   `gh workflow run release-java.yml --ref main -f tag=java/spring-ai/v0.1.0-RC1`.
   The full compatibility matrix tests the injected version; its primary
   Ubuntu/max cell builds and verifies the five Maven artifacts and installs a
   clean consumer. A branch dispatch never signs, uploads, or creates a release.
5. Generate notes with `uv run --project scripts --locked python
   scripts/release/release_tool.py release-notes --plugin-dir java/spring-ai
   --tag java/spring-ai/v0.1.0-RC1 --ref <tested-sha> --output /tmp/release-notes.md`.
   Create the draft with `gh release create java/spring-ai/v0.1.0-RC1
   --target <tested-sha> --draft --prerelease --notes-file /tmp/release-notes.md`.
   Review the notes and attach any wanted tested distributions from the dry run
   before publication. Publish with
   `gh release edit java/spring-ai/v0.1.0-RC1 --draft=false`; GitHub creates the tag
   on the tested commit and starts the workflow. The full matrix runs again, then
   the shared workflow signs the tested bytes, adds checksums, uploads a
   `USER_MANAGED` bundle to Central Portal,
   waits for `VALIDATED`, compares all staged files byte for byte, and installs
   a clean consumer from the authenticated staging endpoint. Candidates remain
   privately staged in Central Portal. The GitHub Release is visible before this
   validation finishes. The signed bundle and deployment metadata are retained as
   Actions artifacts; the workflow leaves the reviewed release unchanged.
6. Publish the tested Maven artifacts.
   The workflow publishes the already verified deployment automatically for non-GA
   plugins; GA plugins require `maven-central-approval` approval first. It waits for
   `PUBLISHED`, proves Maven Central serves the same files, and installs a clean
   consumer after the GitHub Release is published. Candidates retain the private
   staging behavior above.
   Publication and recovery share a 30-minute polling budget within the 45-minute
   job; a polling timeout preserves the deployment for a failed-job rerun.

7. After an upload failure, use **Re-run failed jobs**. The workflow preserves the
   signed bundle, upload intent, and deployment ID even if validation or the consumer
   fails. It reuses those signatures and never retries an ambiguous upload POST.
   Release-event reruns and legacy tag-push reruns inspect every earlier attempt.
   Once the staging command may have run, recovery requires the saved state; a missing or failed state download
   stops the job without signing, uploading, or replacing the saved artifact.
   If all earlier attempts skipped staging, a rerun may perform the first upload.
   A fresh publishing dispatch on the same tag requires `-f deployment-id=<Portal UUID>`;
   recovery downloads the existing signed files and rejects any rebuilt-byte
   mismatch or deployment containing additional coordinates. If an upload response
   was lost, inspect Portal for its ID before retrying. Publishing dispatches cannot
   start a new upload, even when Portal is still validating the original upload;
   the initial upload runs on GitHub Release publication. A dispatch with
   `skip-publish=true` remains a credential-free dry run and does not require a deployment ID.
   Never move a tag or replace uploaded files: a changed candidate gets the next `RCN`.
   Portal does not expose
   TestPyPI-style staging version enumeration; public ordering is checked against
   Maven Central and an existing validated staged version requires explicit recovery.

The Java pipeline uses the
[Central Portal Publisher API](https://central.sonatype.org/publish/publish-portal-api/)
directly. Maven Central remains the public host. Snapshot publishing is not configured.

## Verification commands

```bash
cd python/<name> && make sync && make lint && make test && make build
uv run --project scripts --locked pytest scripts/tests -q # tooling script tests
uv run --project scripts python scripts/ci/check_conventions.py
uv run --project scripts python scripts/ci/detect_changes.py --dry-run --base origin/main
uv run --project scripts python scripts/ci/check_wheel.py --plugin-dir python/<name> --dist python/<name>/dist
```
