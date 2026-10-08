# AGENTS.md — temporalio/ai-integrations

Normative guide for humans and coding agents working in this repository. Other documents
(`README.md`, `CONTRIBUTING.md`, `python/README.md`, `scripts/migrate/README.md`) are short and
link here. Design source: the
[ai-integrations design doc](https://app.notion.com/p/temporalio/ai-integrations-3ca8fc5677388120b5c3cf5567f0327c).

## Do not

1. Do not create `__init__.py` in `python/<plugin>/src/temporalio/` or `src/temporalio/contrib/`. The SDK wheel owns those packages; `scripts/ci/check_wheel.py` fails the build if they appear.
2. Do not edit imported files (`src/...`, `tests/contrib/<plugin>/...`) while the plugin's `plugin.toml` still names an `upstream`. Fix upstream, then re-sync. Preserve documented local-only divergences, such as the `openai_agents` MCP v2 adapter, when resolving a re-sync.
3. Do not apply `history-import` to work that is not reachable from the upstream repository's default branch, including work from an unmerged or closed upstream PR. A valid `history-import` PR must be merged with "Create a merge commit", never squash or rebase.
4. Do not install with plain `uv sync` or run tests with a bare `uv run`. Use `make sync`, `make test`, and the other targets; they export `UV_NO_EDITABLE=1`.
5. Do not add a workflow file, job, or secret for one plugin. Plugin variation lives in `plugin.toml`, `pyproject.toml`, and the shared make targets.
6. Do not add `exclude-newer` without `exclude-newer-package = { temporalio = false }`, and do not replace `--locked` with `--frozen`.
7. Do not add `[tool.uv.sources]` path or workspace entries between plugins. Cross-plugin dependencies use published registry coordinates only.
8. Do not hand-edit the committed `0.0.0` development version. Protected release tags are the published version source; never move or delete a tag. A failed pre-release gets the next `rcN`.
9. Do not write `CHANGELOG.md` files. Release notes are generated from commit messages.
10. Do not modify `samples-python`, `documentation`, or `sdk-python` from this repository. Those edits belong to the cutover checklist below.
11. Do not put Python source or stub files in `python/_shared/`. Python test support is duplicated into each plugin (and its source template) so every file resolves against that plugin's environment. Non-Python make logic and tool configuration may be shared there when the format supports composition.

## Purpose and layout

AI plugins for Temporal SDKs, factored out of the SDK repositories so each plugin is
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
| `typescript/vercel-ai-sdk` | `@temporalio/vercel-ai-sdk` | 1.0.0 | Generally Available | `@temporalio/vercel-ai-sdk` |
| `typescript/google-adk` | `@temporalio/google-adk` | 0.1.0 | Public Preview | `@temporalio/google-adk` |
| `typescript/langsmith` | `@temporalio/langsmith` | continues (1.24.0 next) | Public Preview | `@temporalio/langsmith` |
| `typescript/openai-agents` | `@temporalio/openai-agents` | continues (1.24.0 next) | Generally Available | `@temporalio/openai-agents` |
| `typescript/strands-agents` | `@temporalio/strands-agents` | continues (1.24.0 next) | Pre-release | `@temporalio/strands-agents` |
| `java/spring-ai` | `io.temporal:spring-ai` | 0.1.0 (0.1.0-RC1 planned) | Public Preview | `io.temporal.springai` |
| `go/googleadk` | `go.temporal.io/sdk/contrib/googleadk` | continues (v0.3.0 next) | Public Preview | `googleadk` |

"First version here" values are informational; the registry is the source of truth for the
version policy (below). The Go row has an unresolved problem: a module served by the static vanity
site cannot live in a monorepo subdirectory under an unchanged import path; decide (split mirror
repo, new import path, or staying in sdk-go) before that migration.

Naming derivation, enforced by `scripts/ci/check_conventions.py`: folder name = `plugin.toml`
`name`; Python coordinate = `temporalio-` + name with `_` replaced by `-`; Python root API =
`temporalio.<name>`; release tag = `<language>/<name>/v<version>`. Folders never end in `-plugin`
or `_plugin`. An upstream-backed migration may temporarily retain `temporalio.contrib.<name>` only
while `[release] allow-final = false`.

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
- Each plugin carries `plugin.toml` (metadata only: name, language, coordinate, registry, root API, maturity, docs, optional upstream, `[release] allow-final`, `[ci] runtime-versions`, `[smoke] imports`). It holds no secrets and no owners; `.github/CODEOWNERS` is `* @temporalio/ai-sdk`.
- The root `LICENSE` is the source of truth and every plugin directory carries a committed regular-file copy of it, because each wheel and sdist must ship the license text. The conventions check fails unless the plugin copy is committed and byte-identical to the root file (`cp LICENSE python/<name>/LICENSE` to refresh); `pyproject.toml` declares `license = "MIT"` and `license-files = ["LICENSE"]`; `check_wheel.py` verifies the packaged text equals the root file.
- Python manifests and lockfiles carry the static `0.0.0` development placeholder. A protected release tag is the published version source; the release matrix runs `uv version <tag-version>` before sync, test and build, and only those tested artifacts are published. Ordinary CI and local development keep `0.0.0`.
- No changelog files. `scripts/release/release_tool.py release-notes` derives notes from commit subjects touching the plugin directory since its previous tag.
- Every GitHub Action is pinned to a full commit SHA with a `# vN` comment (org opengrep rule).
- Dependabot mirrors sdk-python: security advisories only (`open-pull-requests-limit: 0`) because its uv support is not yet mature enough for routine bumps; nightly checks with the newest allowed dependencies are the dependency-drift signal.

## Python conventions

- Layout: `python/<name>/{pyproject.toml, uv.lock, plugin.toml, Makefile, README.md, LICENSE, src/temporalio/<name>/, tests/}`. Migrated plugins may temporarily keep the upstream source and test trees while final releases are disabled; flatten and move to the final root at cutover.
- While a plugin still names an `upstream`, preserve imported READMEs unchanged. If upstream-relative links need adaptation for PyPI, add a separate `README.pypi.md` and select it with `[project] readme`; conventions check the published README's links.
- Build backend `uv_build` with `module-name = "temporalio.<name>"`. `py.typed` ships in the leaf package (redundant with the SDK's marker, kept on purpose).
- Installs are non-editable. `temporalio` is a regular package owned by the SDK wheel, so an editable install of a plugin can resolve incorrectly unless the SDK extends its package path. `python/_shared/python.mk` exports `UV_NO_EDITABLE=1` and reinstalls the plugin last to support overlapping migrations; `[tool.uv] cache-keys` includes `src/**/*` so edits trigger a rebuild; `link-mode = "copy"` keeps overwrites deterministic during the transition.
- Provenance guard (`tests/helpers/provenance.py`, mirrored by `scripts/ci/smoke.py`) runs at every pytest session start and fails loudly if the install is editable, any file differs from the distribution's RECORD, files under the package directory are not owned by the distribution, or another distribution ships the same paths. While `plugin.toml` `[release] allow-final = false`, the SDK's overlap (`temporalio<=1.32` ships `temporalio/contrib/openai_agents/*`) is tolerated with a warning. `tests/test_installed_matches_source.py` additionally byte-compares the installed package with `src/`.
- Dependency cooldown: `exclude-newer = "2 weeks"` (org supply-chain policy) with `exclude-newer-package = { temporalio = false }` so a new SDK release is adoptable the day it ships; a plugin that depends on another plugin adds that coordinate too (`temporalio-mcp = false` in `openai_agents`), otherwise a fresh first-party release is invisible to `uv lock` for two weeks. Declare exactly what the package imports at module level; `openinference` and similar lazy imports go in an extra.
- Tooling: ruff, pyright, basedpyright, mypy (`mypy_path = "src"`, `explicit_package_bases`), pydocstyle (google), pytest + xdist (`-n auto --dist=worksteal`; the `os._exit(0)` hook is xdist-aware). `python/_shared/` owns the make recipes and composable Pyright/Ruff defaults; mypy and pydocstyle settings stay in each `pyproject.toml` because those tools cannot extend another config while preserving plugin overrides. All tools are invoked through `make` targets; see `make help`.
- Tests self-provision the Temporal dev server (`WorkflowEnvironment.start_local`, version pinned in `tests/__init__.py`) with its default configuration; add a `--dynamic-config-value` flag in a plugin's conftest only when one of its tests needs a server feature that is off by default. Bind those options in a startup callback passed to `tests/helpers/environment.py`'s `start_local_with_retry`. This plugin-agnostic policy allows three startup attempts with a one-second delay for the SDK's explicit startup-deadline message; other errors and cancellation propagate immediately.
- Test scaffolding: each plugin owns its provenance guard, local dev-server fixtures and pytest hooks. New plugins copy the standard implementations from `python/_template/tests/`; the conventions check keeps `environment.py`, `plugin_meta.py`, `provenance.py`, `test_env.py`, and `test_installed_matches_source.py` byte-identical to their canonical templates. Plugin-specific fixtures stay in that plugin's conftest. Tests must not require real provider credentials; use deterministic local models, mock transports, or in-process servers to exercise provider behavior in CI.

## Java conventions

- Each Java plugin owns a Gradle wrapper with distribution checksum, `build.gradle`,
  `settings.gradle`, `plugin.toml`, and per-Spring-Boot dependency locks. Shared build
  conventions live in `java/_shared/java.gradle`.
- Java development versions are `0.0.0`; a release tag supplies `-PreleaseVersion`
  before tests and builds. Do not publish the placeholder or add snapshot automation.
- Java packages and publication metadata use the root MIT license. Imported source
  and tests stay unchanged until active upstream metadata is removed at ownership handoff.
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

The Spring AI history import is followed by an explicit ownership handoff: remove
active upstream metadata before implementing Spring AI 2 here. Spring AI 1-to-2
workflow-history replay compatibility is outside the upgrade's scope. The candidate
version is `0.1.0-RC1`; workflow streams and OpenTelemetry never migrate here.

## CI

One entry workflow, one reusable workflow per language, plugin as a parameter, no secrets.

- `.github/workflows/ci.yml` (`pull_request`, `merge_group`, push to `main`, nightly, dispatch). Job `changes` runs `scripts/ci/detect_changes.py`: plugins are discovered from `<language>/*/<manifest>` (ignoring `_*`); a changed file under a plugin selects that plugin; a non-plugin file under a language root selects every plugin of that language; `.github/**` and `scripts/ci/**` select everything; `scripts/release/**` and `scripts/migrate/**` select only the script tests; push to `main`, nightly and dispatch select everything. Job `conventions` checks repository invariants and runs the script tests. Job `python` calls `_python-plugin.yml` once per selected plugin. Job `java` calls `_java-plugin.yml` for selected Java plugins. Job `ci-status` fans in and is the only required check (skipped upstream jobs count as success).
- `.github/workflows/_python-plugin.yml`: job `matrix` reads `plugin.toml` `runtime-versions` and emits the same matrix for every run, pull requests included (ubuntu at the min and max versions, macOS and Windows at max); job `test` runs `make sync` (or `sync-latest` on nightly runs), `make lint`, `make test`, then, on the ubuntu/max cell only, the `python-build-check` composite action (`make build`, `check_wheel.py`, isolated `smoke.py` on wheel and sdist). Windows runners install GNU make with choco.
- Dependencies: ordinary CI uses the committed lockfile; local `make sync` uses the existing lockfile. Nightly runs every plugin with the newest allowed dependencies (`sync-latest`) without committing the updated lock; manual dispatch can select that mode too. Shared lint, test and format targets use `uv run --locked` to preserve the dependency versions selected during sync.
- Required checks on `main`: `ci-status`, `Check for CODEOWNERS` and `opengrep/scan` (the last two are org-enforced workflows that run automatically on every PR), plus one approving review from a code owner; `license/cla` joins once the CLA app is installed. Do not add a local opengrep caller; the org one already runs. TRANSITION(sdk-cutover): branch protection, the `testpypi`/`pypi` environments (tag policy `python/*/v*`) and the release-tag ruleset were configured by hand on 2026-09-09. The `pypi` required-reviewer gate was removed on 2026-10-07.
- Nightly failures: `scripts/ci/nightly_report.py` opens or updates one `nightly` issue per failing plugin from the job names `Python (<plugin>) / ...` and `Java (<plugin>) / ...`; `scripts/tests/test_nightly_report.py` fails if `ci.yml` renames those jobs.

## Releases

Trusted publishing by ecosystem: PyPI uses OIDC trusted publishing (`pypa/gh-action-pypi-publish`, no stored token; PyPI cannot bind a reusable workflow, so publish jobs live inline in `release-python.yml`). npm supports OIDC trusted publishing (GitHub-hosted runners, npm >= 11.5.1, one publisher per package, register the calling workflow's filename; provenance is automatic for a public repo and package). Maven Central has no OIDC: Central Portal user token plus GPG signing, kept as environment-scoped secrets. Go has nothing to upload: an immutable tag plus `sum.golang.org` is the release.

Every Python plugin uses the `testpypi` and `pypi` GitHub environments without required reviewers, regardless of maturity. Both environments allow the `python/*/v*` tag pattern and `main` for validated publication recovery. Publishing the GitHub Release starts the release pipeline; eligible versions upload to PyPI automatically after the full test matrix and TestPyPI verification succeed. Trusted publishers remain bound to their existing environment names.

Version policy (`release_tool.py check-version-policy`; version ordering is evaluated against pypi.org, test.pypi.org, or Maven Central's metadata): a coordinate with no published release must start at exactly `1.0.0` (`generally-available`), `0.1.0` (`public-preview`), or `0.0.1` (`pre-release`), pre-releases of that version allowed; an existing coordinate must be strictly greater than its highest published version, yanked releases included. Maturity is independent of PEP 440 version status: a Pre-release plugin can publish a final version such as `0.0.1` to PyPI. TestPyPI versions also move forward. A version already staged on TestPyPI, or already the newest release on pypi.org, only produces a warning (a re-run after an upload is the normal recovery path); an older staged version is rejected. Each Python smoke job proves the index serves exactly the artifacts this run built, none yanked (`verify-index-files`). Java smoke jobs compare staged and public Maven artifacts byte for byte with the tested files. Final versions additionally require `plugin.toml` `[release] allow-final = true` and no `TRANSITION(sdk-cutover)` marker in the plugin.

Runbook for `python/<name>`:
1. Merge every code, dependency and migration change intended for the release. Re-sync from upstream first while the transition rules apply. Do not change the committed `0.0.0` development version.
2. Choose a canonical PEP 440 version and dry-run it on `main`: `gh workflow run release-python.yml --ref main -f tag=python/<name>/v<version>`. The workflow injects the tag version into each checkout, runs policy validation, the full test matrix and the artifact build, but uploads nothing and consumes no tag. `-f skip-publish=true` does the same for a dispatch on an existing tag ref.
3. `git tag -a python/<name>/v<version> -m "python/<name> v<version>"` on the tested `main` commit and push the tag. Tags must match `<language>/<name>/v<version>` and are protected by a tag ruleset. Generate notes with `uv run --project scripts --locked python scripts/release/release_tool.py release-notes --plugin-dir python/<name> --tag python/<name>/v<version> --output /tmp/release-notes.md`, then create a draft GitHub Release for that existing tag and review its notes. If GitHub release assets are wanted, attach the tested distributions downloaded from the successful dry run before publication; the workflow retains its distributions as Actions artifacts.
4. Publish the reviewed GitHub Release using the GitHub UI or `gh release edit python/<name>/v<version> --draft=false`. Mark it as a GitHub pre-release when the plugin's maturity is `pre-release` or the version is a PEP 440 pre-release. Its `release: published` event starts `release-python.yml` for Python tags; pushing a tag alone does not start publication. Publication with a workflow's `GITHUB_TOKEN` does not trigger this event workflow; use a human's session or the manual dispatch fallback on the tag. The GitHub Release is now visible even if package publication later fails.
5. `release-python.yml` validates the main-reachable tag, version policy, published release and GitHub pre-release classification, injects the tag's version, and runs the full test matrix (its ubuntu dist cell builds, checks and smoke-tests the wheel and sdist). It publishes those tested artifacts to TestPyPI (environment `testpypi`), proves TestPyPI serves exactly those files, and smoke-installs from TestPyPI in a clean project. Eligible versions then publish automatically to PyPI (environment `pypi`), with the published release linked from the deployment; the workflow repeats the artifact proof and smoke there. PEP 440 pre-releases go to TestPyPI only unless a tag dispatch uses `publish-prerelease-to-pypi` with `allow-final = true`; GitHub's pre-release flag does not control index routing. The clean-project smoke tolerates overlap during migrations only while `allow-final = false`. The workflow reads the GitHub Release without changing its notes or assets.
6. If a job fails after an upload, use "Re-run failed jobs" on that run: `prepare`'s outputs and the tested artifact survive, the upload is skipped, and the smoke jobs verify the served files. If the tagged workflow or tooling itself needs a repair, merge the repair first, then dispatch on `main` with `-f tag=python/<name>/v<version> -f recover-run=<original-release-run-id>`. Recovery validates that the completed source run is a release-event run (or a legacy tag-push run) for the exact main-reachable tag, passed version validation and every test-matrix job, and retains an unexpired distribution artifact. A published GitHub Release for the tag must exist, and package manifests must still match the tag. It downloads those tested bytes without rebuilding; the normal ref restrictions, registry ordering and file-hash checks still apply. Recovery requires a `main` branch deployment policy in the `testpypi` and `pypi` environments. A fresh dispatch on the tag also passes the version policy (the newest published version is treated as a re-run, with a warning) but rebuilds the artifacts, and `verify-index-files` fails if the rebuild is not byte-identical (a different uv version stamps its `Generator` into the wheel). If the artifacts themselves must change, fix forward with the next `rcN`; uploaded files are immutable and tags are never moved.

Runbook for `java/<name>`:
1. Merge the import (with a merge commit), the Spring AI 2 upgrade, and the release
   pipeline, in that order. Publish `io.temporal:spring-ai` starting at Public
   Preview version `0.1.0` (candidate `0.1.0-RC1`); keep the committed `0.0.0`
   development version. Workflow streams and OpenTelemetry stay in sdk-java.
2. Configure shared environments `maven-central-staging` and `maven-central`, both
   accepting tags `java/*/v*` only. Production requires an `@temporalio/ai-sdk`
   reviewer and prevents self-review. Extend the immutable release-tag ruleset to
   Java without relaxing its Python rules.
   These environments and tag policies were configured on 2026-10-02; credential
   installation remains a maintainer setup step.
3. In staging, install environment secrets `CENTRAL_USERNAME`, `CENTRAL_PASSWORD`
   (a Central Portal user-token pair), `GPG_PRIVATE_KEY` (armored private key or
   sdk-java's base64-encoded secret keyring), and
   `GPG_PASSPHRASE`; set environment variable `GPG_FINGERPRINT` to the full public
   fingerprint. Production needs only the Central token pair, with access to the
   staging deployment. Use credentials from the same publishing account in both.
   The sdk-java equivalents are `RH_USER`, `RH_PASSWORD`, `JAR_SIGNING_KEY`,
   `JAR_SIGNING_KEY_PASSWORD`, and `JAR_SIGNING_KEY_ID` (derive the full fingerprint
   if it contains only a short ID). GitHub cannot export existing secret values;
   an authorized maintainer must install them from the credential source. Publish
   the public key to a supported keyserver before Central validation; see
   [Sonatype's GPG requirements](https://central.sonatype.org/publish/requirements/gpg/).
4. Dry-run the candidate on main:
   `gh workflow run release-java.yml --ref main -f tag=java/spring-ai/v0.1.0-RC1`.
   The full compatibility matrix tests the injected version; its primary
   Ubuntu/max cell builds and verifies the five Maven artifacts and installs a
   clean consumer. A branch dispatch never signs, uploads, or creates a release.
5. Tag the tested main commit with an annotated, immutable tag
   `java/spring-ai/v0.1.0-RC1` and push it. The shared workflow signs the
   tested bytes, adds checksums, uploads a `USER_MANAGED` bundle to Central Portal,
   waits for `VALIDATED`, compares all staged files byte for byte, and installs
   a clean consumer from the authenticated staging endpoint. Candidates remain
   privately staged; they are not public Maven Central releases. A draft GitHub
   release contains generated notes, the tested files, signed bundle, and deployment
   metadata. Review and publish the GitHub draft separately.
6. Final publication additionally requires an agreed SDK cutover plan, removing
   all plugin cutover markers, and setting `allow-final = true`. Publish
   `io.temporal:spring-ai:0.1.0` before sdk-java publishes the one-time relocation
   POM at `io.temporal:temporal-spring-ai:1.41.0`; the template is
   `java/spring-ai/relocation.pom`. Recheck Maven Central for the next available
   old-coordinate version at cutover if sdk-java has released again. Old releases
   remain unchanged. The new pipeline publishes only the new coordinate.
   The production environment approval publishes the already verified deployment;
   the workflow waits for `PUBLISHED`, proves Maven Central serves the same files,
   and installs a clean consumer before drafting a final GitHub release. Do not
   remove these gates merely to run the candidate.
7. After an upload failure, use **Re-run failed jobs**. The workflow preserves the
   signed bundle, upload intent, and deployment ID even if validation or the consumer
   fails. It reuses those signatures and never retries an ambiguous upload POST.
   A fresh publishing dispatch on the same tag requires `-f deployment-id=<Portal UUID>`;
   recovery downloads the existing signed files and rejects any rebuilt-byte
   mismatch or deployment containing additional coordinates. If an upload response
   was lost, inspect Portal for its ID before retrying. Publishing dispatches cannot
   start a new upload, even when Portal is still validating the original upload;
   the initial upload runs on the tag push. A dispatch with `skip-publish=true`
   remains a credential-free dry run and does not require a deployment ID.
   Never move a tag or replace uploaded files: a changed candidate gets the next `RCN`.
   Portal does not expose
   TestPyPI-style staging version enumeration; public ordering is checked against
   Maven Central and an existing validated staged version requires explicit recovery.

The Java pipeline uses the
[Central Portal Publisher API](https://central.sonatype.org/publish/publish-portal-api/)
directly. Maven Central remains the public host. sdk-java publishes the old
coordinate's relocation POM separately after the new package is public. The retired OSSRH service and its compatibility API are unnecessary
for this new pipeline. Snapshot publishing is not configured.

## Migration and re-sync

`scripts/migrate/extract-sdk-python.sh` plus `scripts/migrate/README.md` are the procedure. Only commits reachable from the upstream repository's default branch qualify as imported history. Work from an unmerged or closed PR, feature branch or fork is ordinary local work: do not apply `history-import` and do not add it to `scripts/migrate/IMPORTS.md`. Commit count, `Migrated-From` trailers and use of the migration tooling do not change that classification. For valid imports, find every historical path first; the script is frozen after a plugin's first import; label the PR `history-import` and merge it with a merge commit; keep adaptation files in separate commits on top; and record every import and re-sync in `IMPORTS.md`. Expected verification numbers are in the migration README.

## Cutover sequencing

- Publish `temporalio-openai-agents` 1.0.0 first. Its `temporalio.openai_agents` root does not overlap the SDK's former `temporalio.contrib.openai_agents` files. Python plugins now require Temporal 1.34 or later, which extends the regular `temporalio` package path for split-directory installations. Then make the SDK's `openai-agents` extra forward to the published package, merge the SDK removal and publish `temporalio` 1.34.0.

## Cutover checklist

SDK side: make the `openai-agents` extra depend on `temporalio-openai-agents`, retain explicit compatibility modules under `temporalio.contrib.openai_agents`, add `pkgutil.extend_path` to `temporalio/__init__.py` for split-directory installations (including editable installs), merge the removal PR and publish 1.34.0. Docs: repoint `openai-agents.mdx` and the install text; decide where the plugin API reference is hosted (the SDK's pydoctor site loses these pages). Downstream: update `samples-python` dependency groups.

## Verification commands

```bash
cd python/<name> && make sync && make lint && make test && make build
uv run --project scripts --locked pytest scripts/tests -q # tooling script tests
uv run --project scripts python scripts/ci/check_conventions.py
uv run --project scripts python scripts/ci/detect_changes.py --dry-run --base origin/main
uv run --project scripts python scripts/ci/check_wheel.py --plugin-dir python/<name> --dist python/<name>/dist
```
