# Python plugins

Conventions specific to `python/`; the normative reference is [`AGENTS.md`](../AGENTS.md).

## Layout

```
python/<name>/
├── pyproject.toml  uv.lock  plugin.toml  Makefile  README.md  LICENSE -> ../../LICENSE
├── src/temporalio/contrib/<name>/        # NO __init__.py in src/temporalio or src/temporalio/contrib
└── tests/                                # conftest.py, helpers/, and the plugin's tests
```

`src/temporalio/` and `src/temporalio/contrib/` must never contain `__init__.py`: the SDK wheel
owns those packages and the plugin installs into them. `scripts/ci/check_wheel.py` enforces this.

## Make targets

Every plugin's `Makefile` is two lines that include `python/_shared/python.mk`; `make help`
lists the targets. The important ones: `sync` (non-editable install; run it after pulling),
`lint`, `test`, `build`, `record`.

Why non-editable: `temporalio` is a regular package, so an editable install of a plugin cannot be
imported as `temporalio.contrib.<name>`. The make targets export `UV_NO_EDITABLE=1`, and the
`[tool.uv] cache-keys` in each `pyproject.toml` make `uv` rebuild the plugin when `src/` changes.
A test session that finds an editable or stale install fails immediately with the fix in the
message (`make sync`).

## Offline tests and cassettes

Tests never use a real API key, in CI or locally. Upstream tests that call a provider API are
listed in `plugin.toml` `[offline] skips` (by function name or parametrized id) and are skipped;
their offline twins, which use the SDK's test models, run. `dummy-env` in the same table holds
placeholder values (never real secrets) exported during replay so upstream skip guards do not skip.

Recording real traffic into cassettes is available as an opt-in for teams that want regression
coverage of real provider responses: `tests/conftest.py` replays cassettes from
`tests/contrib/<name>/cassettes/<module>/` with vcrpy (through pytest-recording), and
`OPENAI_API_KEY=<real key> make record` records them locally (`make record` refuses to run in CI).
A `CannotOverwriteExistingCassetteException` means a test made a request that has neither a
cassette nor a skip entry.

## Dependency cooldown

`exclude-newer = "2 weeks"` applies to third-party packages; `exclude-newer-package =
{ temporalio = false }` exempts the SDK so a new release is adoptable immediately. Nightly CI
lanes re-lock to the newest and lowest allowed versions without committing the lock.

## LICENSE

The repository has one `LICENSE` at the root and nothing committed in plugin directories. Because
the wheel and sdist must carry the license text and uv_build fails when a declared license file is
missing, `make` copies the root file into the plugin directory (gitignored) before any command that
may build. Run `make sync` once in a fresh clone before using `uv` directly.

## Platform notes

- Node (`npx`) is required by the MCP tests; GitHub-hosted runners have it.
- Windows: install GNU make (`choco install make`) or use WSL; CI installs it on Windows runners.
- The dev server is downloaded on first run by the test suite (`tests/__init__.py` pins the version).
