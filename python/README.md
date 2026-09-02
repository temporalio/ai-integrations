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
`lint`, `test`, `test-time-skipping`, `build`, `record`.

Why non-editable: `temporalio` is a regular package, so an editable install of a plugin cannot be
imported as `temporalio.contrib.<name>`. The make targets export `UV_NO_EDITABLE=1`, and the
`[tool.uv] cache-keys` in each `pyproject.toml` make `uv` rebuild the plugin when `src/` changes.
A test session that finds an editable or stale install fails immediately with the fix in the
message (`make sync`).

## Offline tests and cassettes

Tests never need a real API key. `tests/conftest.py` records and replays HTTP traffic with vcrpy
(through pytest-recording); cassettes live in `tests/contrib/<name>/cassettes/<module>/`. If a
test fails with `CannotOverwriteExistingCassetteException`, it made a request that has no
recording: run `OPENAI_API_KEY=<real key> make record` locally, review the new cassette for
secrets (the conventions check also scans for them), and commit it. `make record` refuses to run
in CI. Plugin-specific offline settings are data in `plugin.toml` `[offline]`: `dummy-env` holds placeholder
values (never real secrets) exported during replay so upstream skip guards do not skip, and `skips`
lists tests that cannot replay, with a reason.

## Dependency cooldown

`exclude-newer = "2 weeks"` applies to third-party packages; `exclude-newer-package =
{ temporalio = false }` exempts the SDK so a new release is adoptable immediately. Nightly CI
lanes re-lock to the newest and lowest allowed versions without committing the lock.

## LICENSE

Each plugin has a committed relative symlink to the root `LICENSE`, never a copy. On Windows
checkouts without symlink support the link becomes a stub file; that is harmless for development
but is why release builds run only on Linux and `check_wheel.py` verifies the wheel's LICENSE.

## Platform notes

- Node (`npx`) is required by the MCP tests; GitHub-hosted runners have it.
- Windows: install GNU make (`choco install make`) or use WSL; CI installs it on Windows runners.
- The dev server is downloaded on first run by the test suite (`tests/__init__.py` pins the version).
