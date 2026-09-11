# Python plugins

Conventions specific to `python/`; the normative reference is [`AGENTS.md`](../AGENTS.md).

## Layout

```
python/<name>/
├── pyproject.toml  uv.lock  plugin.toml  Makefile  README.md  LICENSE
├── src/temporalio/contrib/<name>/        # NO __init__.py in src/temporalio or src/temporalio/contrib
└── tests/                                # conftest.py, helpers/, and the plugin's tests
```

`src/temporalio/` and `src/temporalio/contrib/` must never contain `__init__.py`: the SDK wheel
owns those packages and the plugin installs into them. `scripts/ci/check_wheel.py` enforces this.

## Make targets

Every plugin's `Makefile` is two lines that include `python/_shared/python.mk`; `make help`
lists the targets. The important ones: `sync` (non-editable install; run it after pulling),
`lint`, `test`, and `build`.

Why non-editable: `temporalio` is a regular package, so an editable install of a plugin cannot be
imported as `temporalio.contrib.<name>`. The make targets export `UV_NO_EDITABLE=1`, and the
`[tool.uv] cache-keys` in each `pyproject.toml` make `uv` rebuild the plugin when `src/` changes.
A test session that finds an editable or stale install fails immediately with the fix in the
message (`make sync`).

Shared pytest support is a separate tooling project. Run `cd python/_shared && make sync` to
create its `.venv` with pytest, pytest-asyncio and Temporal installed; editors can use that
interpreter for files under `_shared`. Its `make lint` runs once in the conventions job.

For VS Code and other Pyright-based editors, open `python/ai-integrations-python.code-workspace`.
Each Python project is then a separate workspace folder, so the language server uses that
project's `.venv`; a single-folder repository workspace cannot represent these independent
environments.

## Provider-independent tests

Tests must not require provider credentials. Exercise provider behavior with deterministic local
models, mock transports, or in-process servers so the same assertions run in every environment.

## Dependency cooldown

`exclude-newer = "2 weeks"` applies to third-party packages; `exclude-newer-package =
{ temporalio = false }` exempts the SDK so a new release is adoptable immediately. Nightly CI
lanes re-lock to the newest and lowest allowed versions without committing the lock.

## LICENSE

Every plugin directory carries a committed regular-file copy of the root `LICENSE`, because each
wheel and sdist must ship the license text. The copy must stay byte-identical to the root file (the
conventions check enforces it; refresh with `cp LICENSE python/<name>/LICENSE`).

## Platform notes

- Windows: install GNU make (`choco install make`) or use WSL; CI installs it on Windows runners.
- The dev server is downloaded on first run by the test suite (`_shared/temporalio_ai_integrations_pytest.py` pins the version).
