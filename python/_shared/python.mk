# Shared make targets for every Python plugin in this repository.
#
# A plugin's Makefile is exactly two lines:
#     DIST := temporalio-<name>
#     include ../_shared/python.mk
#
# Every target is a plain `uv` command. CI runs these same targets
# (.github/workflows/_python-plugin.yml), so there is one definition of "lint" and "test".
# See AGENTS.md ("Python conventions") for the reasoning behind each target.

ifndef DIST
$(error DIST must be set to the plugin's distribution name (e.g. temporalio-openai-agents) before including python.mk)
endif

# `temporalio` is a regular package owned by the SDK wheel, so an editable install of this
# plugin cannot reliably extend it with another package root. Every uv command below runs non-editable.
export UV_NO_EDITABLE := 1

PYTEST_ARGS ?=
# Keep the dependency versions selected by sync or sync-latest through all tool runs.
UV_RUN := uv run --locked
PYTEST := $(UV_RUN) pytest -n auto --dist=worksteal

.PHONY: help sync sync-latest format lint test build clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Reinstall the plugin last so its files win while a plugin is being migrated from a distribution
# that ships the same namespace paths. Keeping this shared behavior makes future migrations safe.
sync: ## Install locked dependencies and this plugin (non-editable); creates uv.lock on first run
	@test -f uv.lock || uv lock
	uv sync --locked
	uv sync --locked --reinstall-package $(DIST)

sync-latest: ## Re-lock to the newest allowed versions and install (nightly lane; lock not committed)
	uv lock --upgrade
	uv sync --locked
	uv sync --locked --reinstall-package $(DIST)

format: ## Fix import order and formatting
	$(UV_RUN) ruff check --select I --fix
	$(UV_RUN) ruff format

lint: ## Import order, formatting, pyright, mypy, basedpyright, docstrings
	$(UV_RUN) ruff check --select I
	$(UV_RUN) ruff format --check
	$(UV_RUN) pyright
	$(UV_RUN) mypy
	$(UV_RUN) basedpyright
	$(UV_RUN) pydocstyle --ignore-decorators=overload src

test: ## Run the suite against a local dev server
	$(PYTEST) $(PYTEST_ARGS)

build: ## Build sdist and wheel into dist/
	rm -rf dist
	uv build --no-sources --out-dir dist

clean: ## Remove build and test artifacts and the virtualenv
	rm -rf dist .venv junit
