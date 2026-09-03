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
# plugin cannot be imported as temporalio.contrib.<name>. Every uv command below runs non-editable.
export UV_NO_EDITABLE := 1

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
PYTEST_ARGS ?=
PYTEST := uv run pytest -n auto --dist=worksteal

.PHONY: help sync sync-latest sync-lowest format lint test build record clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# TRANSITION(sdk-cutover): the second sync reinstalls this plugin last so its files win the
# overlap with temporalio<=1.32, which still ships the same module. Delete it at cutover.
sync: ## Install locked dependencies and this plugin (non-editable); creates uv.lock on first run
	@test -f uv.lock || uv lock
	uv sync --locked
	uv sync --locked --reinstall-package $(DIST)

sync-latest: ## Re-lock to the newest allowed versions and install (nightly lane; lock not committed)
	uv lock --upgrade
	uv sync
	uv sync --reinstall-package $(DIST)

sync-lowest: ## Re-lock to the lowest allowed direct versions and install (nightly lane; lock not committed)
	uv lock --upgrade --resolution lowest-direct
	uv sync
	uv sync --reinstall-package $(DIST)

format: ## Fix import order and formatting
	uv run ruff check --select I --fix
	uv run ruff format

lint: ## Import order, formatting, pyright, mypy, basedpyright, docstrings
	uv run ruff check --select I
	uv run ruff format --check
	uv run pyright
	uv run mypy
	uv run basedpyright
	uv run pydocstyle --ignore-decorators=overload src

test: ## Run the suite against a local dev server (offline: HTTP is replayed from cassettes)
	$(PYTEST) $(PYTEST_ARGS)

build: ## Build sdist and wheel into dist/
	rm -rf dist
	uv build --no-sources --out-dir dist

record: ## Re-record HTTP cassettes with a real OPENAI_API_KEY (local only, never in CI)
	@if [ -n "$$CI" ]; then echo "record: refusing to run in CI" >&2; exit 1; fi
	@if [ -z "$$OPENAI_API_KEY" ] || [ "$$OPENAI_API_KEY" = "sk-cassette-replay" ]; then echo "record: export a real OPENAI_API_KEY first" >&2; exit 1; fi
	RECORD_MODE=$${RECORD_MODE:-once} uv run pytest -n 0 $(PYTEST_ARGS)

clean: ## Remove build and test artifacts and the virtualenv
	rm -rf dist .venv junit
