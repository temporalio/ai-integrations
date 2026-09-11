.PHONY: sync format lint

sync: ## Install the locked shared test-tooling environment
	@test -f uv.lock || uv lock
	uv sync --locked

format: ## Fix import order and formatting
	uv run ruff check --select I --fix
	uv run ruff format

lint: ## Check shared test tooling once
	uv run ruff check --select I
	uv run ruff format --check
	uv run mypy
	uv run basedpyright
