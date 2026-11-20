# Each plugin installs and locks its own published dependencies.
PNPM ?= pnpm
TEST_ARGS ?=

.PHONY: sync sync-latest sync-lowest check-deps lint test build clean help
sync:
	$(PNPM) install --frozen-lockfile

sync-latest:
	$(PNPM) update --latest=false

# Install every peer dependency at the lowest version the plugin allows (nightly lane; nothing committed).
# Earlier build output goes first, or `tsc --build` reuses it instead of compiling against those versions.
sync-lowest:
	node ../_shared/pin-lowest-peers.mjs
	node -e "for (const path of ['lib', 'tsconfig.tsbuildinfo', 'tests/helpers/lib', 'tests/helpers/tsconfig.tsbuildinfo']) require('node:fs').rmSync(path, {recursive: true, force: true})"
	$(PNPM) install --no-frozen-lockfile

# Resolve without the committed lock, preserving the manifest and node_modules.
check-deps:
	@set -eu; \
	lock_dir=$$(mktemp -d); \
	trap 'rm -rf "$$lock_dir"' EXIT; \
	$(PNPM) install --lockfile-only --ignore-scripts --no-frozen-lockfile --lockfile-dir "$$lock_dir"

lint:
	$(PNPM) run lint

test:
	$(PNPM) run build
	$(PNPM) run test $(TEST_ARGS)

build:
	$(PNPM) run build
	$(PNPM) run pack

clean:
	$(PNPM) exec tsc --build --clean
	node -e "for (const path of ['lib', 'dist', 'tests/helpers/lib']) require('node:fs').rmSync(path, {recursive: true, force: true})"

help:
	@echo "sync sync-latest sync-lowest check-deps lint test build clean"
