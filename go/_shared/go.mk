# Shared Go targets, invoked from each plugin directory. Keep the imported
# module's dependency versions; go.mod and go.sum are owned by that plugin.
GO ?= go
TEST_ARGS ?=
GOFLAGS += -mod=readonly
export GOFLAGS
.DEFAULT_GOAL := help

.PHONY: help sync sync-latest lint test build

help:
	@echo "sync   Download and verify the committed module dependencies"
	@echo "sync-latest   Upgrade compatible dependencies for nightly checks"
	@echo "lint   Check formatting and run go vet"
	@echo "test   Run all tests, including local Temporal dev-server coverage"
	@echo "build  Build all packages"

sync:
	$(GO) mod download
	$(GO) mod verify

# These updates stay in the CI checkout; do not commit nightly dependency drift.
sync-latest:
	$(GO) get -u -t ./...
	$(GO) mod tidy
	$(GO) mod verify

lint:
	@test -z "$$(gofmt -l .)" || { gofmt -l .; exit 1; }
	$(GO) vet ./...

test:
	$(GO) test -count=1 -timeout=10m $(TEST_ARGS) ./...

build:
	$(GO) build ./...
