# Shared .NET targets, included from each plugin directory after PACKAGE is set:
#     PACKAGE := Temporalio.MicrosoftAgentFramework
#     include ../_shared/dotnet.mk
# CI runs these same targets (.github/workflows/_dotnet-plugin.yml), so there is one
# definition of "lint" and "test". See AGENTS.md (".NET conventions").
ifndef PACKAGE
$(error PACKAGE must be set to the NuGet package id (e.g. Temporalio.MicrosoftAgentFramework) before including dotnet.mk)
endif

DOTNET ?= dotnet
SOLUTION := $(PACKAGE).slnx
LIBRARY := src/$(PACKAGE)/$(PACKAGE).csproj
TESTS := tests/$(PACKAGE).Tests/$(PACKAGE).Tests.csproj
CONFIGURATION ?= Release
RELEASE_VERSION ?= 0.0.0
TEST_ARGS ?=
# Package entry timestamps come from the commit under test, not from checkout mtimes, so a
# rebuild of the same commit on the same SDK version is byte-identical (a servicing patch
# ships a new compiler and changes the assembly).
DETERMINISTIC_TIMESTAMP ?= $(shell git log -1 --pretty=%ct 2>/dev/null)
# Normalised source paths are needed for reproducible packages but break local debugging.
CI_BUILD ?= $(if $(GITHUB_ACTIONS),true,false)
BUILD_PROPS := -p:Version=$(RELEASE_VERSION) -p:ContinuousIntegrationBuild=$(CI_BUILD) $(if $(DETERMINISTIC_TIMESTAMP),-p:DeterministicTimestamp=$(DETERMINISTIC_TIMESTAMP))

export DOTNET_CLI_TELEMETRY_OPTOUT := 1
export DOTNET_NOLOGO := 1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE := 1
export DOTNET_GENERATE_ASPNET_CERTIFICATE := false
# Every implicit restore (build, test, pack, format) honours the committed packages.lock.json files.
export RestoreLockedMode := true

.DEFAULT_GOAL := help
.PHONY: help sync sync-latest lock check-deps format lint test build clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

sync: ## Restore the committed packages.lock.json files
	$(DOTNET) restore $(SOLUTION) --locked-mode

# `dotnet package update` exits 2 when every reference is already current; an update that cannot
# resolve fails the forced re-evaluation that follows.
sync-latest: export RestoreLockedMode := false
sync-latest: ## Update every direct dependency to its newest version and re-lock (nightly lane; not committed)
	$(DOTNET) package update --project $(SOLUTION) || [ $$? -eq 2 ]
	$(DOTNET) restore $(SOLUTION) --force-evaluate

lock: export RestoreLockedMode := false
lock: ## Re-resolve and rewrite packages.lock.json after editing a csproj
	$(DOTNET) restore $(SOLUTION) --force-evaluate

check-deps: ## Dependency cooldown: every locked version is older than nuget.config's minPublishAgeHours
	uv run --project ../../scripts --locked python ../../scripts/ci/check_dotnet_deps.py --plugin-dir .

format: ## Apply formatting and style fixes
	$(DOTNET) format $(SOLUTION)

lint: ## Formatting, style and analyzer fixers would change nothing
	$(DOTNET) format $(SOLUTION) --verify-no-changes

test: ## Build in Release with the release version and run the suite under Microsoft.Testing.Platform
	$(DOTNET) test --solution $(SOLUTION) -c $(CONFIGURATION) $(BUILD_PROPS) --results-directory TestResults --minimum-expected-tests 1 -- --report-trx $(TEST_ARGS)

build: ## Pack the tested build into dist/ (nupkg and snupkg)
	rm -rf dist
	$(DOTNET) build $(LIBRARY) -c $(CONFIGURATION) $(BUILD_PROPS)
	$(DOTNET) pack $(LIBRARY) -c $(CONFIGURATION) --no-build $(BUILD_PROPS) -o dist

clean: ## Remove build, test and package output
	rm -rf dist TestResults src/*/bin src/*/obj tests/*/bin tests/*/obj
