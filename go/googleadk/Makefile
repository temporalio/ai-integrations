include ../_shared/go.mk

# go.mod requires ADK v2.3.0, which calls a toolset's Tools once per agent
# run. ADK v2.6.0 and later call it before every model step, and the MCP
# toolset must list the same way with both. So sync and test also run the MCP
# tests with go.adk-v2.6.mod, which pins ADK v2.6.0. After changing go.mod, run
# `make adk-v2.6-mod` to rebuild that file; sync fails until it matches.
ADK_V2_6 = -modfile=go.adk-v2.6.mod
# -modfile does not work in workspace mode.
ADK_V2_6_GO = GOWORK=off $(GO)
# adk_v2_6_mod writes $(1).mod and $(1).sum: go.mod with ADK v2.6.0.
adk_v2_6_mod = cp go.mod $(1).mod && cp go.sum $(1).sum && \
	$(ADK_V2_6_GO) get -modfile=$(1).mod google.golang.org/adk/v2@v2.6.0 && \
	$(ADK_V2_6_GO) mod tidy -modfile=$(1).mod

.PHONY: adk-v2.6-mod sync-adk-v2.6 test-adk-v2.6
sync: sync-adk-v2.6
test: test-adk-v2.6

adk-v2.6-mod:
	$(call adk_v2_6_mod,go.adk-v2.6)

sync-adk-v2.6:
	$(ADK_V2_6_GO) mod download $(ADK_V2_6)
	$(ADK_V2_6_GO) mod verify $(ADK_V2_6)
	@d=$$(mktemp -d); \
	if ! ( $(call adk_v2_6_mod,$$d/go) ) > $$d/log 2>&1; then cat $$d/log; rm -rf $$d; exit 1; fi; \
	if ! cmp -s $$d/go.mod go.adk-v2.6.mod || ! cmp -s $$d/go.sum go.adk-v2.6.sum; then \
		rm -rf $$d; echo "go.adk-v2.6.mod is out of date with go.mod: run make adk-v2.6-mod"; exit 1; \
	fi; \
	rm -rf $$d

test-adk-v2.6:
	$(ADK_V2_6_GO) test $(ADK_V2_6) -count=1 -timeout=10m -run 'Mcp|MCP' $(TEST_ARGS) ./...
