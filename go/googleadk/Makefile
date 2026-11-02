include ../_shared/go.mk

# go.mod requires ADK v2.3.0, which calls a toolset's Tools once per agent
# run. ADK v2.6.0 and later call it before every model step, and the MCP
# toolset must list the same way with both. So sync and test also run the MCP
# tests with go.adk-v2.6.mod, which pins ADK v2.6.0. (ADK v2.7.0 needs Go
# 1.26.10, above this plugin's Go floor.)
ADK_V2_6 = -modfile=go.adk-v2.6.mod

.PHONY: sync-adk-v2.6 test-adk-v2.6
sync: sync-adk-v2.6
test: test-adk-v2.6

sync-adk-v2.6:
	$(GO) mod download $(ADK_V2_6)
	$(GO) mod verify $(ADK_V2_6)

test-adk-v2.6:
	$(GO) test $(ADK_V2_6) -count=1 -timeout=10m $(TEST_ARGS) -run 'Mcp|MCP' ./...
