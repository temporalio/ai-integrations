# Shared test launcher for plugins using an external Temporal dev server.
TEMPORAL_TEST_PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)
TEMPORAL_TEST := $(TEMPORAL_TEST_PYTHON) "$(dir $(lastword $(MAKEFILE_LIST)))run_temporal_tests.py"
