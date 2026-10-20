"""Bazel entry point for the native Python tooling test suite."""

import sys
from pathlib import Path

import pytest

sys.exit(pytest.main([str(Path(__file__).parent / sys.argv[1]), "-q", *sys.argv[2:]]))
