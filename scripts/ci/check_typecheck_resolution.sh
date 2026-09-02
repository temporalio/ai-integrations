#!/usr/bin/env bash
# One-time proof that pyright and mypy resolve `temporalio.contrib.<name>` from
# the plugin's src/ tree rather than from the copy installed in site-packages.
#
# Usage: scripts/ci/check_typecheck_resolution.sh python/<plugin>
# Creates a throwaway module that exists ONLY in src/, a test that imports it,
# runs both type checkers on that test, and always cleans up.
set -euo pipefail
plugin_dir=${1:?usage: $0 python/<plugin>}
cd "$plugin_dir"
name=$(basename "$PWD")
probe="src/temporalio/contrib/$name/_resolution_probe.py"
test_file="tests/test__resolution_probe.py"
cleanup() { rm -f "$probe" "$test_file"; }
trap cleanup EXIT
[ -e "$probe" ] && { echo "refusing to overwrite existing $probe"; exit 2; }
printf 'RESOLUTION_PROBE = "src"\n' > "$probe"
cat > "$test_file" <<PY
from temporalio.contrib.${name}._resolution_probe import RESOLUTION_PROBE


def test_resolution_probe() -> None:
    assert RESOLUTION_PROBE == "src"
PY
status=0
echo "== pyright"; uv run pyright "$test_file" || status=1
echo "== mypy";    uv run mypy "$test_file"    || status=1
if [ "$status" -eq 0 ]; then
  echo "PASS: type checkers resolve temporalio.contrib.$name from src/"
else
  echo "FAIL: a type checker resolved temporalio.contrib.$name from site-packages; add executionEnvironments/mypy_path pointing at src/"
fi
exit "$status"
