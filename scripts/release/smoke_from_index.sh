#!/usr/bin/env bash
# Install a just-published distribution from a package index into a brand-new uv
# project (no repository paths on sys.path) and run the stdlib-only smoke test.
#
# Env: COORDINATE, VERSION, ROOT_API (required); SMOKE_IMPORTS (comma list);
#      INDEX_URL (optional; when set, ONLY $COORDINATE is resolved from that index,
#      every dependency still comes from PyPI); ALLOW_OVERLAP_WITH_CORE.
set -euo pipefail
: "${COORDINATE:?}" "${VERSION:?}" "${ROOT_API:?}"
workspace=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
proj=$(mktemp -d)
cat > "$proj/pyproject.toml" <<TOML
[project]
name = "smoke"
version = "0"
requires-python = ">=3.10"
dependencies = ["${COORDINATE}==${VERSION}"]
TOML
if [ -n "${INDEX_URL:-}" ]; then
  cat >> "$proj/pyproject.toml" <<TOML

[[tool.uv.index]]
name = "release-index"
url = "${INDEX_URL}"
explicit = true

[tool.uv.sources]
"${COORDINATE}" = { index = "release-index" }
TOML
fi
cat "$proj/pyproject.toml"
# Newly uploaded files take a little while to appear on the simple index; retry.
for attempt in $(seq 1 10); do
  if (cd "$proj" && UV_LINK_MODE=copy uv sync --no-cache); then break; fi
  if [ "$attempt" -eq 10 ]; then echo "::error::could not install ${COORDINATE}==${VERSION} from the index"; exit 1; fi
  echo "attempt $attempt failed; waiting for index propagation"; sleep 30
done
args=(--in-env --coordinate "$COORDINATE" --root-api "$ROOT_API")
[ -n "${SMOKE_IMPORTS:-}" ] && args+=(--imports "$SMOKE_IMPORTS")
EXPECTED_VERSION="$VERSION" ALLOW_OVERLAP_WITH_CORE="${ALLOW_OVERLAP_WITH_CORE:-0}" \
  uv run --project "$proj" --no-sync python "$workspace/scripts/ci/smoke.py" "${args[@]}"
