#!/usr/bin/env bash
# Extract one plugin's default-branch history from temporalio/sdk-go into a
# throwaway clone, renamed into this repository's layout. See README.md.
#
# FROZEN AFTER THE FIRST IMPORT: changing filters, renames, message rewriting,
# the callback, or the filter-repo version changes every rewritten commit SHA.
#
# Environment:
#   PLUGIN    plugin folder name (default: googleadk)
#   SRC_REF   commit reachable from upstream main (default: main)
#   SRC_REPO  upstream clone URL (default: https://github.com/temporalio/sdk-go.git)
set -euo pipefail

PLUGIN="${PLUGIN:-googleadk}"
SRC_REF="${SRC_REF:-main}"
SRC_REPO="${SRC_REPO:-https://github.com/temporalio/sdk-go.git}"
FILTER_REPO_VERSION=2.47.0
REPO="$(git rev-parse --show-toplevel)"

case "$PLUGIN" in
  googleadk)
    # All files originated under contrib/googleadk in sdk-go@6adaffdaef7cc18dc922cc53bfe1f55852e49d1c.
    # Full main-branch path archaeology at b7c1605ccd85e18d5bb2bb153c7b4077f2f4aa4b
    # found no earlier locations or renames outside this directory.
    PATH_ARGS=(--path contrib/googleadk/)
    RENAME_ARGS=(--path-rename "contrib/googleadk/:go/googleadk/")
    ;;
  *)
    echo "error: no path archaeology recorded for PLUGIN=$PLUGIN; see scripts/migrate/README.md" >&2
    exit 2
    ;;
esac

WORK="$(mktemp -d)"
echo "cloning $SRC_REPO into $WORK/src"
git clone --quiet --no-local --no-tags --single-branch --branch main "$SRC_REPO" "$WORK/src"
cd "$WORK/src"
SRC_SHA="$(git rev-parse --verify "$SRC_REF^{commit}")"
if ! git merge-base --is-ancestor "$SRC_SHA" refs/remotes/origin/main; then
  echo "error: SRC_REF=$SRC_REF is not reachable from upstream main; not a history-import input" >&2
  exit 2
fi
git reset --quiet --hard "$SRC_SHA"
echo "source: temporalio/sdk-go@$SRC_SHA"

cat > "$WORK/replace-message.txt" <<'RULE'
regex:(?<![\w/#])#(\d+)\b==>temporalio/sdk-go#\1
RULE

# The clone is disposable; --force allows an explicit SRC_REF pin. Keep
# originally empty upstream commits out of the plugin history and retain SHAs
# mentioned in upstream messages. Add provenance beside existing trailers.
uvx --from "git-filter-repo==$FILTER_REPO_VERSION" git-filter-repo \
  --force --preserve-commit-hashes --prune-empty always \
  "${PATH_ARGS[@]}" \
  "${RENAME_ARGS[@]}" \
  --replace-message "$WORK/replace-message.txt" \
  --commit-callback '
lines = commit.message.rstrip(b"\n").split(b"\n")
sep = b"\n" if len(lines) > 1 and re.match(rb"^[A-Za-z][A-Za-z-]*: ", lines[-1]) else b"\n\n"
commit.message = b"\n".join(lines) + sep + b"Migrated-From: temporalio/sdk-go@" + commit.original_id + b"\n"'

COMMITS="$(git rev-list --count HEAD)"
IDENTITIES="$(git shortlog -sne HEAD | wc -l | tr -d ' ')"
echo
echo "SRC_SHA=$SRC_SHA"
echo "FILTER_REPO=$FILTER_REPO_VERSION"
echo "WORK=$WORK/src"
echo "commits=$COMMITS identities=$IDENTITIES tags=$(git tag | wc -l | tr -d ' ')"
cat <<MSG

Next, in $REPO (on an import branch):
  git remote add sdk-go-filtered "$WORK/src"
  git fetch --no-tags sdk-go-filtered main
  git merge --allow-unrelated-histories --no-ff \\
    -m "Import $PLUGIN from temporalio/sdk-go@$SRC_SHA (history preserved; git-filter-repo $FILTER_REPO_VERSION)" \\
    sdk-go-filtered/main
  git remote remove sdk-go-filtered
Then append a row to scripts/migrate/IMPORTS.md and use history-import on the PR (merge commit, never squash).
MSG
