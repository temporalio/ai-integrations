#!/usr/bin/env bash
# Extract one plugin's history from temporalio/sdk-python into a throwaway clone whose `main`
# contains only that plugin's files, already renamed into this repository's layout.
#
# The output is meant to be merged with `git merge --allow-unrelated-histories --no-ff`; the exact
# commands are printed at the end. See scripts/migrate/README.md before running.
#
# THIS SCRIPT IS FROZEN AFTER A PLUGIN'S FIRST IMPORT. Re-syncs must produce byte-identical
# rewritten commits so that `git merge` only brings new upstream commits. Any change to the
# --path/--path-rename rules, the message rewriting, the callback, or the git-filter-repo version
# rewrites every SHA and turns the next re-sync into a full re-import.
#
# Environment:
#   PLUGIN    plugin folder name (default: openai_agents)
#   SRC_REF   upstream ref or SHA to import (default: main)
#   SRC_REPO  upstream clone URL (default: https://github.com/temporalio/sdk-python.git)
set -euo pipefail

PLUGIN="${PLUGIN:-openai_agents}"
SRC_REF="${SRC_REF:-main}"
SRC_REPO="${SRC_REPO:-https://github.com/temporalio/sdk-python.git}"
FILTER_REPO_VERSION=2.47.0

REPO="$(git rev-parse --show-toplevel)"
REPLACE_MESSAGE="$REPO/scripts/migrate/replace-message.txt"
P="python/$PLUGIN"

# Path archaeology per plugin: every path the plugin's files EVER lived at upstream, found with
#   git log --all --diff-filter=R --name-status -- <current paths>
#   git log --all --name-only --format= -- <current paths> | sort -u
# List every --path before every --path-rename (filters are evaluated against renamed names).
# Rename the README out of the package directory BEFORE renaming the directory.
case "$PLUGIN" in
  openai_agents)
    PATH_ARGS=(
      --path temporalio/contrib/openai_agents/
      --path tests/contrib/openai_agents/
      --path tests/contrib/test_openai.py
      --path tests/contrib/research_agents/
    )
    RENAME_ARGS=(
      --path-rename "temporalio/contrib/openai_agents/README.md:$P/README.md"
      --path-rename "temporalio/contrib/openai_agents/:$P/src/temporalio/contrib/openai_agents/"
      --path-rename "tests/contrib/openai_agents/:$P/tests/contrib/openai_agents/"
      --path-rename "tests/contrib/test_openai.py:$P/tests/contrib/openai_agents/test_openai.py"
      --path-rename "tests/contrib/research_agents/:$P/tests/contrib/openai_agents/research_agents/"
    )
    ;;
  *)
    echo "error: no path archaeology recorded for PLUGIN=$PLUGIN; add a case to $0 (see README.md)" >&2
    exit 2
    ;;
esac

WORK="$(mktemp -d)"
echo "cloning $SRC_REPO into $WORK/src"
git clone --quiet --no-local --no-tags --single-branch --branch main "$SRC_REPO" "$WORK/src"
cd "$WORK/src"
if [ "$SRC_REF" != "main" ]; then
  git reset --quiet --hard "$SRC_REF"
fi
SRC_SHA="$(git rev-parse HEAD)"
echo "source: temporalio/sdk-python@$SRC_SHA"

# --force: only because the optional SRC_REF pin breaks git-filter-repo's fresh-clone check; the
#          clone is throwaway. --prune-empty always: upstream has originally-empty commits (the 2022
#          "Initial commit", a dependabot bump) that would otherwise survive as unrelated roots.
uvx --from "git-filter-repo==$FILTER_REPO_VERSION" git-filter-repo \
  --force --preserve-commit-hashes --prune-empty always \
  "${PATH_ARGS[@]}" \
  "${RENAME_ARGS[@]}" \
  --replace-message "$REPLACE_MESSAGE" \
  --commit-callback '
lines = commit.message.rstrip(b"\n").split(b"\n")
sep = b"\n" if len(lines) > 1 and re.match(rb"^[A-Za-z][A-Za-z-]*: ", lines[-1]) else b"\n\n"
commit.message = b"\n".join(lines) + sep + b"Migrated-From: temporalio/sdk-python@" + commit.original_id + b"\n"'

COMMITS="$(git rev-list --count HEAD)"
IDENTITIES="$(git shortlog -sne HEAD | wc -l | tr -d ' ')"
echo
echo "SRC_SHA=$SRC_SHA"
echo "FILTER_REPO=$FILTER_REPO_VERSION"
echo "WORK=$WORK/src"
echo "commits=$COMMITS identities=$IDENTITIES tags=$(git tag | wc -l | tr -d ' ')"
cat <<MSG

Next, in $REPO (on an import branch):
  git remote add sdk-python-filtered "$WORK/src"
  git fetch --no-tags sdk-python-filtered main
  git merge --allow-unrelated-histories --no-ff \\
    -m "Import temporalio.contrib.$PLUGIN from temporalio/sdk-python@$SRC_SHA (history preserved; git-filter-repo $FILTER_REPO_VERSION)" \\
    sdk-python-filtered/main
  git remote remove sdk-python-filtered
Then append a row to scripts/migrate/IMPORTS.md and open a PR labelled history-import (merge commit, never squash).
MSG
