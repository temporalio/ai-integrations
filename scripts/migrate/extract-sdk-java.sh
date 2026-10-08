#!/usr/bin/env bash
# Extract Spring AI's default-branch history into a disposable SDK clone.
# Frozen after the first import: paths, callbacks and filter-repo version must
# remain identical so subsequent extractions produce the same commit IDs.
set -euo pipefail

SRC_REF="${SRC_REF:-main}"
SRC_REPO="${SRC_REPO:-https://github.com/temporalio/sdk-java.git}"
FILTER_REPO_VERSION=2.47.0
REPO="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
P=java/temporal-spring-ai

git clone --quiet --no-local --no-tags --single-branch --branch main "$SRC_REPO" "$WORK/src"
cd "$WORK/src"
git merge-base --is-ancestor "$SRC_REF" origin/main || {
  echo 'error: source must be reachable from the upstream default branch' >&2
  exit 1
}
git reset --quiet --hard "$SRC_REF"
SRC_SHA="$(git rev-parse HEAD)"
uvx --from "git-filter-repo==$FILTER_REPO_VERSION" git-filter-repo \
  --force --preserve-commit-hashes --prune-empty always \
  --path temporal-spring-ai/ --path contrib/temporal-spring-ai/ \
  --path-rename "temporal-spring-ai/README.md:$P/_upstream/README.md" \
  --path-rename "temporal-spring-ai/build.gradle:$P/_upstream/build.gradle" \
  --path-rename "contrib/temporal-spring-ai/README.md:$P/_upstream/README.md" \
  --path-rename "contrib/temporal-spring-ai/build.gradle:$P/_upstream/build.gradle" \
  --path-rename "temporal-spring-ai/:$P/" \
  --path-rename "contrib/temporal-spring-ai/:$P/" \
  --replace-message "$REPO/scripts/migrate/replace-message-java.txt" \
  --commit-callback '
commit.message = commit.message.rstrip(b"\n") + b"\n\nMigrated-From: temporalio/sdk-java@" + commit.original_id + b"\n"'

cat <<MSG
SRC_SHA=$SRC_SHA
FILTER_REPO=$FILTER_REPO_VERSION
WORK=$WORK/src
commits=$(git rev-list --count HEAD)
identities=$(git shortlog -sne HEAD | wc -l | tr -d ' ')

In $REPO, fetch main from $WORK/src and merge using
git merge --allow-unrelated-histories --no-ff. Record the import in IMPORTS.md.
Label the PR history-import and merge it with Create a merge commit.
MSG
