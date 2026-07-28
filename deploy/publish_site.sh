#!/usr/bin/env bash
# Publish web/ to the gh-pages branch, which GitHub Pages serves as the site root.
#
# Why a branch and not an Action: GitHub Pages can only serve from a repository
# root or /docs, and the site lives in web/. The usual fix is a Pages workflow,
# but pushing anything under .github/workflows/ requires a token with the
# `workflow` scope, which the default `gh` login does not have. This script does
# the same job with no extra scope.
#
# To switch to the Action instead (automatic deploys on every push to main):
#   gh auth refresh -s workflow
#   mkdir -p .github/workflows
#   mv deploy/github-pages-workflow.yml.optional .github/workflows/pages.yml
#   git add -A && git commit -m "Deploy Pages via Actions" && git push
#   gh api -X POST repos/:owner/:repo/pages -f build_type=workflow
#
# Usage:  ./deploy/publish_site.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=gh-pages
SRC=web

# Refuse to publish a site whose claims have drifted from the data.
echo "→ verifying claims before publishing"
python3 analysis/verify_tour_claims.py > /tmp/phos_verify.txt 2>&1 || {
    echo "✗ verify_tour_claims.py FAILED — not publishing. See /tmp/phos_verify.txt"
    tail -8 /tmp/phos_verify.txt
    exit 1
}
tail -2 /tmp/phos_verify.txt

# Refuse to publish a half-built site.
python3 - <<'PY'
import json, pathlib, sys
root = pathlib.Path('web/data')
idx = json.loads((root / 'sessions.json').read_text())
missing = [s['file'] for s in idx['sessions'] if not (root / s['file']).exists()]
if missing:
    sys.exit(f"✗ sessions.json lists bundles that are absent: {missing}")
print(f"→ {len(idx['sessions'])} session bundles present")
PY

echo "→ building $BRANCH from $SRC/"
WORKTREE=$(mktemp -d)
git worktree add --detach "$WORKTREE" >/dev/null 2>&1
trap 'git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true' EXIT

REPO_ROOT=$PWD
SRC_SHA=$(git rev-parse --short HEAD)
cd "$WORKTREE"

# A throwaway branch name, never "$BRANCH": `git checkout --orphan gh-pages`
# fails once that branch exists, which silently broke every run after the first.
git checkout -q --orphan "publish-$$"
git rm -rq --cached . 2>/dev/null || true
find . -maxdepth 1 ! -name . ! -name .git -exec rm -rf {} +
cp -R "$REPO_ROOT/$SRC/." .
git add -A
git commit -q -m "Publish site from $SRC_SHA"
NEW_SHA=$(git rev-parse HEAD)
git push -q --force origin "HEAD:refs/heads/$BRANCH"

# Positive assertion. A push that "did not error" is not a push that landed.
cd "$REPO_ROOT"
git fetch -q origin "$BRANCH"
REMOTE_SHA=$(git rev-parse "origin/$BRANCH")
if [ "$NEW_SHA" != "$REMOTE_SHA" ]; then
    echo "✗ push did not land: built $NEW_SHA, remote has $REMOTE_SHA"
    exit 1
fi

echo "✓ $BRANCH now at ${NEW_SHA:0:7} (built from $SRC_SHA)"
echo "  https://sdeture.github.io/Phosphenes/  — CDN takes a minute"
