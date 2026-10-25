#!/usr/bin/env bash
# External one-use launcher. A busy shared heavy lock is a preflight refusal.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
(( $# == 3 )) || die "usage: $0 EXPECTED_COMMIT EXPECTED_TREE FRESH_ABSOLUTE_EVIDENCE_DIRECTORY"
expected_commit=$1 expected_tree=$2 out=$3
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ && "$expected_tree" =~ ^[0-9a-f]{40}$ ]] || \
  die "expected source identities must be full Git hashes"
[[ "$out" == /* && ! -e "$out" && ! -L "$out" ]] || \
  die "N4 evidence directory must be fresh and absolute"
cd -- "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$expected_commit" && \
   "$(git rev-parse HEAD^{tree})" == "$expected_tree" ]] || \
  die "N4 launcher source identity differs from pinned commit/tree"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "N4 launcher refuses a dirty source tree"
lock=/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/heavy.lock
[[ -f "$lock" && ! -L "$lock" ]] || die "shared heavy lock is unavailable"
flock -n -x "$lock" bash "$PROJECT_ROOT/scripts/test-e3-n4-sparse-subnormal.sh" "$out"
