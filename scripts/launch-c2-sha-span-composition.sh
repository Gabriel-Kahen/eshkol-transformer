#!/usr/bin/env bash
# One opt-in supported AOT witness. Run only when the build slot is assigned.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 2 && "$1" =~ ^[0-9a-f]{40}$ && "$2" =~ ^[0-9a-f]{40}$ ]] || {
  echo 'usage: launch-c2-sha-span-composition.sh EXPECTED_HEAD EXPECTED_TREE' >&2
  exit 2
}
head=$1 tree=$2
common_git="$(git -C "$root" rev-parse --git-common-dir)"
[[ "$common_git" == /* ]] || { echo 'worktree common Git path is not absolute' >&2; exit 1; }
toolchain=/home/gabe/.cache/eshkol-fe9dfd52-ubuntu22
image=sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6
exec 9>/home/gabe/.cache/eshkol-transformer-orchestration-build.lock
flock -n -x 9 || { echo 'orchestration build lock busy' >&2; exit 75; }
exec 8>/tmp/eshkol-transformer-heavy-build.lock
flock -n -x 8 || { echo 'heavy build lock busy' >&2; exit 75; }
[[ "$(git -C "$root" rev-parse HEAD)" == "$head" &&
   "$(git -C "$root" rev-parse HEAD^{tree})" == "$tree" &&
   -z "$(git -C "$root" status --porcelain --untracked-files=all)" ]] || {
  echo 'source differs from reviewed commit/tree or is dirty' >&2; exit 1;
}
[[ "$(sha256sum "$toolchain/build/eshkol-run" | awk '{print $1}')" == \
   55a552fbf25a2c26daf2f951ad4ff0a9259228ba41f88bed70083849444683b2 ]] || {
  echo 'pinned Eshkol runner changed' >&2; exit 1;
}
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]] || {
  echo 'supported image digest changed' >&2; exit 1;
}
docker run --rm --network none --pull never --cpus=3 \
  --memory=6g --memory-swap=7g \
  -v "$root:/workspace:rw" -v "$common_git:$common_git:ro" \
  -v "$toolchain:/tc:ro" -w /workspace \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/workspace \
  -e GIT_OPTIONAL_LOCKS=0 \
  -e ESHKOL_SOURCE_DIR=/tc/src -e ESHKOL_BUILD_DIR=/tc/build \
  -e CC=/usr/bin/clang-21 -e CXX=/usr/bin/clang++-21 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$image" python3 -B scripts/test-c2-sha-span-composition.py
