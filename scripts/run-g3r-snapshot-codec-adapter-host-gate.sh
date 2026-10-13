#!/usr/bin/env bash
# One-use supported-host gate for the accepted, private G3-R adapter source.
set -euo pipefail

adapter_head=615114236150253d2fc04e0129a608ed1ab78bc6
adapter_tree=eda477c449c520223165644150a6c4fda69d1335
main_base=6f6ef9ada4c586573d495c40283ad4b9410d0957
recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# != 2 )); then
  echo "usage: $0 ACCEPTED_ADAPTER_CHECKOUT NEW_EVIDENCE_DIRECTORY" >&2
  exit 2
fi
checkout="$(realpath -- "$1")"
evidence="$2"
[[ -d "$checkout/.git" || -f "$checkout/.git" ]] || {
  echo "adapter checkout is not a Git worktree: $checkout" >&2
  exit 1
}
[[ -z "$(git -C "$recipe_root" status --porcelain --untracked-files=all)" ]] || {
  echo "recipe checkout is dirty" >&2
  exit 1
}
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$adapter_head" ]] || {
  echo "adapter HEAD differs from the accepted exact head" >&2
  exit 1
}
[[ "$(git -C "$checkout" rev-parse HEAD^{tree})" == "$adapter_tree" ]] || {
  echo "adapter tree differs from the accepted exact tree" >&2
  exit 1
}
[[ "$(git -C "$checkout" merge-base "$adapter_head" "$main_base")" == "$main_base" ]] || {
  echo "adapter is not based on the accepted main commit" >&2
  exit 1
}
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=all)" ]] || {
  echo "adapter checkout is dirty" >&2
  exit 1
}
[[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" != 1 ]] || {
  echo "unsupported-host compatibility override is forbidden" >&2
  exit 1
}
[[ "${CC:-clang-21}" == clang-21 && "${CXX:-clang++-21}" == clang++-21 ]] || {
  echo "CC and CXX must use the pinned Clang 21 commands" >&2
  exit 1
}
[[ "${LLVM_CONFIG_EXECUTABLE:-llvm-config-21}" == llvm-config-21 ]] || {
  echo "LLVM_CONFIG_EXECUTABLE must be llvm-config-21" >&2
  exit 1
}
[[ -n "${TMPDIR:-}" && -d "$TMPDIR" && -w "$TMPDIR" ]] || {
  echo "set TMPDIR to a writable, task-specific directory outside /tmp" >&2
  exit 1
}
[[ "$(realpath -- "$TMPDIR")" != /tmp* ]] || {
  echo "TMPDIR must be outside /tmp" >&2
  exit 1
}
evidence="$(realpath -m -- "$evidence")"
[[ "$evidence" != "$checkout" && "$evidence" != "$checkout/"* ]] || {
  echo "evidence directory must be outside the accepted source checkout" >&2
  exit 1
}
[[ ! -e "$evidence" ]] || {
  echo "evidence directory already exists: $evidence" >&2
  exit 1
}

mkdir -p -- "$evidence"
evidence="$(realpath -- "$evidence")"
finalize() {
  local result=$?
  trap - EXIT
  printf 'exit_status\t%s\n' "$result" >>"$evidence/run.tsv"
  git -C "$checkout" status --porcelain --untracked-files=all \
    >"$evidence/post.status" || true
  (cd "$evidence" && find . -type f ! -name SHA256SUMS -print0 |
    sort -z | xargs -0 -r sha256sum >SHA256SUMS) || true
  exit "$result"
}
trap finalize EXIT
{
  printf 'adapter_head\t%s\n' "$adapter_head"
  printf 'adapter_tree\t%s\n' "$adapter_tree"
  printf 'main_base\t%s\n' "$main_base"
  printf 'recipe_head\t%s\n' "$(git -C "$recipe_root" rev-parse HEAD)"
  printf 'recipe_tree\t%s\n' "$(git -C "$recipe_root" rev-parse HEAD^{tree})"
  printf 'host\t%s\n' "$(uname -a)"
  printf 'python\t%s\n' "$(python3 --version)"
} >"$evidence/run.tsv"

[[ "$(python3 --version)" == 'Python 3.14.6' ]] || {
  echo "oracle Python must be 3.14.6" >&2
  exit 1
}
export CC=clang-21 CXX=clang++-21 LLVM_CONFIG_EXECUTABLE=llvm-config-21
make -C "$checkout" toolchain >"$evidence/toolchain.stdout" 2>"$evidence/toolchain.stderr"
"$checkout/scripts/test-g3r-snapshot-codec-adapter-aggregate.sh" \
  "$evidence/aggregate" >"$evidence/runner.stdout" 2>"$evidence/runner.stderr"
[[ ! -s "$evidence/runner.stderr" ]]
[[ ! -s "$evidence/aggregate/normal.compile.stderr" ]]
[[ ! -s "$evidence/aggregate/sanitize.compile.stderr" ]]
cmp "$evidence/aggregate/normal.stdout" "$evidence/aggregate/repeat.stdout"
cmp "$evidence/aggregate/normal.stdout" "$evidence/aggregate/sanitize.stdout"
cmp "$evidence/aggregate/normal.oracle.stdout" "$evidence/aggregate/sanitize.oracle.stdout"
grep -Fxq 'G3-R snapshot codec adapter oracle PASS' \
  "$evidence/aggregate/normal.oracle.stdout"
git -C "$checkout" diff --check
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=all)" ]]
