#!/usr/bin/env bash
# Pending compiled admission. Reuses the independently bounded linked-owner
# archive, then adds only the feature-private 136-byte bridge and test root.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
[[ $# == 1 ]] || die "usage: $0 FRESH_EVIDENCE_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" && ! -L "$out" ]] || die "evidence directory must be fresh"
[[ "$out" != "$PROJECT_ROOT" && "$out" != "$PROJECT_ROOT/"* ]] || \
  die "evidence must remain outside checkout"
mkdir -p "$out" "$out/cache"
"$PROJECT_ROOT/scripts/test-g3r-original-owner-linked.sh" "$out/linked"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
"$cc" -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -fvisibility=hidden -fno-common -I "$PROJECT_ROOT/src" \
  -DET_G3R_COMMITTED_HISTORY_PRIVATE -DET_G3R_SAME_OWNER_SAVE_PRIVATE \
  -c "$PROJECT_ROOT/native/g3r_held_image_bridge.c" \
  -o "$out/held-image-bridge.o" \
  >"$out/bridge-build.stdout" 2>"$out/bridge-build.stderr"
cp -- "$out/linked/libg3r_linked_owner.a" "$out/libg3r_held_image.a"
ar rcsD "$out/libg3r_held_image.a" "$out/held-image-bridge.o"
nm -g --defined-only "$out/libg3r_held_image.a" | \
  rg -q '[[:space:]]et_g3r_private_held_image_copy_v1$' || \
  die "archive lacks held-image bridge"
runner="$(eshkol_build_dir)/eshkol-run"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$out/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  ESHKOL_CXX_COMPILER="$cxx" \
  timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
    --strict-types --no-stdlib -O 0 \
    -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/internal/p1/lib" \
    -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/internal/t2/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/internal/d2/lib" \
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" \
    -L "$out" --lib g3r_held_image \
    "$PROJECT_ROOT/tests/g3r/held_image_adapter_runtime.esk" \
    -o "$out/held-image" \
    >"$out/compile.stdout" 2>"$out/compile.stderr"
[[ -s "$out/held-image" && ! -s "$out/compile.stderr" ]] || \
  die "held-image compilation failed"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
  "$out/held-image" "$out/linked/corpus" normal \
  >"$out/normal.stdout" 2>"$out/normal.stderr"
[[ ! -s "$out/normal.stderr" ]] || die "normal case wrote stderr"
rg -q '^G3R HELD IMAGE ADAPTER PASS: [0-9]+ checks$' \
  "$out/normal.stdout" || die "normal case missed pass marker"
for mode in uncertain-release-raise uncertain-release-return; do
  if ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "$out/held-image" "$out/linked/corpus" "$mode" \
    >"$out/$mode.stdout" 2>"$out/$mode.stderr"; then
    status=0
  else
    status=$?
  fi
  [[ "$status" == 134 ]] || die "$mode did not fail stop: $status"
  rg -q '^G3R HELD IMAGE UNCERTAIN RELEASE$' \
    "$out/$mode.stderr" || die "$mode cut was not reached"
  ! rg -q 'G3R HELD IMAGE ADAPTER PASS' "$out/$mode.stdout" || \
    die "$mode escaped"
done
{
  printf 'commit\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  printf 'tree\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})"
  printf 'archive_sha256\t%s\n' \
    "$(sha256sum "$out/libg3r_held_image.a" | cut -d ' ' -f 1)"
  printf 'result\tPASS\n'
} >"$out/manifest.tsv"
printf 'G3R held image compiled PASS: %s\n' "$out"
