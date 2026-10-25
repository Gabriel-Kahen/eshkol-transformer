#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(( $# == 1 )) || die "usage: $0 fresh-absolute-evidence-directory"
out=$1
[[ $out == /* && ! -e $out && ! -L $out ]] || die "evidence directory must be fresh"
mkdir -- "$out"
python3 "$PROJECT_ROOT/scripts/check-g3r-p1-candidate-contract.py" \
  >"$out/source-contract.stdout" 2>"$out/source-contract.stderr"
verify_toolchain >"$out/toolchain.stdout" 2>"$out/toolchain.stderr"
for command in ar cmp grep nm timeout; do require_command "$command"; done
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "$provenance" cc_path)"
cxx="$(tsv_value "$provenance" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
[[ $cc == /* && -x $cc && $cxx == /* && -x $cxx && -x $runner ]] || \
  die "pinned compiler or runner is unavailable"
include=(-I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/tests/p1/providers"
         -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native")
for mode in candidate ordinary; do
  mkdir "$out/$mode" "$out/$mode/cache"
  flags=(-DET_P1_TRUSTED_BUILD=1)
  if [[ $mode == candidate ]]; then flags+=(-DET_G3R_CANDIDATE_RETIRE_PRIVATE=1); fi
  "$cc" -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden \
    -fno-common -I "$PROJECT_ROOT/native" "${flags[@]}" \
    -c "$PROJECT_ROOT/native/p1_identity.c" -o "$out/$mode/p1_identity.o" \
    >"$out/$mode/native.stdout" 2>"$out/$mode/native.stderr"
  ar rcsD "$out/$mode/libeshkol_transformer_p1_identity.a" \
    "$out/$mode/p1_identity.o"
  nm -g --defined-only --format=posix "$out/$mode/p1_identity.o" \
    >"$out/$mode/defined.nm"
  if [[ $mode == candidate ]]; then
    [[ $(awk '$1 == "et_p1_private_candidate_construction_begin_v1" {n++} END {print n+0}' \
      "$out/$mode/defined.nm") == 1 ]]
  else
    ! grep -F 'et_p1_private_candidate_' "$out/$mode/defined.nm"
  fi
  source_file="$PROJECT_ROOT/tests/p1/construction_test.esk"
  if [[ $mode == candidate ]]; then
    source_file="$PROJECT_ROOT/tests/g3r/p1_candidate_construction_test.esk"
  fi
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$out/$mode/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    ESHKOL_CXX_COMPILER="$cxx" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "$runner" \
    --strict-types --no-stdlib "${include[@]}" \
    -L "$out/$mode" --lib eshkol_transformer_p1_identity \
    "$source_file" -o "$out/$mode/caller" \
    >"$out/$mode/compile.stdout" 2>"$out/$mode/compile.stderr"
  for run in a b; do
    ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
      120s "$out/$mode/caller" >"$out/$mode/$run.stdout" \
      2>"$out/$mode/$run.stderr"
    test ! -s "$out/$mode/$run.stderr"
  done
  cmp "$out/$mode/a.stdout" "$out/$mode/b.stdout"
  if [[ $mode == candidate ]]; then
    grep -E '^G3-R P1 candidate construction PASS: [0-9]+$' \
      "$out/$mode/a.stdout" >/dev/null
  else
    grep -E '^P1 construction PASS: [0-9]+$' "$out/$mode/a.stdout" >/dev/null
  fi
done
echo 'G3-R P1 candidate handshake PASS: feature-on candidate and ordinary feature-off normal/repeat'
