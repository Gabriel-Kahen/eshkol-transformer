#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(( $# == 1 )) || die "usage: $0 fresh-absolute-evidence-directory"
out=$1
[[ $out == /* && ! -e $out && ! -L $out ]] || die "evidence directory must be fresh"
mkdir -- "$out"
python3 "$PROJECT_ROOT/scripts/check-g3r-i2-candidate-contract.py" \
  >"$out/source-contract.stdout" 2>"$out/source-contract.stderr"
verify_toolchain >"$out/toolchain.stdout" 2>"$out/toolchain.stderr"
for command in ar cmp nm sha256sum timeout; do require_command "$command"; done
for item in \
  'f1314ccd4c5a382733115005c93d8712290bd63c5373eac2509093989fefad33  native/i2_wave2_root.esk' \
  '66bdd604a8bffcf90913961084993b2410bd89d3dbb7a9aed0de0524115164dc  native/i2_wave2_extension.esk' \
  'fd3a314af97e6cc432ad3b1455110ac11cb8a23d2da1ff7a1f96280f3d45de67  native/i2_wave2_package_bridge.c' \
  '71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815  native/m3t_transport_extension.esk'; do
  (cd "$PROJECT_ROOT" && printf '%s\n' "$item" | sha256sum -c -) \
    >>"$out/feature-off-source.txt"
done
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "$provenance" cc_path)"
cxx="$(tsv_value "$provenance" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
[[ $cc == /* && -x $cc && $cxx == /* && -x $cxx && -x $runner ]] || \
  die "pinned compiler or runner is unavailable"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -ffp-contract=off -fexcess-precision=standard -frounding-math
        -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
        -I "$PROJECT_ROOT/src" -DET_F32_TENSOR_TESTING)
for mode in candidate ordinary; do
  mkdir "$out/$mode" "$out/$mode/cache"
  "$cc" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$out/$mode/i2_bridge.o"
  for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
                n3k_primitives_provider n2_primitives_provider a2_attention_provider; do
    "$cc" "${cflags[@]}" -c "$PROJECT_ROOT/native/$source.c" \
      -o "$out/$mode/$source.o"
  done
  for source in m3t_transport m3t_f32_integration; do
    "$cc" "${cflags[@]}" -c "$PROJECT_ROOT/src/eshkol_transformer/$source.c" \
      -o "$out/$mode/$source.o"
  done
  candidate_flag=()
  if [[ $mode == candidate ]]; then
    candidate_flag=(-DET_G3R_CANDIDATE_RETIRE_PRIVATE=1)
  fi
  "$cc" "${cflags[@]}" "${candidate_flag[@]}" \
    -c "$PROJECT_ROOT/tests/p1/i2_construction_test_bridge.c" \
    -o "$out/$mode/test_bridge.o"
  nm -g --defined-only --format=posix "$out/$mode/test_bridge.o" \
    >"$out/$mode/defined.nm"
  if [[ $mode == candidate ]]; then
    [[ $(awk '$1 == "et_p1_private_candidate_construction_begin_v1" {n++} END {print n+0}' \
      "$out/$mode/defined.nm") == 1 ]]
  else
    ! grep -F 'et_p1_private_candidate_' "$out/$mode/defined.nm"
  fi
  ar rcsD "$out/$mode/libm3t_construction_test.a" "$out/$mode"/*.o
  source_file="$PROJECT_ROOT/tests/g3r/i2_candidate_feature_off_test.esk"
  if [[ $mode == candidate ]]; then
    source_file="$PROJECT_ROOT/tests/g3r/i2_candidate_construction_test.esk"
  fi
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$out/$mode/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    ESHKOL_CXX_COMPILER="$cxx" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "$runner" \
    --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" \
    -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$out/$mode" --lib m3t_construction_test \
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
    grep -E '^G3-R I2 candidate construction PASS: [0-9]+$' \
      "$out/$mode/a.stdout" >/dev/null
  else
    grep -Fx 'G3-R I2 ordinary feature-off PASS' \
      "$out/$mode/a.stdout" >/dev/null
  fi
done
cmp "$out/candidate/i2_bridge.o" "$out/ordinary/i2_bridge.o"
echo 'G3-R I2 candidate handshake PASS: candidate and ordinary normal/repeat'
