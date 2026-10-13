#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.g3c4.test_p2_g2_held_holder_eshkol_source \
  tests.g3c4.test_p2_g2_held_holder_compiled_source
if [[ "${G3C4_HELD_SOURCE_ONLY:-0}" == 1 ]]; then
  printf 'P2/G2 held-holder compiled gate source preparation PASS (runtime not run)\n'
  exit 0
fi

verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-held-holder-compiled}"
mkdir -p "$evidence"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status != 0 )); then
  printf "P2/G2 held-holder compiled failure retained: %s\n" "$tmp" >&2;
fi' EXIT

compile_mode() {
  local mode=$1 directory="$tmp/$1" source fixture expected label
  local -a optimization=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math
    -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
    -I "$(eshkol_source_dir)/inc")
  for fixture in on off; do
    source="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_composer_native.c"
    [[ "$fixture" == on ]] && \
      source="$PROJECT_ROOT/tests/g3c4/p2_g2_held_holder_compiled_native.c"
    "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
      -DET_G3C4_P2_G1_PENDING_PRIVATE \
      -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
      -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE \
      -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE \
      -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
      -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
      -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
      -c "$source" -o "$directory/objects/owner-$fixture.o"
  done
  for source in data_io checkpoint_io n3k_primitives_provider \
      g3n_primitives_provider n2_primitives_provider \
      g3c4_primitives_provider g3s_sampling_provider \
      a2_attention_provider a2_kv_cache; do
    "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
      -c "$PROJECT_ROOT/native/$source.c" \
      -o "$directory/objects/$source.o"
  done
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    -o "$directory/objects/kernel_fail_allocator.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_native_helpers.o"
  "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    "${optimization[@]}" -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "$PROJECT_ROOT/tests/g3c4/call_entry_allocation_shim.cpp" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -o "$directory/objects/p1_identity.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -o "$directory/objects/t1_i64_shell.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -o "$directory/objects/m3_i64_integration.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -o "$directory/objects/m3_model.o"
  local -a shared_objects=()
  local object
  for object in "$directory/objects"/*.o; do
    [[ "$object" == "$directory/objects/owner-on.o" ||
       "$object" == "$directory/objects/owner-off.o" ]] ||
      shared_objects+=("$object")
  done
  for fixture in on off; do
    ar rcsD "$directory/libg3c4_held_$fixture.a" \
      "$directory/objects/owner-$fixture.o" "${shared_objects[@]}"
  done
  nm -g --defined-only "$directory/objects/owner-on.o" |
    rg 'et_g3c4_private_p2g2_holder_(new|generation|bind|begin_capture|witness|normal_close|abort_attempt|end_after_busy|dispose)_v1' \
    >"$evidence/$mode-holder-on-symbols.txt"
  test "$(wc -l <"$evidence/$mode-holder-on-symbols.txt")" -eq 9
  if nm -g --defined-only "$directory/objects/owner-off.o" |
      rg 'et_g3c4_private_p2g2_holder_.*_v1'; then
    die 'feature-off object exports holder ABI'
  fi
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && \
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
  -Wl,--wrap=arena_allocate_vector_with_header \\
  -Wl,--wrap=arena_allocate_with_header \\
  -Wl,--wrap=arena_allocate_cons_with_header \\
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler \\
  -Wl,--wrap=et_kernel_runtime_dispatch \\
  -Wl,--wrap=et_i64_tensor_borrow_view_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_first_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_eos_first_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_second_frame_carrier_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_prefix_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_call_abort_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_begin_capture_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_abort_attempt_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_end_after_busy_v1
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  for fixture in on off; do
    local input="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk"
    [[ "$fixture" == on ]] && \
      input="$PROJECT_ROOT/tests/g3c4/p2_g2_held_holder_compiled_test.esk"
    mkdir -p "$directory/cache/$fixture"
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$runner" --strict-types --optimize 0 --no-stdlib \
      --emit-depfile "$directory/$fixture.d" \
      -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
      -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
      -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
      -L "$directory" --lib "g3c4_held_$fixture" \
      "$input" -o "$directory/held-$fixture" \
      >"$evidence/$mode-$fixture-compile.stdout" \
      2>"$evidence/$mode-$fixture-compile.stderr"
    test ! -s "$evidence/$mode-$fixture-compile.stderr"
  done
  python3 - "$PROJECT_ROOT" "$directory/on.d" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
depfile = Path(sys.argv[2]).read_text().replace('\\\n', ' ')
actual = []
for source in depfile.split(':', 1)[1].split():
    path = Path(source)
    if not path.is_absolute():
        path = Path(sys.argv[2]).parent / path
    path = path.resolve()
    if path.is_relative_to(root):
        name = path.relative_to(root).as_posix()
        if name != 'tests/g3c4/p2_g2_held_holder_compiled_test.esk':
            actual.append(name)
expected = (root / 'native/g3c4_p2g2_terminal_held_holder_source_closure.txt').read_text().splitlines()
assert actual == expected, (actual, expected)
PY
  for fixture in on off; do
    local expected='^G3-C4 held-holder compiled PASS: checks=[1-9][0-9]*$'
    [[ "$fixture" == off ]] && \
      expected='^G3-C4 P2/G2 terminal off PASS: checks=[1-9][0-9]*$'
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/held-$fixture" normal \
      >"$evidence/$mode-$fixture.stdout" 2>"$evidence/$mode-$fixture.stderr"
    test ! -s "$evidence/$mode-$fixture.stderr"
    rg "$expected" "$evidence/$mode-$fixture.stdout" >/dev/null
  done
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 "$directory/held-on" normal \
    >"$evidence/$mode-repeat.stdout" 2>"$evidence/$mode-repeat.stderr"
  test ! -s "$evidence/$mode-repeat.stderr"
  cmp "$evidence/$mode-on.stdout" "$evidence/$mode-repeat.stdout"
  for fixture in repeat-busy postzero; do
    local status=0 marker=R
    [[ "$fixture" == postzero ]] && marker=Z
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/held-on" "$fixture" \
      3>"$evidence/$mode-$fixture.marker" \
      >"$evidence/$mode-$fixture.stdout" \
      2>"$evidence/$mode-$fixture.stderr" || status=$?
    test "$status" -eq 134
    test "$(cat "$evidence/$mode-$fixture.marker")" = "$marker"
    ! rg 'held-holder compiled PASS' "$evidence/$mode-$fixture.stdout"
  done
}

compile_mode normal
compile_mode sanitize
cmp "$evidence/normal-on.stdout" "$evidence/sanitize-on.stdout"
cmp "$evidence/normal-off.stdout" "$evidence/sanitize-off.stdout"
sha256sum "$PROJECT_ROOT/native/g3c4_p2g2_terminal_held_holder_source_closure.txt" \
  >"$evidence/closure.sha256"
printf 'P2/G2 held-holder compiled normal/repeat/sanitizer/off/fail-stop PASS\n' \
  | tee "$evidence/result.txt"
