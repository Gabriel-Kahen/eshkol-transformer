#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp env git grep python3 sha256sum sort timeout xargs; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-first-enrollment-test}"
mkdir -p "$evidence"
evidence="$(cd -- "$evidence" && pwd)"
cd "$PROJECT_ROOT"
python3 -B -m unittest tests.g3c4.test_p2_g2_first_enrollment >"$evidence/static.stdout" 2>&1
python3 -B scripts/check-g3c4-p2g2-terminal-carriers.py >>"$evidence/static.stdout"
closure=native/g3c4_p2g2_terminal_carriers_source_closure.txt
{
  cat "$closure"
  printf '%s\n' native/g3c4_p2g2_first_enrollment_extension.esk \
    tests/g3c4/p2_g2_first_enrollment_native.c tests/g3c4/p2_g2_first_enrollment_test.esk \
    tests/g3c4/test_p2_g2_first_enrollment.py scripts/test-g3c4-p2g2-first-enrollment.sh
} | LC_ALL=C sort -u | xargs -d '\n' sha256sum -- >"$evidence/source.sha256"
sha256sum "$eshkol_runner" "$cc" "$cxx" >"$evidence/toolchain.sha256"
git log -1 --format='%H %T' >"$evidence/revision.txt"

tmp="$(mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$tmp"; else
  printf "P2/G2 first-enrollment failure retained: %s\n" "$tmp" >&2; fi' EXIT

compile_mode() {
  local mode=$1 directory="$tmp/$1"
  local -a optimization=(-O2)
  local sanitizer_link=
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  # Mirror the admitted anchor runner without editing its sealed closure.
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math
    -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
    -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${optimization[@]}" -DET_A2_KV_CACHE_TESTING \
    -DET_G3C4_P2_G1_PENDING_PRIVATE -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_first_enrollment_native.c" \
    -o "$directory/objects/output_envelope_native.o"
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
    "${optimization[@]}" -DET_G3C4_TERMINAL_EXTENT_TESTING -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "$PROJECT_ROOT/tests/g3c4/call_entry_allocation_shim.cpp" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -o "$directory/objects/p1_identity.o"
  for source in data_io checkpoint_io n3k_primitives_provider \
      g3n_primitives_provider n2_primitives_provider \
      g3c4_primitives_provider g3s_sampling_provider \
      a2_attention_provider a2_kv_cache; do
    "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -c "$PROJECT_ROOT/native/$source.c" \
      -o "$directory/objects/$source.o"
  done
  "$cc" "${common[@]}" "${optimization[@]}" -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -o "$directory/objects/t1_i64_shell.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -o "$directory/objects/m3_i64_integration.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -o "$directory/objects/m3_model.o"
  ar rcsD "$directory/libg3c4_first_enrollment.a" "$directory"/objects/*.o
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
  -Wl,--wrap=arena_allocate_vector_with_header -Wl,--wrap=arena_allocate_with_header \\
  -Wl,--wrap=arena_allocate_cons_with_header -Wl,--wrap=arena_allocate_aligned \\
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler \\
  -Wl,--wrap=et_kernel_runtime_dispatch -Wl,--wrap=et_g3c4_private_prompt_prefill_v1 \\
  -Wl,--wrap=et_g3c4_private_token_frame_begin_last_v1 -Wl,--wrap=et_g3c4_private_call_acquire_v1
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$eshkol_runner" --strict-types --optimize 0 --no-stdlib \
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
    -L "$directory" --lib g3c4_first_enrollment \
    "$PROJECT_ROOT/tests/g3c4/p2_g2_first_enrollment_test.esk" \
    -o "$directory/p2g2-first-enrollment" \
    >"$evidence/$mode-compile.stdout" 2>"$evidence/$mode-compile.stderr"
  test ! -s "$evidence/$mode-compile.stderr"
}

run_fixture() {
  local mode=$1 binary=$2
  env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$binary" >"$evidence/$mode-run.stdout" 2>"$evidence/$mode-run.stderr"
  test ! -s "$evidence/$mode-run.stderr"
  grep -Eq '^P2/G2 first enrollment PASS checks=[1-9][0-9]*$' "$evidence/$mode-run.stdout"
}

compile_mode normal
run_fixture normal "$tmp/normal/p2g2-first-enrollment"
run_fixture repeat "$tmp/normal/p2g2-first-enrollment"
compile_mode sanitize
run_fixture sanitize "$tmp/sanitize/p2g2-first-enrollment"
cmp "$evidence/normal-run.stdout" "$evidence/repeat-run.stdout"
cmp "$evidence/normal-run.stdout" "$evidence/sanitize-run.stdout"
git diff --check
sha256sum -c --quiet "$evidence/source.sha256"
(cd "$evidence" && sha256sum -- *.stdout *.stderr *.sha256 revision.txt >SHA256SUMS-RUN)
cat "$evidence/normal-run.stdout"
printf 'P2/G2 first-enrollment evidence %s\n' "$evidence"
