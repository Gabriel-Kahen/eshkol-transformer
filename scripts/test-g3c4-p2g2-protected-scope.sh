#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp env git grep python3 sha256sum timeout; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-protected-scope-test}"
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-protected-scope.py" \
  >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_python_isolation >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
bash "$PROJECT_ROOT/scripts/test-g3c4-p2g1-pending.sh" \
  "$evidence/p2g1-predecessor" >"$evidence/p2g1-predecessor.stdout"
bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-carrier-bridge.sh" \
  "$evidence/carrier-predecessor" >"$evidence/carrier-predecessor.stdout"

tmp="$(mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status == 0 )); then rm -rf -- "$tmp"; else
  printf "P2/G2 ownership failure retained: %s\n" "$tmp" >&2; fi' EXIT

compile_mode() {
  local mode=$1 directory="$tmp/$1" source
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
  "$cc" "${common[@]}" "${optimization[@]}" -DET_A2_KV_CACHE_TESTING \
    -DET_G3C4_P2_G1_PENDING_PRIVATE -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_protected_scope_native.c" \
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
    "${optimization[@]}" -I "$(eshkol_source_dir)/inc" \
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
  ar rcsD "$directory/libg3c4_p2g2_ownership.a" "$directory"/objects/*.o
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
  -Wl,--wrap=et_kernel_runtime_dispatch
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  local fixture prefix source expected log
  for fixture in protected; do
    prefix=
    source="$PROJECT_ROOT/tests/g3c4/p2_g2_protected_scope_test.esk"
    expected='^G3-C4 P2/G2 protected PASS: checks=[1-9][0-9]* pending-only=1$'
    mkdir -p "$directory/cache/$fixture"
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$eshkol_runner" --strict-types --optimize 0 --no-stdlib \
      -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
      -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
      -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
      -L "$directory" --lib g3c4_p2g2_ownership \
      "$source" -o "$directory/p2g2-$fixture" \
      >"$directory/${prefix}compile.stdout" \
      2>"$directory/${prefix}compile.stderr"
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/p2g2-$fixture" >"$directory/${prefix}run.stdout" \
      2>"$directory/${prefix}run.stderr"
    test ! -s "$directory/${prefix}run.stderr"
    grep -E "$expected" "$directory/${prefix}run.stdout" >/dev/null
    for log in compile.stdout compile.stderr run.stdout run.stderr; do
      cp "$directory/${prefix}$log" "$evidence/$mode-${prefix}$log"
    done
  done
}

compile_mode normal
ESHKOL_ARENA_POISON=1 "$tmp/normal/p2g2-protected" \
  >"$evidence/repeat-run.stdout" 2>"$evidence/repeat-run.stderr"
test ! -s "$evidence/repeat-run.stderr"
compile_mode sanitize
cmp "$evidence/normal-run.stdout" "$evidence/repeat-run.stdout"
cmp "$evidence/normal-run.stdout" "$evidence/sanitize-run.stdout"
git -C "$PROJECT_ROOT" diff --check
sha256sum "$PROJECT_ROOT/native/g3c4_p2g2_protected_scope_source_closure.txt" \
  >"$evidence/closure.sha256"
cat "$evidence/static.stdout" "$evidence/normal-run.stdout"
printf 'G3-C4 P2/G2 protected evidence: %s\n' "$evidence"
