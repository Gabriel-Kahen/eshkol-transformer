#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-terminal-composer-test}"
retain_emitted="${G3C4_TERMINAL_RETAIN_EMITTED:-0}"
[[ "$retain_emitted" == 0 || "$retain_emitted" == 1 ]] || {
  echo "G3C4_TERMINAL_RETAIN_EMITTED must be 0 or 1" >&2
  exit 2
}
mkdir -p "$evidence"
python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-terminal-composer.py" \
  >"$evidence/static.stdout"
python3 "$PROJECT_ROOT/tests/g3c4/test_context_cache_checker.py" \
  >"$evidence/step4-regression.stdout" \
  2>"$evidence/step4-regression.stderr"
python3 "$PROJECT_ROOT/tests/g3c4/test_active_call_checker.py" \
  >"$evidence/step5-regression.stdout" \
  2>"$evidence/step5-regression.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-t1-prefix-coordinator.sh" \
  "$evidence/prefix-predecessor" >"$evidence/prefix-predecessor.stdout"
bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-terminal-ids-clone.sh" \
  "$evidence/native-terminal-clone-predecessor" \
  >"$evidence/native-terminal-clone-predecessor.stdout"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
if (( retain_emitted )); then
  printf '%s\n' "$tmp" >"$evidence/build-path.txt"
fi
trap 'status=$?; if (( status == 0 && retain_emitted == 0 )); then rm -rf -- "$tmp"; else
  printf "P2/G2 terminal composer failure retained: %s\n" "$tmp" >&2; fi' EXIT

compile_mode() {
  local mode=$1 directory="$tmp/$1" source
  local -a optimization=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  if (( retain_emitted )); then mkdir -p "$directory/generated"; fi
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math
    -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
    -I "$(eshkol_source_dir)/inc")
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
    -DET_G3C4_P2_G1_PENDING_PRIVATE \
    -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE \
    -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_composer_native.c" \
    -MMD -MF "$directory/objects/composer_native.d" \
    -MT "$directory/objects/composer_native.o" \
    -o "$directory/objects/composer_native.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    -MMD -MF "$directory/objects/kernel_fail_allocator.d" \
    -MT "$directory/objects/kernel_fail_allocator.o" \
    -o "$directory/objects/kernel_fail_allocator.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -MMD -MF "$directory/objects/i2_native_helpers.d" \
    -MT "$directory/objects/i2_native_helpers.o" \
    -o "$directory/objects/i2_native_helpers.o"
  "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    "${optimization[@]}" -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "$PROJECT_ROOT/tests/g3c4/call_entry_allocation_shim.cpp" \
    -MMD -MF "$directory/objects/call_entry_allocation_shim.d" \
    -MT "$directory/objects/call_entry_allocation_shim.o" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -MMD -MF "$directory/objects/p1_identity.d" \
    -MT "$directory/objects/p1_identity.o" \
    -o "$directory/objects/p1_identity.o"
  for source in data_io checkpoint_io n3k_primitives_provider \
      g3n_primitives_provider n2_primitives_provider \
      g3c4_primitives_provider g3s_sampling_provider \
      a2_attention_provider a2_kv_cache; do
    "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
      -c "$PROJECT_ROOT/native/$source.c" \
      -MMD -MF "$directory/objects/$source.d" \
      -MT "$directory/objects/$source.o" \
      -o "$directory/objects/$source.o"
  done
  "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -MMD -MF "$directory/objects/t1_i64_shell.d" \
    -MT "$directory/objects/t1_i64_shell.o" \
    -o "$directory/objects/t1_i64_shell.o"
  "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -MMD -MF "$directory/objects/m3_i64_integration.d" \
    -MT "$directory/objects/m3_i64_integration.o" \
    -o "$directory/objects/m3_i64_integration.o"
  "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -MMD -MF "$directory/objects/m3_model.d" \
    -MT "$directory/objects/m3_model.o" \
    -o "$directory/objects/m3_model.o"
  ar rcsD "$directory/libg3c4_p2g2_terminal_composer.a" \
    "$directory"/objects/*.o
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
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_snapshot_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_terminal_ids_clone_copy_v1 \\
  -Wl,--wrap=et_g3c4_private_call_abort_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  local -a eshkol_args=(--strict-types --optimize 0 --no-stdlib
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native"
    -L "$directory" --lib g3c4_p2g2_terminal_composer)

  compile_fixture() {
    local fixture=$1 label=$2 fixture_source=$3 raw object
    mkdir -p "$directory/cache/$fixture"
    local -a env_args=(env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$directory/cache/$fixture"
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap"
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib")
    if (( retain_emitted )); then
      raw="$directory/generated/$fixture.raw.o"
      sha256sum "$fixture_source" >"$evidence/$label-source.before"
      "${env_args[@]}" timeout --foreground --signal=TERM --kill-after=5s 600s \
        "$runner" "${eshkol_args[@]}" --emit-object --dump-ir \
        --emit-depfile "$directory/generated/$fixture.d" \
        "$fixture_source" -o "$raw" \
        >"$evidence/$label-compile.stdout" \
        2>"$evidence/$label-compile.stderr"
      test -s "$raw" && test -s "$raw.bc" && test -s "$raw.ll"
      test -s "$directory/generated/$fixture.d"
      object="$raw"
      if [[ "$mode" == sanitize ]]; then
        python3 "$PROJECT_ROOT/tests/g3c4/prepare_terminal_asan_ir.py" \
          "$raw.ll" "$directory/generated/$fixture.asan.ll" \
          "$directory/generated/$fixture.asan.json" \
          >"$evidence/$label-asan-prepare.stdout" \
          2>"$evidence/$label-asan-prepare.stderr"
        "$cc" -O0 -g -fPIC -fsanitize=address -fno-omit-frame-pointer \
          -c -x ir "$directory/generated/$fixture.asan.ll" \
          -o "$directory/generated/$fixture.asan.o" \
          >"$evidence/$label-asan-compile.stdout" \
          2>"$evidence/$label-asan-compile.stderr"
        object="$directory/generated/$fixture.asan.o"
      fi
      "${env_args[@]}" timeout --foreground --signal=TERM --kill-after=5s 600s \
        "$runner" "${eshkol_args[@]}" "$object" \
        -o "$directory/composer-$fixture" \
        >"$evidence/$label-link.stdout" \
        2>"$evidence/$label-link.stderr"
      sha256sum "$fixture_source" >"$evidence/$label-source.after"
      cmp "$evidence/$label-source.before" "$evidence/$label-source.after"
    else
      "${env_args[@]}" timeout --foreground --signal=TERM --kill-after=5s 600s \
        "$runner" "${eshkol_args[@]}" "$fixture_source" \
        -o "$directory/composer-$fixture" \
        >"$evidence/$label-compile.stdout" \
        2>"$evidence/$label-compile.stderr"
    fi
  }
  local fixture label expected fixture_source
  for fixture in terminal feature-off; do
    label="$mode"
    expected='^G3-C4 P2/G2 Eshkol terminal PASS: checks=[1-9][0-9]*$'
    fixture_source="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_composer_test.esk"
    if [[ "$fixture" == feature-off ]]; then
      label="$mode-feature-off"
      expected='^G3-C4 P2/G2 terminal off PASS: checks=[1-9][0-9]*$'
      fixture_source="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk"
    fi
    compile_fixture "$fixture" "$label" "$fixture_source"
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/composer-$fixture" >"$evidence/$label-run.stdout" \
      2>"$evidence/$label-run.stderr"
    test ! -s "$evidence/$label-run.stderr"
    grep -E "$expected" "$evidence/$label-run.stdout" >/dev/null
  done

  python3 "$PROJECT_ROOT/tests/g3c4/generate_p2_g2_terminal_fault_overlays.py" \
    "$directory/overlays" >"$evidence/$mode-overlay-generation.stdout"
  (cd "$directory/overlays" && sha256sum -c SHA256SUMS) \
    >"$evidence/$mode-overlay-identity.stdout"
  cp "$directory/overlays/SHA256SUMS" "$evidence/$mode-overlay-SHA256SUMS"
  for fixture in parity-original parity-overlay unmarked decode-first decode-second \
      precommit postzero lease-i1 lease-a2; do
    compile_fixture "$fixture" "$mode-$fixture" \
      "$directory/overlays/$fixture.esk"
    local exit_status=0
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/composer-$fixture" \
      3>"$evidence/$mode-$fixture.marker" \
      >"$evidence/$mode-$fixture-run.stdout" \
      2>"$evidence/$mode-$fixture-run.stderr" || exit_status=$?
    if [[ "$fixture" == postzero ]]; then
      test "$exit_status" -eq 134
      test "$(cat "$evidence/$mode-$fixture.marker")" = Z
      ! grep -Eq 'ILLEGAL_TERMINAL_RETURN|OVERLAY_PARITY|OVERLAY_ABORT|ESH_TERMINAL' \
        "$evidence/$mode-$fixture-run.stdout"
    else
      test "$exit_status" -eq 0
      test ! -s "$evidence/$mode-$fixture.marker"
      test ! -s "$evidence/$mode-$fixture-run.stderr"
      if [[ "$fixture" == parity-* ]]; then
        grep -Eq '^OVERLAY_PARITY [0-9]+ [0-9]+ 4$' \
          "$evidence/$mode-$fixture-run.stdout"
      elif [[ "$fixture" == unmarked ]]; then
        grep -Fx OVERLAY_UNMARKED \
          "$evidence/$mode-$fixture-run.stdout" >/dev/null
      else
        grep -Fx "OVERLAY_ABORT ${fixture#decode-}" \
          "$evidence/$mode-$fixture-run.stdout" >/dev/null
      fi
    fi
    printf '%s %s\n' "$fixture" "$exit_status" \
      >>"$evidence/$mode-overlay-exits.txt"
  done
  cmp "$evidence/$mode-parity-original-run.stdout" \
      "$evidence/$mode-parity-overlay-run.stdout"
  if (( retain_emitted )); then
    python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_terminal_emitted_evidence.py" \
      "$directory" "$evidence" "$mode" \
      >"$evidence/$mode-emitted-check.stdout" \
      2>"$evidence/$mode-emitted-check.stderr"
  fi
}

compile_mode normal
awk '/^COORD_ORACLE / { if (++rows > 2) next } { print }' \
  "$evidence/normal-run.stdout" >"$evidence/prefix-oracle-input.stdout"
python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_t1_prefix_oracle.py" \
  "$evidence/prefix-oracle-input.stdout" >"$evidence/oracle.stdout"
python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_terminal_composer_oracle.py" \
  "$evidence/normal-run.stdout" >"$evidence/second-oracle.stdout"
ESHKOL_ARENA_POISON=1 "$tmp/normal/composer-terminal" \
  >"$evidence/repeat-run.stdout" 2>"$evidence/repeat-run.stderr"
ESHKOL_ARENA_POISON=1 "$tmp/normal/composer-feature-off" \
  >"$evidence/repeat-feature-off-run.stdout" \
  2>"$evidence/repeat-feature-off-run.stderr"
test ! -s "$evidence/repeat-run.stderr"
test ! -s "$evidence/repeat-feature-off-run.stderr"
compile_mode sanitize
cmp "$evidence/normal-run.stdout" "$evidence/repeat-run.stdout"
cmp "$evidence/normal-run.stdout" "$evidence/sanitize-run.stdout"
cmp "$evidence/normal-feature-off-run.stdout" \
    "$evidence/repeat-feature-off-run.stdout"
cmp "$evidence/normal-feature-off-run.stdout" \
    "$evidence/sanitize-feature-off-run.stdout"
git -C "$PROJECT_ROOT" diff --check
sha256sum "$PROJECT_ROOT/native/g3c4_p2g2_terminal_composer_source_closure.txt" \
  >"$evidence/closure.sha256"
if (( retain_emitted )); then
  printf '%s\n' 'HOLD: compiler artifacts and imports retained; successful-zero no-allocation and complete launcher lifecycle remain unproved' \
    >"$evidence/emitted-acceptance.status"
fi
cat "$evidence/normal-run.stdout"
