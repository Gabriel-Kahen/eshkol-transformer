#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-terminal-composer-test}"
retain_emitted="${G3C4_TERMINAL_RETAIN_EMITTED:-0}"
[[ "$retain_emitted" == 0 || "$retain_emitted" == 1 ]] || {
  echo "G3C4_TERMINAL_RETAIN_EMITTED must be 0 or 1" >&2
  exit 2
}
mkdir -p "$evidence"
source "$PROJECT_ROOT/scripts/g3c4_terminal_command_receipt.sh"
terminal_command_index=0
terminal_commands_root=$(mktemp -d "$evidence/commands.XXXXXX")
printf '%s\n' "$terminal_commands_root" >"$evidence/commands-path.txt"
terminal_record_command toolchain-verify - - -- \
  bash -c 'source "$1"; verify_toolchain' bash "$PROJECT_ROOT/scripts/common.sh"
terminal_restore_toolchain_export
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
terminal_record_command static static.stdout static.stderr -- \
  python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-terminal-composer.py"
terminal_record_command step4-regression step4-regression.stdout step4-regression.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/test_context_cache_checker.py"
terminal_record_command step5-regression step5-regression.stdout step5-regression.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/test_active_call_checker.py"
terminal_record_command q0 q0.stdout q0.stderr -- \
  env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_python_isolation
terminal_record_command prefix-predecessor prefix-predecessor.stdout prefix-predecessor.stderr -- \
  bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-t1-prefix-coordinator.sh" \
  "$evidence/prefix-predecessor"
terminal_record_command native-terminal-clone-predecessor \
  native-terminal-clone-predecessor.stdout native-terminal-clone-predecessor.stderr -- \
  bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-terminal-ids-clone.sh" \
  "$evidence/native-terminal-clone-predecessor"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
if (( retain_emitted )); then
  printf '%s\n' "$tmp" >"$evidence/build-path.txt"
fi
trap 'status=$?; if (( status == 0 )); then
  if (( retain_emitted == 0 )); then rm -rf -- "$tmp"; else
    printf "P2/G2 terminal composer emitted evidence retained: %s\n" "$tmp" >&2; fi
else printf "P2/G2 terminal composer failure retained: %s\n" "$tmp" >&2; fi' EXIT

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
  terminal_record_command "$mode-native-composer" - - -- \
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
  terminal_record_command "$mode-native-kernel-fail-allocator" - - -- \
    "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    -MMD -MF "$directory/objects/kernel_fail_allocator.d" \
    -MT "$directory/objects/kernel_fail_allocator.o" \
    -o "$directory/objects/kernel_fail_allocator.o"
  terminal_record_command "$mode-native-i2-helpers" - - -- \
    "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -MMD -MF "$directory/objects/i2_native_helpers.d" \
    -MT "$directory/objects/i2_native_helpers.o" \
    -o "$directory/objects/i2_native_helpers.o"
  terminal_record_command "$mode-native-call-entry-shim" - - -- \
    "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    "${optimization[@]}" -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "$PROJECT_ROOT/tests/g3c4/call_entry_allocation_shim.cpp" \
    -MMD -MF "$directory/objects/call_entry_allocation_shim.d" \
    -MT "$directory/objects/call_entry_allocation_shim.o" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  terminal_record_command "$mode-native-p1-identity" - - -- \
    "$cc" "${common[@]}" "${optimization[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -MMD -MF "$directory/objects/p1_identity.d" \
    -MT "$directory/objects/p1_identity.o" \
    -o "$directory/objects/p1_identity.o"
  for source in data_io checkpoint_io n3k_primitives_provider \
      g3n_primitives_provider n2_primitives_provider \
      g3c4_primitives_provider g3s_sampling_provider \
      a2_attention_provider a2_kv_cache; do
    terminal_record_command "$mode-native-$source" - - -- \
      "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
      -c "$PROJECT_ROOT/native/$source.c" \
      -MMD -MF "$directory/objects/$source.d" \
      -MT "$directory/objects/$source.o" \
      -o "$directory/objects/$source.o"
  done
  terminal_record_command "$mode-native-t1-shell" - - -- \
    "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -MMD -MF "$directory/objects/t1_i64_shell.d" \
    -MT "$directory/objects/t1_i64_shell.o" \
    -o "$directory/objects/t1_i64_shell.o"
  terminal_record_command "$mode-native-m3-i64" - - -- \
    "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -MMD -MF "$directory/objects/m3_i64_integration.d" \
    -MT "$directory/objects/m3_i64_integration.o" \
    -o "$directory/objects/m3_i64_integration.o"
  terminal_record_command "$mode-native-m3-model" - - -- \
    "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -MMD -MF "$directory/objects/m3_model.d" \
    -MT "$directory/objects/m3_model.o" \
    -o "$directory/objects/m3_model.o"
  terminal_record_command "$mode-native-archive" - - -- \
    ar rcsD "$directory/libg3c4_p2g2_terminal_composer.a" \
    "$directory"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && \
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
set -euo pipefail
evidence=$(printf '%q' "$evidence")
terminal_commands_root=$(printf '%q' "$terminal_commands_root")
source $(printf '%q' "$PROJECT_ROOT/scripts/g3c4_terminal_command_receipt.sh")
terminal_record_nested_link "\${G3C4_TERMINAL_RECEIPT_LABEL:?}" \\
  $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
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
      terminal_record_command "$label-source-before" "$label-source.before" - -- \
        sha256sum "$fixture_source"
      terminal_record_command "$label-compile" "$label-compile.stdout" "$label-compile.stderr" -- \
        "${env_args[@]}" G3C4_TERMINAL_RECEIPT_LABEL="$label-compile" \
        timeout --foreground --signal=TERM --kill-after=5s 600s \
        "$runner" "${eshkol_args[@]}" --emit-object --dump-ir \
        --emit-depfile "$directory/generated/$fixture.d" \
        "$fixture_source" -o "$raw"
      terminal_record_command "$label-raw-artifacts" - - -- \
        bash -c 'for path do test -s "$path" || exit; done' bash \
        "$raw" "$raw.bc" "$raw.ll" "$directory/generated/$fixture.d"
      object="$raw"
      if [[ "$mode" == sanitize ]]; then
        terminal_record_command "$label-asan-prepare" "$label-asan-prepare.stdout" "$label-asan-prepare.stderr" -- \
          python3 "$PROJECT_ROOT/tests/g3c4/prepare_terminal_asan_ir.py" \
          "$raw.ll" "$directory/generated/$fixture.asan.ll" \
          "$directory/generated/$fixture.asan.json"
        terminal_record_command "$label-asan-compile" "$label-asan-compile.stdout" "$label-asan-compile.stderr" -- \
          "$cc" -O0 -g -fPIC -fsanitize=address -fno-omit-frame-pointer \
          -c -x ir "$directory/generated/$fixture.asan.ll" \
          -o "$directory/generated/$fixture.asan.o"
        object="$directory/generated/$fixture.asan.o"
      fi
      terminal_record_command "$label-link" "$label-link.stdout" "$label-link.stderr" -- \
        "${env_args[@]}" G3C4_TERMINAL_RECEIPT_LABEL="$label-link" \
        timeout --foreground --signal=TERM --kill-after=5s 600s \
        "$runner" "${eshkol_args[@]}" "$object" \
        -o "$directory/composer-$fixture"
      terminal_record_command "$label-source-after" "$label-source.after" - -- \
        sha256sum "$fixture_source"
      terminal_record_command "$label-source-match" - - -- \
        cmp "$evidence/$label-source.before" "$evidence/$label-source.after"
    else
      terminal_record_command "$label-compile" "$label-compile.stdout" "$label-compile.stderr" -- \
        "${env_args[@]}" G3C4_TERMINAL_RECEIPT_LABEL="$label-compile" \
        timeout --foreground --signal=TERM --kill-after=5s 600s \
        "$runner" "${eshkol_args[@]}" "$fixture_source" \
        -o "$directory/composer-$fixture"
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
    terminal_record_command "$label-run" "$label-run.stdout" "$label-run.stderr" -- \
      "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/composer-$fixture"
    terminal_record_command "$label-no-stderr" - - -- \
      test ! -s "$evidence/$label-run.stderr"
    terminal_record_command "$label-pass-marker" - - -- \
      grep -E "$expected" "$evidence/$label-run.stdout"
  done

  terminal_record_command "$mode-overlay-generation" "$mode-overlay-generation.stdout" "$mode-overlay-generation.stderr" -- \
    python3 "$PROJECT_ROOT/tests/g3c4/generate_p2_g2_terminal_fault_overlays.py" \
    "$directory/overlays"
  terminal_record_command "$mode-overlay-identity" \
    "$mode-overlay-identity.stdout" "$mode-overlay-identity.stderr" -- \
    bash -c 'cd -- "$1" && sha256sum -c SHA256SUMS' bash "$directory/overlays"
  terminal_record_command "$mode-overlay-manifest-copy" - - -- \
    cp "$directory/overlays/SHA256SUMS" "$evidence/$mode-overlay-SHA256SUMS"
  for fixture in parity-original parity-overlay unmarked decode-first decode-second \
      precommit postzero lease-i1 lease-a2; do
    compile_fixture "$fixture" "$mode-$fixture" \
      "$directory/overlays/$fixture.esk"
    local exit_status=0 marker
    terminal_record_command "$mode-$fixture-run" "$mode-$fixture-run.stdout" "$mode-$fixture-run.stderr" -- \
      "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/composer-$fixture" \
      3>"$evidence/$mode-$fixture.marker" || exit_status=$?
    if [[ "$fixture" == postzero ]]; then
      terminal_record_command "$mode-$fixture-exit-check" - - -- test "$exit_status" -eq 134
      marker=$(<"$evidence/$mode-$fixture.marker")
      terminal_record_command "$mode-$fixture-marker-check" - - -- test "$marker" = Z
      ! terminal_record_command "$mode-$fixture-no-return" - - -- \
        grep -Eq 'ILLEGAL_TERMINAL_RETURN|OVERLAY_PARITY|OVERLAY_ABORT|ESH_TERMINAL' \
        "$evidence/$mode-$fixture-run.stdout"
    else
      terminal_record_command "$mode-$fixture-exit-check" - - -- test "$exit_status" -eq 0
      terminal_record_command "$mode-$fixture-no-marker" - - -- \
        test ! -s "$evidence/$mode-$fixture.marker"
      terminal_record_command "$mode-$fixture-no-stderr" - - -- \
        test ! -s "$evidence/$mode-$fixture-run.stderr"
      if [[ "$fixture" == parity-* ]]; then
        terminal_record_command "$mode-$fixture-parity-marker" - - -- \
          grep -Eq '^OVERLAY_PARITY [0-9]+ [0-9]+ 4$' \
          "$evidence/$mode-$fixture-run.stdout"
      elif [[ "$fixture" == unmarked ]]; then
        terminal_record_command "$mode-$fixture-unmarked-marker" - - -- \
          grep -Fx OVERLAY_UNMARKED "$evidence/$mode-$fixture-run.stdout"
      else
        terminal_record_command "$mode-$fixture-abort-marker" - - -- \
          grep -Fx "OVERLAY_ABORT ${fixture#decode-}" \
          "$evidence/$mode-$fixture-run.stdout"
      fi
    fi
    printf '%s %s\n' "$fixture" "$exit_status" \
      >>"$evidence/$mode-overlay-exits.txt"
  done
  terminal_record_command "$mode-parity-match" - - -- \
    cmp "$evidence/$mode-parity-original-run.stdout" \
      "$evidence/$mode-parity-overlay-run.stdout"
  if (( retain_emitted )); then
    terminal_record_command "$mode-emitted-check" "$mode-emitted-check.stdout" "$mode-emitted-check.stderr" -- \
      python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_terminal_emitted_evidence.py" \
      "$directory" "$evidence" "$mode"
  fi
}

compile_mode normal
terminal_record_command prefix-oracle-input prefix-oracle-input.stdout prefix-oracle-input.stderr -- \
  awk '/^COORD_ORACLE / { if (++rows > 2) next } { print }' \
  "$evidence/normal-run.stdout"
terminal_record_command prefix-oracle oracle.stdout oracle.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_t1_prefix_oracle.py" \
  "$evidence/prefix-oracle-input.stdout"
terminal_record_command second-oracle second-oracle.stdout second-oracle.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_terminal_composer_oracle.py" \
  "$evidence/normal-run.stdout"
terminal_record_command repeat-run repeat-run.stdout repeat-run.stderr -- \
  env ESHKOL_ARENA_POISON=1 "$tmp/normal/composer-terminal"
terminal_record_command repeat-feature-off-run repeat-feature-off-run.stdout repeat-feature-off-run.stderr -- \
  env ESHKOL_ARENA_POISON=1 "$tmp/normal/composer-feature-off"
terminal_record_command repeat-no-stderr - - -- test ! -s "$evidence/repeat-run.stderr"
terminal_record_command repeat-feature-off-no-stderr - - -- \
  test ! -s "$evidence/repeat-feature-off-run.stderr"
compile_mode sanitize
terminal_record_command normal-repeat-match - - -- \
  cmp "$evidence/normal-run.stdout" "$evidence/repeat-run.stdout"
terminal_record_command normal-sanitize-match - - -- \
  cmp "$evidence/normal-run.stdout" "$evidence/sanitize-run.stdout"
terminal_record_command feature-off-repeat-match - - -- \
  cmp "$evidence/normal-feature-off-run.stdout" \
    "$evidence/repeat-feature-off-run.stdout"
terminal_record_command feature-off-sanitize-match - - -- \
  cmp "$evidence/normal-feature-off-run.stdout" \
    "$evidence/sanitize-feature-off-run.stdout"
terminal_record_command git-diff-check - - -- git -C "$PROJECT_ROOT" diff --check
terminal_record_command closure-hash closure.sha256 - -- \
  sha256sum "$PROJECT_ROOT/native/g3c4_p2g2_terminal_composer_source_closure.txt"
terminal_record_command receipts-verify receipts-verify.stdout receipts-verify.stderr -- \
  python3 "$PROJECT_ROOT/tests/g3c4/check_terminal_command_receipts.py" \
  "$evidence" --allow-current
if (( retain_emitted )); then
  printf '%s\n' 'HOLD: compiler artifacts and imports retained; successful-zero no-allocation and complete launcher lifecycle remain unproved' \
    >"$evidence/emitted-acceptance.status"
fi
cat "$evidence/normal-run.stdout"
