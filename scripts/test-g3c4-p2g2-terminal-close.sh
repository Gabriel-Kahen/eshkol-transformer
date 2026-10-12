#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/g3c4_proof_command.sh"
if [[ -n ${G3C4_PROOF_RECEIPTS:-} ]]; then
  export G3C4_PROOF_SCRIPT="${BASH_SOURCE[0]##*/}"
fi
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-p2g2-terminal-close-test}"
proof_command mkdir -p "$evidence"
proof_command python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-terminal-close.py" \
  >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 proof_command python3 -m unittest -v \
  tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
proof_command bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-t1-prefix-coordinator.sh" \
  "$evidence/prefix-predecessor" >"$evidence/prefix-predecessor.stdout"
proof_command bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-terminal-ids-clone.sh" \
  "$evidence/native-terminal-clone-predecessor" \
  >"$evidence/native-terminal-clone-predecessor.stdout"
tmp="$(proof_command mktemp -d "$evidence/build.XXXXXX")"
trap 'status=$?; if (( status == 0 )) && [[ -z ${G3C4_PROOF_RECEIPTS:-} ]]; then rm -rf -- "$tmp"; else
  printf "P2/G2 terminal composer failure retained: %s\n" "$tmp" >&2; fi' EXIT

compile_mode() {
  local mode=$1 directory="$tmp/$1" source
  local -a optimization=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi
  proof_command mkdir -p "$directory/objects" "$directory/cache"
  local -a common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all -ffp-contract=off
    -fexcess-precision=standard -fno-fast-math
    -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
    -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer"
    -I "$(eshkol_source_dir)/inc")
  proof_command "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
    -DET_G3C4_P2_G1_PENDING_PRIVATE \
    -DET_G3C4_P2_G2_FIRST_FRAME_PRIVATE \
    -DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE \
    -DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_close_native.c" \
    -o "$directory/objects/composer_native.o"
  proof_command "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/tests/m3t/kernel_fail_allocator.c" \
    -o "$directory/objects/kernel_fail_allocator.o"
  proof_command "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2_native_helpers.o"
  proof_command "$cxx" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    "${optimization[@]}" -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_close_allocation_shim.cpp" \
    -o "$directory/objects/call_entry_allocation_shim.o"
  proof_command "$cc" "${common[@]}" "${optimization[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "$PROJECT_ROOT/native/p1_identity.c" \
    -o "$directory/objects/p1_identity.o"
  for source in data_io checkpoint_io n3k_primitives_provider \
      g3n_primitives_provider n2_primitives_provider \
      g3c4_primitives_provider g3s_sampling_provider \
      a2_attention_provider a2_kv_cache; do
    proof_command "$cc" "${common[@]}" "${optimization[@]}" \
      -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE \
      -c "$PROJECT_ROOT/native/$source.c" \
      -o "$directory/objects/$source.o"
  done
  proof_command "$cc" "${common[@]}" "${optimization[@]}" \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_T1_I64_SHELL_TESTING \
    -c "$PROJECT_ROOT/native/t1_i64_shell.c" \
    -o "$directory/objects/t1_i64_shell.o"
  proof_command "$cc" "${common[@]}" "${optimization[@]}" -DET_I64_TENSOR_TESTING \
    -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \
    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    -o "$directory/objects/m3_i64_integration.o"
  proof_command "$cc" "${common[@]}" "${optimization[@]}" \
    -c "$PROJECT_ROOT/src/eshkol_transformer/m3_model.c" \
    -o "$directory/objects/m3_model.o"
  proof_command ar rcsD "$directory/libg3c4_p2g2_terminal_composer.a" \
    "$directory"/objects/*.o
  local sanitizer_link=
  [[ "$mode" == sanitize ]] && \
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  proof_command cat >"$directory/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
PROJECT_ROOT=$(printf '%q' "$PROJECT_ROOT")
source "\$PROJECT_ROOT/scripts/g3c4_proof_command.sh"
link_command=exec
if [[ -n \${G3C4_PROOF_RECEIPTS:-} ]]; then link_command=proof_command; fi
if [[ -n \${G3C4_PROOF_LINK_MAP:-} ]]; then
  set -- "\$@" "-Wl,-Map=\$G3C4_PROOF_LINK_MAP"
fi
"\$link_command" $(printf '%q' "$cxx") $sanitizer_link "\$@" \\
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
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1 \\
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1
WRAPPER
  proof_command chmod 0500 "$directory/cxx-wrap"
  local -a eshkol_args=(--strict-types --optimize 0 --no-stdlib
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
    -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native"
    -L "$directory" --lib g3c4_p2g2_terminal_composer)
  capture_emitted_fixture() {
    local fixture=$1 fixture_source=$2 raw
    raw="$directory/emitted/$fixture.raw.o"
    local relinked="$directory/emitted/$fixture.relinked"
    proof_command mkdir -p "$directory/emitted" "$directory/cache/emitted-$fixture"
    proof_command env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/emitted-$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$runner" "${eshkol_args[@]}" --emit-object --dump-ir \
      --emit-depfile "$directory/emitted/$fixture.d" \
      "$fixture_source" -o "$raw" \
      >"$evidence/$mode-$fixture-emit.stdout" \
      2>"$evidence/$mode-$fixture-emit.stderr"
    test -s "$raw" && test -s "$raw.bc" && test -s "$raw.ll" && \
      test -s "$directory/emitted/$fixture.d"
    proof_command env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/emitted-$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      G3C4_PROOF_LINK_MAP="$directory/emitted/$fixture.map" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$runner" "${eshkol_args[@]}" "$raw" -o "$relinked" \
      >"$evidence/$mode-$fixture-relink.stdout" \
      2>"$evidence/$mode-$fixture-relink.stderr"
    test -s "$directory/emitted/$fixture.map"
    proof_command cmp "$directory/composer-$fixture" "$relinked"
  }
  local fixture label expected fixture_source
  for fixture in terminal feature-off; do
    label="$mode"
    expected='^G3-C4 P2/G2 Eshkol terminal PASS: checks=[1-9][0-9]*$'
    fixture_source="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_close_test.esk"
    if [[ "$fixture" == feature-off ]]; then
      label="$mode-feature-off"
      expected='^G3-C4 P2/G2 terminal off PASS: checks=[1-9][0-9]*$'
      fixture_source="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_close_feature_off_test.esk"
    fi
    proof_command mkdir -p "$directory/cache/$fixture"
    proof_command env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$runner" "${eshkol_args[@]}" \
      "$fixture_source" \
      -o "$directory/composer-$fixture" \
      >"$evidence/$label-compile.stdout" \
      2>"$evidence/$label-compile.stderr"
    if [[ ${G3C4_PROOF_EMIT:-0} == 1 ]]; then
      capture_emitted_fixture "$fixture" "$fixture_source"
    fi
    proof_command "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/composer-$fixture" >"$evidence/$label-run.stdout" \
      2>"$evidence/$label-run.stderr"
    test ! -s "$evidence/$label-run.stderr"
    proof_command grep -E "$expected" "$evidence/$label-run.stdout" >/dev/null
  done

  proof_command python3 "$PROJECT_ROOT/tests/g3c4/generate_p2_g2_terminal_close_fault_overlays.py" \
    "$directory/overlays" >"$evidence/$mode-overlay-generation.stdout"
  (cd "$directory/overlays" && proof_command sha256sum -c SHA256SUMS) \
    >"$evidence/$mode-overlay-identity.stdout"
  proof_command cp "$directory/overlays/SHA256SUMS" "$evidence/$mode-overlay-SHA256SUMS"
  for fixture in parity-original parity-overlay unmarked decode-first decode-second \
      precommit postzero; do
    proof_command mkdir -p "$directory/cache/$fixture"
    proof_command env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache/$fixture" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 600s \
      "$runner" "${eshkol_args[@]}" \
      "$directory/overlays/$fixture.esk" \
      -o "$directory/composer-$fixture" \
      >"$evidence/$mode-$fixture-compile.stdout" \
      2>"$evidence/$mode-$fixture-compile.stderr"
    if [[ ${G3C4_PROOF_EMIT:-0} == 1 ]]; then
      capture_emitted_fixture "$fixture" "$directory/overlays/$fixture.esk"
    fi
    local exit_status=0
    proof_command "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/composer-$fixture" \
      3>"$evidence/$mode-$fixture.marker" \
      >"$evidence/$mode-$fixture-run.stdout" \
      2>"$evidence/$mode-$fixture-run.stderr" || exit_status=$?
    if [[ "$fixture" == postzero ]]; then
      test "$exit_status" -eq 134
      test "$(cat "$evidence/$mode-$fixture.marker")" = Z
      ! proof_command grep -Eq 'ILLEGAL_TERMINAL_RETURN|OVERLAY_PARITY|OVERLAY_ABORT|ESH_TERMINAL' \
        "$evidence/$mode-$fixture-run.stdout"
    else
      test "$exit_status" -eq 0
      test ! -s "$evidence/$mode-$fixture.marker"
      test ! -s "$evidence/$mode-$fixture-run.stderr"
      if [[ "$fixture" == parity-* ]]; then
        proof_command grep -Eq '^OVERLAY_PARITY [0-9]+ [0-9]+ 4$' \
          "$evidence/$mode-$fixture-run.stdout"
      elif [[ "$fixture" == unmarked ]]; then
        proof_command grep -Fx OVERLAY_UNMARKED \
          "$evidence/$mode-$fixture-run.stdout" >/dev/null
      else
        proof_command grep -Fx "OVERLAY_ABORT ${fixture#decode-}" \
          "$evidence/$mode-$fixture-run.stdout" >/dev/null
      fi
    fi
    printf '%s %s\n' "$fixture" "$exit_status" \
      >>"$evidence/$mode-overlay-exits.txt"
  done
  proof_command cmp "$evidence/$mode-parity-original-run.stdout" \
      "$evidence/$mode-parity-overlay-run.stdout"
}

compile_mode normal
proof_command awk '/^COORD_ORACLE / { if (++rows > 2) next } { print }' \
  "$evidence/normal-run.stdout" >"$evidence/prefix-oracle-input.stdout"
proof_command python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_t1_prefix_oracle.py" \
  "$evidence/prefix-oracle-input.stdout" >"$evidence/oracle.stdout"
proof_command python3 "$PROJECT_ROOT/tests/g3c4/check_p2_g2_terminal_close_oracle.py" \
  "$evidence/normal-run.stdout" >"$evidence/second-oracle.stdout"
ESHKOL_ARENA_POISON=1 proof_command "$tmp/normal/composer-terminal" \
  >"$evidence/repeat-run.stdout" 2>"$evidence/repeat-run.stderr"
ESHKOL_ARENA_POISON=1 proof_command "$tmp/normal/composer-feature-off" \
  >"$evidence/repeat-feature-off-run.stdout" \
  2>"$evidence/repeat-feature-off-run.stderr"
test ! -s "$evidence/repeat-run.stderr"
test ! -s "$evidence/repeat-feature-off-run.stderr"
compile_mode sanitize
proof_command cmp "$evidence/normal-run.stdout" "$evidence/repeat-run.stdout"
proof_command cmp "$evidence/normal-run.stdout" "$evidence/sanitize-run.stdout"
proof_command cmp "$evidence/normal-feature-off-run.stdout" \
    "$evidence/repeat-feature-off-run.stdout"
proof_command cmp "$evidence/normal-feature-off-run.stdout" \
    "$evidence/sanitize-feature-off-run.stdout"
proof_command git -C "$PROJECT_ROOT" diff --check
proof_command sha256sum "$PROJECT_ROOT/native/g3c4_p2g2_terminal_close_source_closure.txt" \
  >"$evidence/closure.sha256"
proof_command cat "$evidence/normal-run.stdout"
