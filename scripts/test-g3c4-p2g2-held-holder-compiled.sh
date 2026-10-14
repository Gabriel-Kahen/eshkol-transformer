#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$PROJECT_ROOT/scripts/g3c4-held-holder-eshkol-compile.sh"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.g3c4.test_p2_g2_held_holder_eshkol_source \
  tests.g3c4.test_p2_g2_held_holder_compiled_source \
  tests.g3c4.test_p2_g2_held_holder_symbol_gate
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
  nm -g --defined-only "$directory/objects/owner-on.o" \
    >"$evidence/$mode-holder-on-nm.txt"
  python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-held-holder-symbols.py" \
    on "$evidence/$mode-holder-on-nm.txt" \
    >"$evidence/$mode-holder-on-symbols.txt"
  nm -g --defined-only "$directory/objects/owner-off.o" \
    >"$evidence/$mode-holder-off-nm.txt"
  python3 "$PROJECT_ROOT/scripts/check-g3c4-p2g2-held-holder-symbols.py" \
    off "$evidence/$mode-holder-off-nm.txt"
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
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_begin_capture_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_witness_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_abort_attempt_v1 \\
  -Wl,--wrap=et_g3c4_private_p2g2_holder_end_after_busy_v1
WRAPPER
  chmod 0500 "$directory/cxx-wrap"
  python3 "$PROJECT_ROOT/tests/g3c4/p2_g2_held_holder_phase_overlay.py" \
    "$PROJECT_ROOT" "$directory/overlay"
  for fixture in on off; do
    local overlay_arg=
    local input="$PROJECT_ROOT/tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk"
    if [[ "$fixture" == on ]]; then
      input="$PROJECT_ROOT/tests/g3c4/p2_g2_held_holder_compiled_test.esk"
      overlay_arg="$directory/overlay"
    fi
    compile_eshkol_fixture "$fixture" "$input" "$directory" "$evidence" \
      "$mode" "$runner" "$cxx" "$PROJECT_ROOT" "$overlay_arg"
  done
  python3 - "$PROJECT_ROOT" "$directory/on.d" "$directory/off.d" \
    "$evidence" "$mode" "$directory/overlay" <<'PY'
from pathlib import Path
import hashlib
import subprocess
import sys
from scripts.g3c4_held_holder_depfile import dependencies

root = Path(sys.argv[1]).resolve()
on_depfile, off_depfile, evidence, mode = (
    Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
overlay_dir = Path(sys.argv[6]).resolve(strict=True)
overrides = {
    (overlay_dir / name).resolve(strict=True): f'native/{name}'
    for name in ('g3c4_p2g2_terminal_held_holder_root.esk',
                 'g3c4_p2g2_terminal_held_holder_extension.esk')
}
assert len(overrides) == 2
root_overlay, extension_overlay = overrides
assert root_overlay.read_bytes() == (root / overrides[root_overlay]).read_bytes()
assert extension_overlay.read_bytes() != (root / overrides[extension_overlay]).read_bytes()
assert all(f'(held-phase {phase})' in extension_overlay.read_text()
           for phase in range(30, 43))

on = dependencies(on_depfile, root, overrides)
off = dependencies(off_depfile, root, {})
expected = (root / 'native/g3c4_p2g2_held_holder_compiled_eshkol_deps.txt').read_text().splitlines()
assert on == expected, (on, expected)
assert 'native/g3c4_p2g2_terminal_composer_extension.esk' in off, off
assert not any('held_holder' in name for name in off), off

closure = (root / 'native/g3c4_p2g2_terminal_held_holder_source_closure.txt').read_text().splitlines()
assert len(closure) == 198 and len(closure) == len(set(closure))
assert {'native/g3c4_p2g2_terminal_held_holder_root.esk',
        'native/g3c4_p2g2_terminal_held_holder_extension.esk'}.issubset(closure)
non_eshkol = [name for name in closure if not name.endswith('.esk')]
assert len(non_eshkol) == 162
required_transitive = {
    'native/e1b_error_consumer_private.esk',
    'lib/transformer/error_internal.esk',
    'lib/transformer/error_core.esk',
    'native/x1_config_private.esk',
    'internal/p1/lib/transformer/module.esk',
    'internal/c1/lib/transformer/persistence_policy_internal.esk',
    'internal/c1/lib/transformer/checkpoint_internal.esk',
    'internal/c1/lib/c1_sha256.esk',
}
assert required_transitive.issubset(on) and required_transitive.issubset(closure)
for name in non_eshkol:
    path = (root / name).resolve(strict=True)
    assert path.is_relative_to(root) and path.is_file(), name
subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', '--',
                *non_eshkol], check=True, stdout=subprocess.DEVNULL)
subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--',
                *non_eshkol], check=True)
(evidence / f'{mode}-on-eshkol-deps.txt').write_text(
    ''.join(name + '\n' for name in on))
(evidence / f'{mode}-off-eshkol-deps.txt').write_text(
    ''.join(name + '\n' for name in off))
(evidence / f'{mode}-non-eshkol-sources.sha256').write_text(
    ''.join(f'{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n'
            for name in non_eshkol))
PY
  for fixture in on off; do
    local expected='^G3-C4 held-holder compiled PASS: checks=[1-9][0-9]*$'
    [[ "$fixture" == off ]] && \
      expected='^G3-C4 P2/G2 terminal off PASS: checks=[1-9][0-9]*$'
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$directory/held-$fixture" normal \
      >"$evidence/$mode-$fixture.stdout" 2>"$evidence/$mode-$fixture.stderr" \
      4>"$evidence/$mode-$fixture.trace"
    [[ "$fixture" != on ]] || test -s "$evidence/$mode-$fixture.trace"
    test ! -s "$evidence/$mode-$fixture.stderr"
    rg "$expected" "$evidence/$mode-$fixture.stdout" >/dev/null
  done
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 "$directory/held-on" normal \
    >"$evidence/$mode-repeat.stdout" 2>"$evidence/$mode-repeat.stderr" \
    4>"$evidence/$mode-repeat.trace"
  test -s "$evidence/$mode-repeat.trace"
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
      2>"$evidence/$mode-$fixture.stderr" \
      4>"$evidence/$mode-$fixture.trace" || status=$?
    test -s "$evidence/$mode-$fixture.trace"
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
  "$PROJECT_ROOT/native/g3c4_p2g2_held_holder_compiled_eshkol_deps.txt" \
  >"$evidence/closure.sha256"
printf 'P2/G2 held-holder compiled normal/repeat/sanitizer/off/fail-stop PASS\n' \
  | tee "$evidence/result.txt"
