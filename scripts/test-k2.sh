#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar awk cmp cp env grep ldd ln nm readelf rg sha256sum sort strings timeout tr wc; do
  require_command "${command}"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
k2_runner="$(eshkol_build_dir)/eshkol-run"
k2_compiler_timeout="${K2_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${k2_compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "K2_COMPILER_TIMEOUT_SECONDS must be a positive integer"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-k2.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT
artifact_dir="$(project_build_dir)/k2"
wave2_object="${artifact_dir}/k2_wave2.o"
wave2_library="${artifact_dir}/libeshkol_transformer_wave2.a"
wave2_evidence="${wave2_object}.evidence"

[[ -r "${wave2_object}" && -r "${wave2_library}" ]] || \
  die "canonical K2 Wave 2 aggregate is missing"
[[ "$(ar t "${wave2_library}")" == "k2_wave2.o" ]] || \
  die "canonical K2 archive must contain exactly k2_wave2.o"
for evidence_name in global-defined.txt package-exports.txt undefined.txt \
    expected-undefined.txt public-strings.txt source-closure.txt \
    native-source-closure.txt readelf-symbols.txt nm.txt strings.txt \
    private.d link.map allowlist-provenance.tsv; do
  [[ -s "${wave2_evidence}/${evidence_name}" ]] || \
    die "K2 aggregate evidence omits ${evidence_name}"
done
cmp "${PROJECT_ROOT}/native/k2_wave2_defined_symbols.txt" \
  "${wave2_evidence}/global-defined.txt"
cmp "${PROJECT_ROOT}/native/k2_wave2_undefined_symbols.txt" \
  "${wave2_evidence}/undefined.txt"
cmp "${PROJECT_ROOT}/native/k2_wave2_public_exports.txt" \
  "${wave2_evidence}/package-exports.txt"
cmp "${PROJECT_ROOT}/native/k2_wave2_public_strings.txt" \
  "${wave2_evidence}/public-strings.txt"
cmp "${PROJECT_ROOT}/native/k2_wave2_source_closure.txt" \
  "${wave2_evidence}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/k2_wave2_native_source_closure.txt" \
  "${wave2_evidence}/native-source-closure.txt"
grep -Fx $'package_policy\tk2-wave2-aggregate' \
  "${wave2_evidence}/allowlist-provenance.tsv" >/dev/null
for manifest in k2_wave2_defined_symbols.txt \
    k2_wave2_public_exports.txt k2_wave2_public_strings.txt \
    k2_wave2_private_renames.txt k2_wave2_undefined_symbols.txt; do
  LC_ALL=C sort -c "${PROJECT_ROOT}/native/${manifest}"
done
awk 'NF != 2 { bad = 1 } END { exit bad ? 1 : 0 }' \
  "${PROJECT_ROOT}/native/k2_wave2_private_renames.txt" || \
  die "K2 private rename manifest must contain exact symbol pairs"

[[ "$(wc -l <"${wave2_evidence}/global-defined.txt")" == 59 ]] || \
  die "K2 aggregate must expose exactly 59 global definitions"
[[ "$(wc -l <"${wave2_evidence}/package-exports.txt")" == 53 ]] || \
  die "K2 aggregate must expose exactly 53 package exports"
[[ "$(wc -l <"${wave2_evidence}/public-strings.txt")" == 59 ]] || \
  die "K2 aggregate must contain exactly 59 public-name strings"
[[ "$(wc -l <"${wave2_evidence}/source-closure.txt")" == 15 ]] || \
  die "K2 aggregate must have exactly 15 trusted Eshkol sources"
rename_count="$(wc -l <\
  "${PROJECT_ROOT}/native/k2_wave2_private_renames.txt")"
[[ "${rename_count}" == 12 ]] || \
  die "K2 aggregate must have exactly 12 private facade renames"
if grep -Ev '^et_e1b_(error|public)_[a-z0-9_]+_v1$' \
    "${wave2_evidence}/global-defined.txt" >/dev/null; then
  die "K2 aggregate exposes a global outside the E1B public namespace"
fi
if nm -g --defined-only --format=posix "${wave2_library}" | \
    grep -E '^eshkol_transformer_kernel_provider_v1[[:space:]]'; then
  die "K2 archive defines the forbidden canonical K1 provider symbol"
fi
nm -s "${wave2_library}" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${temporary_dir}/archive-index.txt"
if grep -E 'et_(k2|i2|f32|p1|kernel)_|et_e1b_private_|k2-(public|private)' \
    "${temporary_dir}/archive-index.txt" >/dev/null; then
  die "K2 archive index exposes a localized authority"
fi

run_k2_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${temporary_dir}/compiler-cache/${cache_name}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${temporary_dir}/compiler-cache/${cache_name}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${k2_compiler_timeout}s" "${k2_runner}" "$@"
}

for repetition in 1 2; do
  mkdir -p "${temporary_dir}/public-contract-${repetition}"
  run_k2_compiler "public-contract-${repetition}" \
    --strict-types --no-stdlib --compile-only \
    -I "${PROJECT_ROOT}/lib" \
    "${PROJECT_ROOT}/tests/k2/compile_public_api.esk" \
    -o "${temporary_dir}/public-contract-${repetition}/api.o" \
    >"${temporary_dir}/public-contract-${repetition}.log" 2>&1
  mkdir -p "${temporary_dir}/public-run-${repetition}"
  run_k2_compiler "public-aot-${repetition}" \
    --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
    "${PROJECT_ROOT}/tests/k2/public_runtime.esk" \
    -o "${temporary_dir}/public-run-${repetition}/k2-public-runtime" \
    >"${temporary_dir}/public-aot-${repetition}.log" 2>&1
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/public-run-${repetition}/k2-public-runtime" \
    >"${temporary_dir}/public-run-${repetition}.stdout"
done
cmp "${temporary_dir}/public-contract-1/api.o" \
  "${temporary_dir}/public-contract-2/api.o"
grep '^et_e1b_public_k2_' \
  "${PROJECT_ROOT}/native/k2_wave2_public_exports.txt" \
  >"${temporary_dir}/expected-public-bridge-undefined.txt"
nm -u --format=posix "${temporary_dir}/public-contract-1/api.o" | \
  awk '$1 ~ /^et_e1b_public_k2_/ { print $1 }' | LC_ALL=C sort -u \
  >"${temporary_dir}/public-bridge-undefined.txt"
cmp "${temporary_dir}/expected-public-bridge-undefined.txt" \
  "${temporary_dir}/public-bridge-undefined.txt"
[[ "$(wc -l <"${temporary_dir}/public-bridge-undefined.txt")" == 12 ]] || \
  die "K2 public contract object must use exactly 12 public bridge symbols"
if nm --format=posix "${temporary_dir}/public-contract-1/api.o" | \
    awk '{ print $1 }' | \
    grep -E '^et_(k2_private|e1b_private_k2)_' >/dev/null; then
  die "K2 public contract object references localized native authority"
fi
cmp "${temporary_dir}/public-run-1/k2-public-runtime" \
  "${temporary_dir}/public-run-2/k2-public-runtime"
cmp "${temporary_dir}/public-run-1.stdout" \
  "${temporary_dir}/public-run-2.stdout"
grep -Fx 'k2-public-runtime:v1 45' \
  "${temporary_dir}/public-run-1.stdout" >/dev/null
ldd "${temporary_dir}/public-run-1/k2-public-runtime" \
  >"${temporary_dir}/public-runtime.ldd"
if grep -Ei 'python|torch' "${temporary_dir}/public-runtime.ldd" >/dev/null; then
  die "K2 production executable links a Python/PyTorch runtime"
fi

report_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -I "${PROJECT_ROOT}/include"
)
"${cc}" "${report_cflags[@]}" \
  "${PROJECT_ROOT}/tests/k2/report_i2.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" -lm \
  -o "${temporary_dir}/report-i2"
for repetition in 1 2; do
  LC_ALL=C timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/report-i2" \
    >"${temporary_dir}/report-i2-${repetition}.json"
done
cmp "${temporary_dir}/report-i2-1.json" \
  "${temporary_dir}/report-i2-2.json"
[[ "$(wc -c <"${temporary_dir}/report-i2-1.json")" == 3059 ]] || \
  die "K2 genuine I2/K1 report is not exactly 3,059 bytes"
report_sha256="$(sha256sum "${temporary_dir}/report-i2-1.json" | awk '{print $1}')"
[[ "${report_sha256}" == \
   742800ea988627d9093f8fe394c8ad2be801e35413e20bbcad816f2431be86cb ]] || \
  die "K2 genuine I2/K1 report SHA-256 drifted: ${report_sha256}"
printf 'K2 REPORT GOLDEN PASS: 3059 bytes sha256=%s\n' "${report_sha256}"

private_runtime_dir="${temporary_dir}/private-runtime"
mkdir -p "${private_runtime_dir}"
private_native_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "$(eshkol_source_dir)/inc"
)
"${cc}" "${private_native_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${private_runtime_dir}/i2_wave2_native_bridge.o"
"${cc}" "${private_native_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${private_runtime_dir}/p1_identity.o"
private_runtime_objects=(
  "${private_runtime_dir}/i2_wave2_native_bridge.o"
  "${private_runtime_dir}/p1_identity.o"
)
for private_source in data_io checkpoint_io kernel_abi i64_tensor \
    t1_i64_shell f32_tensor; do
  "${cc}" "${private_native_cflags[@]}" \
    -c "${PROJECT_ROOT}/native/${private_source}.c" \
    -o "${private_runtime_dir}/${private_source}.o"
  private_runtime_objects+=("${private_runtime_dir}/${private_source}.o")
done
"${cc}" "${private_native_cflags[@]}" -DET_K2_TESTING \
  -c "${PROJECT_ROOT}/native/k2_capabilities.c" \
  -o "${private_runtime_dir}/k2_capabilities.o"
private_runtime_objects+=("${private_runtime_dir}/k2_capabilities.o")
ar rcsD "${private_runtime_dir}/libeshkol_transformer_k2_private_test.a" \
  "${private_runtime_objects[@]}"
for test_hook in et_k2_test_runtime_drop_v1 et_k2_test_fork_v1 \
    et_k2_test_wait_child_v1 et_k2_test_exit_child_v1; do
  nm -g --defined-only "${private_runtime_dir}/libeshkol_transformer_k2_private_test.a" | \
    grep -E "[[:space:]]T ${test_hook}$" >/dev/null || \
    die "K2 private test archive omits ${test_hook}"
  if nm -a "${wave2_object}" | grep -E "[[:space:]]${test_hook}$" >/dev/null; then
    die "K2 production aggregate contains private test hook ${test_hook}"
  fi
done
run_k2_compiler private-identity-aot \
  --strict-types --optimize 0 --no-stdlib \
  -I "${PROJECT_ROOT}/native" \
  -I "${PROJECT_ROOT}/internal/p1/lib" \
  -I "${PROJECT_ROOT}/internal/c1/lib" \
  -I "${PROJECT_ROOT}/internal/t1/lib" \
  -I "${PROJECT_ROOT}/src" \
  -L "${private_runtime_dir}" --lib eshkol_transformer_k2_private_test \
  "${PROJECT_ROOT}/tests/k2/private_identity_runtime.esk" \
  -o "${private_runtime_dir}/private-identity" \
  >"${private_runtime_dir}/compile.stdout" \
  2>"${private_runtime_dir}/compile.stderr"
timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${private_runtime_dir}/private-identity" \
  >"${private_runtime_dir}/run.stdout"
grep -Fx 'K2 PRIVATE IDENTITY PASS: 12 checks' \
  "${private_runtime_dir}/run.stdout" >/dev/null

mkdir -p "${temporary_dir}/arena-retention"
run_k2_compiler "arena-retention-aot" \
  --strict-types --no-stdlib -O 2 -I "${PROJECT_ROOT}/lib" \
  -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
  "${PROJECT_ROOT}/tests/k2/arena_retention.esk" \
  -o "${temporary_dir}/arena-retention/k2-arena-retention" \
  >"${temporary_dir}/arena-retention/compile.log" 2>&1
for horizon in 1024 8192; do
  ESHKOL_ARENA_REPORT=1 \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${temporary_dir}/arena-retention/k2-arena-retention" "${horizon}" \
    >"${temporary_dir}/arena-retention/${horizon}.stdout" \
    2>"${temporary_dir}/arena-retention/${horizon}.stderr"
  grep -Fx "K2 ARENA RETENTION PASS: ${horizon} lexical-region iterations" \
    "${temporary_dir}/arena-retention/${horizon}.stdout" >/dev/null
  retained_bytes="$(awk -F= '/global_total_allocated_bytes=/{print $2}' \
    "${temporary_dir}/arena-retention/${horizon}.stderr" | tail -1)"
  [[ "${retained_bytes}" =~ ^[0-9]+$ ]] || \
    die "K2 ${horizon} run omitted the Eshkol retained root-arena counter"
  printf '%s\n' "${retained_bytes}" \
    >"${temporary_dir}/arena-retention/${horizon}.bytes"
done
arena_1024="$(<"${temporary_dir}/arena-retention/1024.bytes")"
arena_8192="$(<"${temporary_dir}/arena-retention/8192.bytes")"
[[ "${arena_1024}" == "${arena_8192}" ]] || \
  die "K2 retained root arena grew from ${arena_1024} to ${arena_8192} bytes"
printf 'K2 ARENA RETENTION PASS: 1024=%s bytes 8192=%s bytes slope=0\n' \
  "${arena_1024}" "${arena_8192}"

arity_stems=(
  discover request verified require report_entries entry_name entry_status
  entry_implementation entry_version entry_constraints entry_deterministic
  entry_evidence
)
arity_symbols=(
  capability-discover capability-request capability-verified?
  capability-require capability-report-entries capability-entry-name
  capability-entry-status capability-entry-implementation
  capability-entry-version capability-entry-constraints
  capability-entry-deterministic? capability-entry-evidence
)
arity_expected=(0 5 3 3 1 1 1 1 1 1 1 1)
for index in "${!arity_stems[@]}"; do
  stem="${arity_stems[${index}]}"
  output="${temporary_dir}/negative-${stem}.o"
  log="${temporary_dir}/negative-${stem}.log"
  if run_k2_compiler "negative-${stem}" \
      --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/k2/negative_${stem}_arity.esk" \
      -o "${output}" >"${log}" 2>&1; then
    die "K2 wrong-arity fixture unexpectedly compiled: ${stem}"
  fi
  grep -F "Arity mismatch: ${arity_symbols[${index}]} expects ${arity_expected[${index}]} arguments" \
    "${log}" >/dev/null || \
    die "K2 wrong-arity diagnostic drifted: ${arity_symbols[${index}]}"
  [[ ! -e "${output}" ]] || \
    die "K2 wrong-arity fixture emitted ${output}"
done

private_bindings=(
  k2-public-capability-discover
  k2-runtime-ensure!
  k2-native-runtime-ensure
  k2-report-query
)
for private_binding in "${private_bindings[@]}"; do
  private_output="${temporary_dir}/private-${private_binding}"
  if run_k2_compiler "private-${private_binding}" \
      --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
      -e "(begin (require transformer.capabilities) ${private_binding})" \
      >"${private_output}.stdout" 2>"${private_output}.stderr"; then
    die "K2 public facade exposed trusted binding ${private_binding}"
  fi
  grep -F "${private_binding}" "${private_output}.stderr" >/dev/null || \
    die "K2 private-binding rejection failed for the wrong reason"
done

guessed_private_source="${temporary_dir}/guessed-private-native.esk"
printf '%s\n' \
  '(extern i64 guessed-k2-runtime-pid :real et_k2_private_runtime_pid_v1)' \
  '(guessed-k2-runtime-pid)' >"${guessed_private_source}"
run_k2_compiler guessed-private-object \
  --strict-types --no-stdlib --emit-object \
  "${guessed_private_source}" \
  -o "${temporary_dir}/guessed-private-native.o" \
  >"${temporary_dir}/guessed-private-object.stdout" \
  2>"${temporary_dir}/guessed-private-object.stderr"
nm -u --format=posix "${temporary_dir}/guessed-private-native.o" | \
  awk '{ print $1 }' | \
  grep -Fx 'et_k2_private_runtime_pid_v1' >/dev/null
if run_k2_compiler guessed-private-aot \
    --strict-types --no-stdlib -L "${artifact_dir}" \
    --lib eshkol_transformer_wave2 \
    "${guessed_private_source}" \
    -o "${temporary_dir}/guessed-private-native" \
    >"${temporary_dir}/guessed-private-aot.stdout" \
    2>"${temporary_dir}/guessed-private-aot.stderr"; then
  die "K2 public archive linked a guessed localized native seam"
fi
grep -F 'et_k2_private_runtime_pid_v1' \
  "${temporary_dir}/guessed-private-aot.stderr" >/dev/null || \
  die "K2 guessed-native rejection failed for the wrong reason"
[[ ! -e "${temporary_dir}/guessed-private-native" ]] || \
  die "rejected K2 guessed-native AOT left an executable"

native_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -DET_K2_TESTING -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -x c++-header -fsyntax-only \
  "${PROJECT_ROOT}/native/k2_capabilities_internal.h"
"${cc}" "${native_cflags[@]}" \
  "${PROJECT_ROOT}/tests/k2/test_capabilities_native.c" \
  "${PROJECT_ROOT}/native/k2_capabilities.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" -lm \
  -o "${temporary_dir}/test-k2-native"
for repetition in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/test-k2-native" \
    >"${temporary_dir}/test-k2-native-${repetition}.stdout"
done
cmp "${temporary_dir}/test-k2-native-1.stdout" \
  "${temporary_dir}/test-k2-native-2.stdout"
grep -Fx 'K2 native capability checks passed' \
  "${temporary_dir}/test-k2-native-1.stdout" >/dev/null
"${cc}" "${native_cflags[@]}" \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/k2/test_capabilities_native.c" \
  "${PROJECT_ROOT}/native/k2_capabilities.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" -lm \
  -o "${temporary_dir}/test-k2-native-sanitized"
ASAN_OPTIONS=detect_leaks="${K2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${temporary_dir}/test-k2-native-sanitized" >/dev/null

allocator_link_flags=(
  -Wl,--wrap=malloc -Wl,--wrap=calloc
  -Wl,--wrap=realloc -Wl,--wrap=free
)
allocator_measurement='K2 K1 allocation measurement: calls=58 requested=2487 live-blocks=54 live-bytes=2407 peak-bytes=2487'
for allocator_mode in normal sanitized; do
  allocator_flags=(-DET_K2_ALLOCATOR_WRAP)
  if [[ "${allocator_mode}" == sanitized ]]; then
    allocator_flags+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${cc}" "${native_cflags[@]}" "${allocator_flags[@]}" \
    "${PROJECT_ROOT}/tests/k2/test_capabilities_native.c" \
    "${PROJECT_ROOT}/native/k2_capabilities.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" -lm \
    "${allocator_link_flags[@]}" \
    -o "${temporary_dir}/test-k2-allocator-${allocator_mode}"
  ASAN_OPTIONS=detect_leaks="${K2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/test-k2-allocator-${allocator_mode}" \
    >"${temporary_dir}/test-k2-allocator-${allocator_mode}.stdout"
  grep -Fx "${allocator_measurement}" \
    "${temporary_dir}/test-k2-allocator-${allocator_mode}.stdout" >/dev/null
  grep -Fx 'K2 native capability checks passed' \
    "${temporary_dir}/test-k2-allocator-${allocator_mode}.stdout" >/dev/null
done

for rebuild in a b; do
  ESHKOL_PATH="${temporary_dir}/missing-${rebuild}" \
  ESHKOL_LIB_DIR="${temporary_dir}/missing-${rebuild}" \
  ESHKOL_JIT_CACHE_DIR="${temporary_dir}/cache-${rebuild}" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-k2.sh" \
      "${temporary_dir}/rebuild-${rebuild}"
done
cmp "${temporary_dir}/rebuild-a/k2_wave2.o" \
  "${temporary_dir}/rebuild-b/k2_wave2.o"
cmp "${temporary_dir}/rebuild-a/libeshkol_transformer_wave2.a" \
  "${temporary_dir}/rebuild-b/libeshkol_transformer_wave2.a"
for deterministic_evidence in global-defined.txt package-exports.txt \
    undefined.txt expected-undefined.txt public-strings.txt source-closure.txt \
    native-source-closure.txt private.d link.map allowlist-provenance.tsv; do
  cmp "${temporary_dir}/rebuild-a/k2_wave2.o.evidence/${deterministic_evidence}" \
    "${temporary_dir}/rebuild-b/k2_wave2.o.evidence/${deterministic_evidence}"
done
cmp "${wave2_object}" "${temporary_dir}/rebuild-a/k2_wave2.o"

mkdir -p "${temporary_dir}/shadow/transformer"
printf '%s\n' '(error "hostile K2 private root loaded")' \
  >"${temporary_dir}/shadow/k2_wave2_root.esk"
printf '%s\n' '(error "hostile capabilities facade loaded")' \
  >"${temporary_dir}/shadow/transformer/capabilities.esk"
ESHKOL_PATH="${temporary_dir}/shadow" \
ESHKOL_LIB_DIR="${temporary_dir}/shadow" \
E1B_COMPILER_TIMEOUT_SECONDS="${k2_compiler_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-k2.sh" \
    "${temporary_dir}/shadow-build"
cmp "${wave2_object}" "${temporary_dir}/shadow-build/k2_wave2.o"
if grep -F "${temporary_dir}/shadow" \
    "${temporary_dir}/shadow-build/k2_wave2.o.evidence/private.d" >/dev/null; then
  die "K2 build admitted a hostile module path"
fi

canonical_builder_args=(
  "${PROJECT_ROOT}/native/k2_wave2_root.esk"
  "${PROJECT_ROOT}/native/k2_wave2_package_bridge.c"
  "${PROJECT_ROOT}/native/k2_wave2_private_renames.txt"
  "${PROJECT_ROOT}/native/k2_wave2_public_exports.txt"
)
canonical_include_args=(
  "${PROJECT_ROOT}/internal/p1/lib"
  "${PROJECT_ROOT}/internal/c1/lib"
  "${PROJECT_ROOT}/internal/t1/lib"
  "${PROJECT_ROOT}/src"
)

reject_k2_builder_input() {
  local label=$1
  shift
  local root=$1 bridge=$2 renames=$3 exports=$4
  shift 4
  local output="${temporary_dir}/rejected-${label}.o"
  if E1B_COMPILER_TIMEOUT_SECONDS="${k2_compiler_timeout}" \
      /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
      "${root}" "${bridge}" "${renames}" "${exports}" "${output}" "$@" \
      >"${output}.stdout" 2>"${output}.stderr"; then
    die "K2 builder admitted ${label}"
  fi
  [[ ! -e "${output}" && ! -e "${output}.evidence" ]] || \
    die "rejected K2 builder input published ${label} output"
}

for input_kind in root bridge renames exports; do
  case "${input_kind}" in
    root) source=${canonical_builder_args[0]} ;;
    bridge) source=${canonical_builder_args[1]} ;;
    renames) source=${canonical_builder_args[2]} ;;
    exports) source=${canonical_builder_args[3]} ;;
  esac
  cp "${source}" "${temporary_dir}/copied-${input_kind}"
  copied_args=("${canonical_builder_args[@]}")
  case "${input_kind}" in
    root) copied_args[0]="${temporary_dir}/copied-${input_kind}" ;;
    bridge) copied_args[1]="${temporary_dir}/copied-${input_kind}" ;;
    renames) copied_args[2]="${temporary_dir}/copied-${input_kind}" ;;
    exports) copied_args[3]="${temporary_dir}/copied-${input_kind}" ;;
  esac
  reject_k2_builder_input "copied-${input_kind}" \
    "${copied_args[@]}" "${canonical_include_args[@]}"

  ln -s "${source}" "${temporary_dir}/symlink-${input_kind}"
  symlink_args=("${canonical_builder_args[@]}")
  case "${input_kind}" in
    root) symlink_args[0]="${temporary_dir}/symlink-${input_kind}" ;;
    bridge) symlink_args[1]="${temporary_dir}/symlink-${input_kind}" ;;
    renames) symlink_args[2]="${temporary_dir}/symlink-${input_kind}" ;;
    exports) symlink_args[3]="${temporary_dir}/symlink-${input_kind}" ;;
  esac
  reject_k2_builder_input "symlink-${input_kind}" \
    "${symlink_args[@]}" "${canonical_include_args[@]}"
done
reject_k2_builder_input all-symlink-inputs \
  "${temporary_dir}/symlink-root" "${temporary_dir}/symlink-bridge" \
  "${temporary_dir}/symlink-renames" "${temporary_dir}/symlink-exports" \
  "${canonical_include_args[@]}"

wrong_args=("${canonical_builder_args[@]}")
wrong_args[0]="${PROJECT_ROOT}/native/i2_wave2_root.esk"
reject_k2_builder_input wrong-root \
  "${wrong_args[@]}" "${canonical_include_args[@]}"
wrong_args=("${canonical_builder_args[@]}")
wrong_args[1]="${PROJECT_ROOT}/native/i2_wave2_package_bridge.c"
reject_k2_builder_input wrong-bridge \
  "${wrong_args[@]}" "${canonical_include_args[@]}"
wrong_args=("${canonical_builder_args[@]}")
wrong_args[2]="${PROJECT_ROOT}/native/i2_wave2_private_renames.txt"
reject_k2_builder_input wrong-renames \
  "${wrong_args[@]}" "${canonical_include_args[@]}"
wrong_args=("${canonical_builder_args[@]}")
wrong_args[3]="${PROJECT_ROOT}/native/i2_wave2_public_exports.txt"
reject_k2_builder_input wrong-exports \
  "${wrong_args[@]}" "${canonical_include_args[@]}"
relative_args=("${canonical_builder_args[@]}")
relative_args[0]="native/../native/k2_wave2_root.esk"
reject_k2_builder_input relative-escape-root \
  "${relative_args[@]}" "${canonical_include_args[@]}"

reject_k2_builder_input reordered-includes \
  "${canonical_builder_args[@]}" \
  "${canonical_include_args[1]}" "${canonical_include_args[0]}" \
  "${canonical_include_args[2]}" "${canonical_include_args[3]}"
reject_k2_builder_input removed-include \
  "${canonical_builder_args[@]}" \
  "${canonical_include_args[0]}" "${canonical_include_args[1]}" \
  "${canonical_include_args[2]}"
reject_k2_builder_input duplicate-include \
  "${canonical_builder_args[@]}" "${canonical_include_args[@]}" \
  "${canonical_include_args[3]}"
mkdir -p "${temporary_dir}/hostile-include"
reject_k2_builder_input added-hostile-include \
  "${canonical_builder_args[@]}" "${canonical_include_args[@]}" \
  "${temporary_dir}/hostile-include"
ln -s "${canonical_include_args[0]}" "${temporary_dir}/symlink-include"
reject_k2_builder_input symlink-include \
  "${canonical_builder_args[@]}" "${temporary_dir}/symlink-include" \
  "${canonical_include_args[1]}" "${canonical_include_args[2]}" \
  "${canonical_include_args[3]}"

override_output="${temporary_dir}/rejected-policy-override.o"
if E1B_PACKAGE_POLICY=base-e1b \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${canonical_builder_args[@]}" "${override_output}" \
    "${canonical_include_args[@]}" >/dev/null 2>&1; then
  die "K2 builder admitted an environment policy override"
fi
[[ ! -e "${override_output}" && ! -e "${override_output}.evidence" ]] || \
  die "rejected K2 policy override published output"

collision_libraries=(
  "${wave2_library}"
  "$(project_build_dir)/i2/libeshkol_transformer_wave2.a"
  "$(project_build_dir)/t2/libeshkol_transformer_wave2.a"
  "$(project_build_dir)/d2/libeshkol_transformer_wave2.a"
  "$(project_build_dir)/o2/libeshkol_transformer_wave2.a"
)
for collision_library in "${collision_libraries[@]}"; do
  [[ -r "${collision_library}" ]] || \
    die "affected aggregate is missing for K2 collision gate: ${collision_library}"
  collision_name="$(basename -- "$(dirname -- "${collision_library}")")"
  if "${cc}" -r -Wl,--whole-archive "${wave2_library}" \
      "${collision_library}" -Wl,--no-whole-archive \
      -o "${temporary_dir}/collision-${collision_name}.o" \
      >"${temporary_dir}/collision-${collision_name}.stdout" \
      2>"${temporary_dir}/collision-${collision_name}.stderr"; then
    die "linker admitted duplicate K2/aggregate authority: ${collision_name}"
  fi
  grep -E 'multiple definition.*et_e1b_(error|public)_' \
    "${temporary_dir}/collision-${collision_name}.stderr" >/dev/null || \
    die "K2 collision rejection did not identify duplicate E1B ownership"
  [[ ! -e "${temporary_dir}/collision-${collision_name}.o" ]] || \
    die "rejected K2 collision left a linked output"
done

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/lib/transformer/capabilities.esk" \
    "${PROJECT_ROOT}/native/k2_wave2_root.esk" \
    "${PROJECT_ROOT}/native/k2_wave2_extension.esk" \
    "${PROJECT_ROOT}/native/k2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/k2_wave2_public_wrappers.inc" \
    "${PROJECT_ROOT}/native/k2_capabilities.c" \
    "${PROJECT_ROOT}/native/k2_capabilities_internal.h"; then
  die "K2 production source contains a forbidden Python/PyTorch reference"
fi
if nm -u --format=posix "${wave2_object}" | \
    grep -Ei '(^|[[:space:]])(_?Py|python|torch)' >/dev/null; then
  die "K2 production object depends on a Python/PyTorch runtime"
fi

printf 'K2 PACKAGING PASS: exact 59/53/59 surface, closures, deterministic rebuilds, hostile inputs, and collision isolation\n'
