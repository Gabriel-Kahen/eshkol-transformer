#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command cp
require_command cmp
require_command env
require_command grep
require_command nm
require_command rg
require_command strings
require_command timeout
require_command tr

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
i2_runner="$(eshkol_build_dir)/eshkol-run"
i2_compiler_timeout="${I2_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${i2_compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "I2_COMPILER_TIMEOUT_SECONDS must be a positive integer"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-i2.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT
artifact_dir="$(project_build_dir)/i2"
library="${artifact_dir}/libeshkol_transformer_f32.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
wave2_object="${artifact_dir}/i2_wave2.o"
wave2_library="${artifact_dir}/libeshkol_transformer_wave2.a"
wave2_evidence="${wave2_object}.evidence"

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)

[[ -r "${library}" && -r "${artifact_dir}/f32_tensor.o" ]] || \
  die "canonical I2 archive or object is missing"
[[ "$(ar t "${library}")" == "f32_tensor.o" ]] || \
  die "canonical I2 archive has unexpected members"
[[ -r "${k1_library}" ]] || die "canonical K1 archive is missing"
[[ -r "${wave2_object}" && -r "${wave2_library}" ]] || \
  die "canonical Wave 2 aggregate is missing"
[[ "$(ar t "${wave2_library}")" == "i2_wave2.o" ]] || \
  die "canonical Wave 2 archive has unexpected members"
cmp "${PROJECT_ROOT}/native/i2_wave2_defined_symbols.txt" \
  "${wave2_evidence}/global-defined.txt"
cmp "${PROJECT_ROOT}/native/i2_wave2_undefined_symbols.txt" \
  "${wave2_evidence}/undefined.txt"
cmp "${PROJECT_ROOT}/native/i2_wave2_public_exports.txt" \
  "${wave2_evidence}/package-exports.txt"
cmp "${PROJECT_ROOT}/native/i2_wave2_public_strings.txt" \
  "${wave2_evidence}/public-strings.txt"
sed -e 's/^[^:]*://' -e 's/\\//g' "${wave2_evidence}/private.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${temporary_dir}/wave2-source-closure.txt"
cmp "${PROJECT_ROOT}/native/i2_wave2_source_closure.txt" \
  "${temporary_dir}/wave2-source-closure.txt"

for source in test_f32_tensor test_f32_parameter; do
  "${cc}" "${cflags[@]}" "${PROJECT_ROOT}/tests/i2/${source}.c" \
    "${library}" "${k1_library}" -o "${temporary_dir}/${source}"
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 90s \
      "${temporary_dir}/${source}" >"${temporary_dir}/${source}-${run}.stdout"
  done
  cmp "${temporary_dir}/${source}-1.stdout" \
    "${temporary_dir}/${source}-2.stdout"
done
grep -E '^I2 storage PASS: [0-9]+ ' \
  "${temporary_dir}/test_f32_tensor-1.stdout" >/dev/null
grep -E '^I2 parameter PASS: [0-9]+ ' \
  "${temporary_dir}/test_f32_parameter-1.stdout" >/dev/null

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests/i2/header_cpp.cpp" \
  "${library}" "${k1_library}" -o "${temporary_dir}/header-cpp"
"${temporary_dir}/header-cpp"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/i2/parameter_header_cpp.cpp" \
  "${library}" "${k1_library}" -o "${temporary_dir}/parameter-header-cpp"
"${temporary_dir}/parameter-header-cpp"

if nm -g --defined-only --format=posix "${library}" | \
    grep -E '^eshkol_transformer_kernel_provider_v1[[:space:]]'; then
  die "I2 archive defines the forbidden canonical K1 provider symbol"
fi

i2_runtime_dir="${temporary_dir}/runtime"
mkdir -p "${i2_runtime_dir}" "${temporary_dir}/run-a" \
  "${temporary_dir}/run-b"
runtime_cflags=(
  "${cflags[@]}" -fPIC -fvisibility=hidden -fno-common
  -fstack-protector-all
)
"${cc}" "${runtime_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${i2_runtime_dir}/i2_wave2_native_bridge.o"
"${cc}" "${runtime_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${i2_runtime_dir}/p1_identity.o"
for runtime_source in data_io checkpoint_io kernel_abi i64_tensor \
    t1_i64_shell f32_tensor; do
  "${cc}" "${runtime_cflags[@]}" \
    -c "${PROJECT_ROOT}/native/${runtime_source}.c" \
    -o "${i2_runtime_dir}/${runtime_source}.o"
done
ar rcsD "${i2_runtime_dir}/libeshkol_transformer_i2_runtime.a" \
  "${i2_runtime_dir}"/*.o
"${cc}" "${cflags[@]}" "${PROJECT_ROOT}/tests/i2/test_wave2_bridge.c" \
  "${i2_runtime_dir}/libeshkol_transformer_i2_runtime.a" \
  -o "${temporary_dir}/test-wave2-bridge"
for run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/test-wave2-bridge" \
    >"${temporary_dir}/test-wave2-bridge-${run}.stdout"
done
cmp "${temporary_dir}/test-wave2-bridge-1.stdout" \
  "${temporary_dir}/test-wave2-bridge-2.stdout"
grep -Fx 'I2 Wave2 bridge PASS: 435 checks' \
  "${temporary_dir}/test-wave2-bridge-1.stdout" >/dev/null

compile_i2_integration() {
  local cache=$1 output=$2 log=$3
  mkdir -p "${cache}"
  if ! timeout --foreground --signal=TERM --kill-after=5s \
      "${i2_compiler_timeout}s" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${cache}" \
      "${i2_runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" -L "${i2_runtime_dir}" \
      --lib eshkol_transformer_i2_runtime \
      "${PROJECT_ROOT}/tests/i2/p1_integration.esk" \
      -o "${output}" >"${log}" 2>&1; then
    sed -n '1,260p' "${log}" >&2
    die "I2 production Eshkol integration compilation failed"
  fi
  [[ -x "${output}" ]] || die "I2 integration executable is missing"
  ! rg -F 'ERROR:' "${log}" >/dev/null || \
    die "I2 compiler reported ERROR while returning success"
}

compile_i2_integration "${temporary_dir}/cache-a" \
  "${temporary_dir}/run-a/i2-integration" "${temporary_dir}/compile-a.log"
compile_i2_integration "${temporary_dir}/cache-b" \
  "${temporary_dir}/run-b/i2-integration" "${temporary_dir}/compile-b.log"
cmp "${temporary_dir}/run-a/i2-integration" \
  "${temporary_dir}/run-b/i2-integration"
for run in a b; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/run-${run}/i2-integration" \
    >"${temporary_dir}/integration-${run}.stdout"
done
cmp "${temporary_dir}/integration-a.stdout" \
  "${temporary_dir}/integration-b.stdout"
grep -Fx 'I2/P1 integration PASS: 30 checks' \
  "${temporary_dir}/integration-a.stdout" >/dev/null

for evidence_name in global-defined.txt package-exports.txt undefined.txt \
    expected-undefined.txt public-strings.txt readelf-symbols.txt nm.txt \
    strings.txt private.d link.map allowlist-provenance.tsv; do
  [[ -s "${wave2_evidence}/${evidence_name}" ]] || \
    die "I2 aggregate evidence omits ${evidence_name}"
done
[[ "$(wc -l <"${wave2_evidence}/global-defined.txt")" == 47 ]] || \
  die "I2 aggregate must preserve the exact 47-global Wave 1 surface"
[[ "$(wc -l <"${wave2_evidence}/package-exports.txt")" == 41 ]] || \
  die "I2 aggregate must preserve the exact 41 package exports"
grep -Fx $'package_policy\ti2-wave2-aggregate' \
  "${wave2_evidence}/allowlist-provenance.tsv" >/dev/null
nm -s "${wave2_library}" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${temporary_dir}/wave2-archive-index.txt"
if grep -E 'et_i2_|et_f32_|et_p1_private_|et_e1b_private_|i2-(provider|native|carrier)' \
    "${temporary_dir}/wave2-archive-index.txt" >/dev/null; then
  die "I2 aggregate archive index exposes a localized authority"
fi

run_i2_boundary_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${temporary_dir}/boundary-cache/${cache_name}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${temporary_dir}/boundary-cache/${cache_name}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${i2_compiler_timeout}s" "${i2_runner}" "$@"
}

for private_binding in i2-provider-name i2-provider i2-carrier \
    i2-native-owned-release i2-storage-identical-request-internal \
    i2-state-tensor-checksum-internal i2-public-module-zero-grad! \
    tensor-provider-preflight-internal state-dict-tensor-borrow-begin-internal; do
  if run_i2_boundary_compiler "private-${private_binding}" \
      --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
      -e "(begin (require transformer.module) ${private_binding})" \
      >"${temporary_dir}/private-${private_binding}.stdout" \
      2>"${temporary_dir}/private-${private_binding}.stderr"; then
    die "I2 aggregate exposed private binding ${private_binding}"
  fi
  grep -F "${private_binding}" \
    "${temporary_dir}/private-${private_binding}.stderr" >/dev/null || \
    die "I2 private-binding negative failed for the wrong reason: ${private_binding}"
done

for negative_stem in negative_public_provider_internal \
    negative_public_module_internal negative_public_handle_internal \
    negative_public_state_entry_internal negative_public_state_dict_internal; do
  negative_source="${PROJECT_ROOT}/tests/p1/${negative_stem}.esk"
  for phase in source object aot; do
    negative_output="${temporary_dir}/${negative_stem}-${phase}"
    negative_args=(--strict-types --no-stdlib -I "${PROJECT_ROOT}/lib"
                   -L "${artifact_dir}" --lib eshkol_transformer_wave2)
    case "${phase}" in
      source) negative_args+=(-r "${negative_source}") ;;
      object) negative_args+=(--emit-object "${negative_source}"
                              -o "${negative_output}.o") ;;
      aot) negative_args+=("${negative_source}" -o "${negative_output}") ;;
    esac
    if run_i2_boundary_compiler \
        "authority-${negative_stem}-${phase}" "${negative_args[@]}" \
        >"${negative_output}.stdout" 2>"${negative_output}.stderr"; then
      die "fresh-cache I2 aggregate ${phase} admitted ${negative_stem}"
    fi
    [[ ! -e "${negative_output}" && ! -e "${negative_output}.o" ]] || \
      die "rejected I2 authority probe published ${negative_stem} ${phase} output"
  done
done

mkdir -p "${temporary_dir}/shadow/transformer"
printf '(error "hostile I2 private root loaded")\n' \
  >"${temporary_dir}/shadow/i2_wave2_root.esk"
printf '(error "hostile checkpoint implementation loaded")\n' \
  >"${temporary_dir}/shadow/transformer/checkpoint_internal.esk"
ESHKOL_PATH="${temporary_dir}/shadow" ESHKOL_LIB_DIR="${temporary_dir}/shadow" \
E1B_COMPILER_TIMEOUT_SECONDS="${i2_compiler_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" \
    "${temporary_dir}/hostile-build"
cmp "${wave2_object}" "${temporary_dir}/hostile-build/i2_wave2.o"
for deterministic_evidence in global-defined.txt package-exports.txt \
    undefined.txt expected-undefined.txt public-strings.txt private.d link.map; do
  cmp "${wave2_evidence}/${deterministic_evidence}" \
    "${temporary_dir}/hostile-build/i2_wave2.o.evidence/${deterministic_evidence}"
done
if grep -F "${temporary_dir}/shadow" \
    "${temporary_dir}/hostile-build/i2_wave2.o.evidence/private.d" >/dev/null; then
  die "I2 build admitted a hostile module path"
fi

reject_i2_builder_input() {
  local label=$1
  local expected=$2
  shift 2
  if E1B_COMPILER_TIMEOUT_SECONDS="${i2_compiler_timeout}" \
      "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" "$@" \
      >"${temporary_dir}/${label}.stdout" \
      2>"${temporary_dir}/${label}.stderr"; then
    die "I2 builder admitted ${label}"
  fi
  grep -F "${expected}" \
    "${temporary_dir}/${label}.stderr" >/dev/null || \
    die "I2 ${label} rejection reported the wrong reason"
}
cp "${PROJECT_ROOT}/native/i2_wave2_root.esk" \
  "${temporary_dir}/copied-i2-root.esk"
for rejected_root in "${temporary_dir}/copied-i2-root.esk" \
    "${wave2_object}" \
    "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk"; do
  label="rejected-root-$(basename -- "${rejected_root}")"
  expected_rejection='repository package components require their exact repository-owned private root'
  if [[ "${rejected_root}" == \
        "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk" ]]; then
    expected_rejection='P1 wider undefined-symbol policy requires the exact reviewed input tuple'
  fi
  reject_i2_builder_input "${label}" "${expected_rejection}" \
    "${rejected_root}" \
    "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/i2_wave2_private_renames.txt" \
    "${PROJECT_ROOT}/native/i2_wave2_public_exports.txt" \
    "${temporary_dir}/${label}.o" \
    "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
  [[ ! -e "${temporary_dir}/${label}.o" && \
     ! -e "${temporary_dir}/${label}.o.evidence" ]] || \
    die "rejected I2 builder input published an artifact"
done

if "${cc}" -r -Wl,--whole-archive "${wave2_library}" "${wave2_library}" \
    -Wl,--no-whole-archive -o "${temporary_dir}/duplicate-authority.o" \
    >"${temporary_dir}/duplicate-authority.stdout" \
    2>"${temporary_dir}/duplicate-authority.stderr"; then
  die "linker admitted duplicate I2 aggregate authority"
fi
grep -E 'multiple definition.*et_e1b_error_(predicate|category)_v1' \
  "${temporary_dir}/duplicate-authority.stderr" >/dev/null || \
  die "duplicate I2 authority rejection did not identify E1 ownership"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-k1.sh" \
  "${temporary_dir}/sanitized-k1" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" \
  "${temporary_dir}/sanitized-i2" sanitize-test
for source in test_f32_tensor test_f32_parameter; do
  "${cc}" "${cflags[@]}" -DET_F32_TENSOR_TESTING \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "${PROJECT_ROOT}/tests/i2/${source}.c" \
    "${temporary_dir}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${temporary_dir}/sanitized-k1/libeshkol_transformer_k1.a" \
    -o "${temporary_dir}/${source}-sanitized"
  ASAN_OPTIONS=detect_leaks="${I2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/${source}-sanitized" >/dev/null
done
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_F32_TENSOR_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/tests/i2/test_wave2_bridge.c" \
  "${temporary_dir}/sanitized-i2/libeshkol_transformer_f32.a" \
  "${temporary_dir}/sanitized-k1/libeshkol_transformer_k1.a" \
  -o "${temporary_dir}/test-wave2-bridge-sanitized"
ASAN_OPTIONS=detect_leaks="${I2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${temporary_dir}/test-wave2-bridge-sanitized" >/dev/null

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/f32_tensor.h"; then
  die "I2 production path contains a forbidden Python/PyTorch reference"
fi

printf 'I2 PASS: C/C++ ABI, exact-f32 storage, gradients, transactions, K1 views, determinism, and sanitizers\n'
