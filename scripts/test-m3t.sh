#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env nm timeout; do require_command "${command}"; done
m3t_time="${M3T_TIME_EXECUTABLE:-/usr/bin/time}"
[[ "${m3t_time}" == /* && -x "${m3t_time}" ]] || die "M3T supplementary RSS witness requires an absolute GNU time executable"
"${m3t_time}" --version | grep -F "GNU Time" >/dev/null || die "M3T supplementary RSS witness requires GNU time"
python3 -m unittest -v tests.m3t.test_package_contract
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3t-native.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3t-construction.sh"
m3t_test_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-m3t.XXXXXX")"
m3t_cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then
    rm -rf -- "${m3t_test_tmp}"
  else
    printf 'M3T failure evidence retained at %s\n' "${m3t_test_tmp}" >&2
  fi
}
trap m3t_cleanup EXIT
m3t_canonical="$(project_build_dir)/m3t"
m3t_i2_object="$(project_build_dir)/i2/i2_wave2.o"
m3t_caller_install="${m3t_test_tmp}/caller-install"
[[ -s "${m3t_canonical}/m3t_package.o" && -s "${m3t_canonical}/libeshkol_transformer_m3t.a" && \
   -s "${m3t_i2_object}" ]] || die "M3T and I2 canonical CI prerequisites are missing"
cmp "${PROJECT_ROOT}/native/m3t_package_defined_symbols.txt" \
  "${m3t_canonical}/m3t_package.o.evidence/global-defined.txt"
m3t_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
m3t_cxx="$(tsv_value "${m3t_provenance}" cxx_path)"
m3t_runner="$(eshkol_build_dir)/eshkol-run"
m3t_timeout="${M3T_COMPILER_TIMEOUT_SECONDS:-600}"
[[ "${m3t_timeout}" =~ ^[1-9][0-9]*$ ]] || die "M3T compiler timeout must be positive"
[[ -s "${PROJECT_ROOT}/tests/m3t/public_runtime.esk" ]] || die "M3T public transport witness missing"
m3t_compile() {
  local installed=$1
  shift
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
    -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${m3t_test_tmp}/cache" ESHKOL_LIB_DIR="${installed}/facades" \
    ESHKOL_CXX_COMPILER="${m3t_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${m3t_timeout}s" \
    "${m3t_runner}" --strict-types --no-stdlib -I "${installed}/facades" "$@"
}
# Imported diagnostic filenames are embedded in caller objects. Use the same
# clean installation path for each independently built package, without stripping.
m3t_install_for_caller() {
  rm -rf -- "${m3t_caller_install:?}"
  mkdir -p "${m3t_caller_install}"
  cp -a "$1/facades" "${m3t_caller_install}/facades"
  cp "$1/libeshkol_transformer_m3t.a" "${m3t_caller_install}/"
}
m3t_compile_public() {
  local installed=$1 source=$2 output=$3
  shift 3
  rm -f -- "${output}.d"
  m3t_compile "${installed}" "$@" --compile-only --emit-depfile "${output}.d" \
    "${source}" -o "${output}.o"
  python3 "${PROJECT_ROOT}/tests/m3t/check_public_closure.py" \
    "${installed}" "${source}" "${output}.d"
  m3t_compile "${installed}" "$@" -L "${installed}" --lib eshkol_transformer_m3t \
    "${source}" -o "${output}"
}
for repetition in 1 2; do
  installed="${m3t_test_tmp}/fresh-${repetition}"
  if [[ "${repetition}" == 2 ]]; then
    mkdir -p "${m3t_test_tmp}/hostile/eshkol_transformer"
    printf '#error hostile ambient include entered trusted compilation\n' \
      >"${m3t_test_tmp}/hostile/stdint.h"
    cp "${m3t_test_tmp}/hostile/stdint.h" \
      "${m3t_test_tmp}/hostile/eshkol_transformer/n3k_primitives_abi.h"
    env CPATH="${m3t_test_tmp}/hostile" C_INCLUDE_PATH="${m3t_test_tmp}/hostile" \
      CPLUS_INCLUDE_PATH="${m3t_test_tmp}/hostile" ESHKOL_PATH="${m3t_test_tmp}/hostile" \
      CLANG_CONFIG_FILE="${m3t_test_tmp}/hostile/forbidden.cfg" \
      /usr/bin/bash "${PROJECT_ROOT}/scripts/build-m3t.sh" "${installed}"
  else
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-m3t.sh" "${installed}"
  fi
  for pair in 'defined_symbols global-defined' 'public_exports package-exports' \
      'public_strings public-strings' 'undefined_symbols undefined' \
      'source_closure source-closure' 'native_source_closure native-source-closure' \
      'native_objects native-objects'; do
    read -r expected observed <<<"${pair}"
    cmp "${PROJECT_ROOT}/native/m3t_package_${expected}.txt" \
      "${installed}/m3t_package.o.evidence/${observed}.txt"
  done
  (cd "${installed}/facades" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
    >"${m3t_test_tmp}/facades-${repetition}.txt"
  cmp "${PROJECT_ROOT}/native/m3t_package_facades.txt" "${m3t_test_tmp}/facades-${repetition}.txt"
  m3t_install_for_caller "${installed}"
  m3t_compile "${m3t_caller_install}" --compile-only \
    "${PROJECT_ROOT}/tests/m3t/compile_api.esk" -o "${installed}/api.o"
  m3t_compile_public "${m3t_caller_install}" \
    "${PROJECT_ROOT}/tests/m3t/public_runtime.esk" "${installed}/public"
  if ! timeout --foreground --signal=TERM --kill-after=5s 120s "${installed}/public" \
      >"${installed}/public.stdout" 2>"${installed}/public.stderr"; then
    cat "${installed}/public.stdout" "${installed}/public.stderr" >&2
    die "M3T installed public witness failed"
  fi
  printf 'M3T-PUBLIC-TRANSPORT-PASS\n' >"${installed}/expected.stdout"
  cmp "${installed}/expected.stdout" "${installed}/public.stdout"
  if nm -u --format=posix "${installed}/api.o" | \
      grep -E 'et_(m3t|i2|f32|p1|kernel)_private|et_e1b_private_'; then
    die "M3T facade retained private authority"
  fi
  nm -u --format=posix "${installed}/api.o" | awk '$1 ~ /^et_e1b_public_m3t_/ {print $1}' | LC_ALL=C sort -u \
    >"${installed}/api-undefined.txt"
  grep '^et_e1b_public_m3t_' "${PROJECT_ROOT}/native/m3t_package_public_exports.txt" \
    >"${installed}/expected-api-undefined.txt"
  cmp "${installed}/expected-api-undefined.txt" "${installed}/api-undefined.txt"
  printf 'M3T fresh package %s: exact artifact and installed public witness passed\n' "${repetition}"
done
for output in m3t_package.o libeshkol_transformer_m3t.a api.o public public.stdout; do
  cmp "${m3t_test_tmp}/fresh-1/${output}" "${m3t_test_tmp}/fresh-2/${output}"
done
cmp "${m3t_canonical}/m3t_package.o" "${m3t_test_tmp}/fresh-1/m3t_package.o"
cmp "${m3t_canonical}/libeshkol_transformer_m3t.a" "${m3t_test_tmp}/fresh-1/libeshkol_transformer_m3t.a"
# Reversed import order still obtains the aggregate's one E1 registry.
{ printf '(require transformer.error_public)\n'; cat "${PROJECT_ROOT}/tests/m3t/public_runtime.esk"; } \
  >"${m3t_test_tmp}/reverse.esk"
m3t_install_for_caller "${m3t_test_tmp}/fresh-1"
m3t_compile_public "${m3t_caller_install}" "${m3t_test_tmp}/reverse.esk" "${m3t_test_tmp}/reverse"
"${m3t_test_tmp}/reverse" >"${m3t_test_tmp}/reverse.stdout"
cmp "${m3t_test_tmp}/fresh-1/public.stdout" "${m3t_test_tmp}/reverse.stdout"
# Local symbols cannot become callable merely by spelling a private extern.
for private in et_m3t_private_input_create_v1 et_e1b_private_m3t_input_create_cabi_v1 \
    et_p1_private_construction_begin_v1 et_f32_tensor_scoped_begin_internal; do
  printf '(extern ptr hidden :real %s)\n(display (hidden))\n' "${private}" \
    >"${m3t_test_tmp}/negative.esk"
  m3t_compile "${m3t_caller_install}" --compile-only \
    "${m3t_test_tmp}/negative.esk" -o "${m3t_test_tmp}/negative.o"
  nm -u --format=posix "${m3t_test_tmp}/negative.o" | \
    awk '{print $1}' | grep -Fx "${private}" >/dev/null || \
    die "M3T private-link negative omitted its intended symbol"
  if m3t_compile "${m3t_caller_install}" -L "${m3t_caller_install}" \
      --lib eshkol_transformer_m3t "${m3t_test_tmp}/negative.esk" \
      -o "${m3t_test_tmp}/negative" >"${m3t_test_tmp}/negative.log" 2>&1; then
    die "M3T private symbol escaped localization: ${private}"
  fi
  grep -F "${private}" "${m3t_test_tmp}/negative.log" >/dev/null || \
    die "M3T private symbol fixture failed for another reason"
done
for private in m3t-public-model-create m3t-workspace-contributions-internal \
    i2-construction-begin-internal module-construction-begin-internal; do
  printf '(require transformer.diagnostic_transport)\n%s\n' "${private}" \
    >"${m3t_test_tmp}/private-binding.esk"
  if m3t_compile "${m3t_caller_install}" --compile-only \
      "${m3t_test_tmp}/private-binding.esk" -o "${m3t_test_tmp}/private-binding.o" \
      >"${m3t_test_tmp}/private-binding.log" 2>&1; then
    die "M3T facade imported private binding ${private}"
  fi
  grep -F "${private}" "${m3t_test_tmp}/private-binding.log" >/dev/null || \
    die "M3T private-binding fixture rejected for another reason"
done
for entry in 'diagnostic-input-create 0' 'diagnostic-initializer-state 1' 'diagnostic-model-create 2'; do
  read -r public arity <<<"${entry}"
  printf '(require transformer.diagnostic_transport)\n(%s #f #f #f)\n' "${public}" \
    >"${m3t_test_tmp}/arity.esk"
  if m3t_compile "${m3t_caller_install}" --compile-only \
      "${m3t_test_tmp}/arity.esk" -o "${m3t_test_tmp}/arity.o" \
      >"${m3t_test_tmp}/arity.log" 2>&1; then
    die "M3T wrong arity was accepted for ${public}"
  fi
  grep -E "Arity mismatch: ${public} expects ${arity} arguments but got 3|function '${public}' expects ${arity} arguments, got 3" \
      "${m3t_test_tmp}/arity.log" >/dev/null || \
    die "M3T arity fixture rejected for another reason"
done
# Two registry objects are always rejected, regardless of archive lazy extraction.
if "${m3t_cxx}" -r "${m3t_test_tmp}/fresh-1/m3t_package.o" \
    "${m3t_test_tmp}/fresh-2/m3t_package.o" -o "${m3t_test_tmp}/duplicate.o" \
    >"${m3t_test_tmp}/duplicate.log" 2>&1; then
  die "M3T duplicate registry objects linked"
fi
grep -E 'multiple definition|duplicate symbol' "${m3t_test_tmp}/duplicate.log" >/dev/null || \
  die "M3T duplicate registry fixture failed for another reason"
if "${m3t_cxx}" -r "${m3t_test_tmp}/fresh-1/m3t_package.o" \
    "${m3t_i2_object}" -o "${m3t_test_tmp}/cross-aggregate.o" \
    >"${m3t_test_tmp}/cross-aggregate.log" 2>&1; then
  die "M3T and I2 registry objects linked into one process"
fi
grep -E 'multiple definition|duplicate symbol' "${m3t_test_tmp}/cross-aggregate.log" >/dev/null || \
  die "M3T cross-aggregate fixture failed for another reason"
# Success-only public full transport witnesses run in reclaimed poisoned regions.
m3t_compile_public "${m3t_caller_install}" "${PROJECT_ROOT}/tests/m3t/arena_retention.esk" \
  "${m3t_test_tmp}/arena-retention" -O 2
for horizon in 1024 8192; do
  ESHKOL_ARENA_REPORT=1 ESHKOL_ARENA_POISON=1 \
    "${m3t_time}" -f 'max_rss_kib=%M' timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${m3t_test_tmp}/arena-retention" "${horizon}" \
      >"${m3t_test_tmp}/arena-${horizon}.stdout" \
      2>"${m3t_test_tmp}/arena-${horizon}.stderr"
  grep -Fx "M3T-ARENA-RETENTION-PASS ${horizon}" \
    "${m3t_test_tmp}/arena-${horizon}.stdout" >/dev/null
  retained="$(awk -F= '/global_total_allocated_bytes=/{print $2}' \
    "${m3t_test_tmp}/arena-${horizon}.stderr" | tail -1)"
  [[ "${retained}" =~ ^[0-9]+$ ]] || die "M3T public run omitted retained arena accounting"
  peak_rss="$(awk -F= '/^max_rss_kib=/{print $2}' "${m3t_test_tmp}/arena-${horizon}.stderr")"
  [[ "${peak_rss}" =~ ^[0-9]+$ ]] || die "M3T public run omitted supplementary peak RSS"
  if [[ "${horizon}" == 1024 ]]; then
    arena_1024="${retained}"; rss_1024="${peak_rss}"
  else
    arena_8192="${retained}"; rss_8192="${peak_rss}"
  fi
done
[[ "${arena_1024}" == "${arena_8192}" ]] || \
  die "M3T retained arena grew from ${arena_1024} to ${arena_8192} bytes"
printf 'M3T public retained arena: 1024=%s 8192=%s bytes\n' "${arena_1024}" "${arena_8192}"
printf 'M3T supplementary peak RSS: 1024=%s 8192=%s KiB\n' "${rss_1024}" "${rss_8192}"
printf 'M3T packaging and installed-facade AOT gate passed\n'
