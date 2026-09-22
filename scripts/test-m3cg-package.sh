#!/usr/bin/env bash
# Fixed development-only composition. This is not an E1B package-policy input API.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in python3 ar cmp nm objcopy timeout; do require_command "${command}"; done
m3cg_dir="$(project_build_dir)/m3cg-package"
mkdir -p "${m3cg_dir}"
m3cg_source="$(eshkol_source_dir)"
m3cg_build="$(eshkol_build_dir)"
m3cg_cc="$(tsv_value "${m3cg_build}/eshkol-transformer-provenance.tsv" cc_path)"
m3cg_cxx="$(tsv_value "${m3cg_build}/eshkol-transformer-provenance.tsv" cxx_path)"
m3cg_env=(env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH
 -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX -u COMPILER_PATH
 -u LIBRARY_PATH -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX
 -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0
 ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${m3cg_cxx}")
m3cg_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
 -fPIC -fvisibility=hidden -fno-common -ffp-contract=off -fexcess-precision=standard
 -frounding-math -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
m3cg_sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
 src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
 src/eshkol_transformer/m3_call_f32_integration.c tests/m3cg/model_harness.c
 native/n2_primitives_provider.c native/n3k_primitives_provider.c native/a2_attention_provider.c)
m3cg_compile() {
 "${m3cg_env[@]}" XDG_CACHE_HOME="${m3cg_pass}/cache" \
 timeout --foreground --signal=TERM --kill-after=5s "${M3CG_COMPILER_TIMEOUT_SECONDS:-600}s" \
 "${m3cg_build}/eshkol-run" --strict-types --no-stdlib "$@"
}
python3 -m unittest -v tests.m3cg.test_closure
for language in c c++; do
 m3cg_standard=c11
 [[ "${language}" == c ]] || m3cg_standard=c++17
 "${m3cg_env[@]}" "${m3cg_cc}" -x "${language}" -std="${m3cg_standard}" -Wall -Wextra -Werror -Wpedantic \
 -I "${PROJECT_ROOT}/include" -c "${PROJECT_ROOT}/tests/m3cg/header_consumer.c" -o "${m3cg_dir}/header-${language}.o"
 nm -u --format=posix "${m3cg_dir}/header-${language}.o" | awk '{print $1}' | LC_ALL=C sort >"${m3cg_dir}/header-${language}.undefined"
done
printf 'et_g3t_model_pins_begin_internal\net_g3t_model_pins_check_internal\net_g3t_model_pins_end_internal\n' >"${m3cg_dir}/header.expected"
cmp "${m3cg_dir}/header.expected" "${m3cg_dir}/header-c.undefined"
cmp "${m3cg_dir}/header.expected" "${m3cg_dir}/header-c++.undefined"
for repetition in normal 1 2; do
 m3cg_pass="${m3cg_dir}/fresh-${repetition}"
 m3cg_root=tests/m3cg/private_root.esk
 m3cg_bridge=tests/m3cg/package_bridge.c
 m3cg_sources[6]=tests/m3cg/model_harness.c
 m3cg_test_flags=(-DET_F32_TENSOR_TESTING)
 if [[ "${repetition}" == normal ]]; then
  m3cg_root=tests/m3cg/normal_root.esk
  m3cg_bridge=native/m3_package_bridge.c
  m3cg_sources[6]=src/eshkol_transformer/m3_model.c
  m3cg_test_flags=()
 fi
 rm -rf -- "${m3cg_pass}"
 mkdir -p "${m3cg_pass}/cache"
 (cd "${m3cg_pass}"; m3cg_compile -I "${PROJECT_ROOT}/internal/p1/lib" \
 -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/t1/lib" \
 -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
 --shared-lib --dump-ir --emit-depfile "${m3cg_pass}/private.d" \
 "${PROJECT_ROOT}/${m3cg_root}" -o private)
 "${m3cg_env[@]}" "${m3cg_cc}" -c -x ir "${m3cg_pass}/private.ll" -o "${m3cg_pass}/private.o"
 cat "${PROJECT_ROOT}/native/e1b_private_renames.txt" \
 "${PROJECT_ROOT}/native/x1_config_private_renames.txt" \
 "${PROJECT_ROOT}/native/i2_wave2_p1_renames.txt" \
 "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
 "${PROJECT_ROOT}/native/t1_wave1_private_renames.txt" \
 "${PROJECT_ROOT}/native/i2_wave2_private_renames.txt" >"${m3cg_pass}/renames.txt"
 sed 's/^m3t-public-/g3t-checked-m3t-/' "${PROJECT_ROOT}/native/m3_package_private_renames.txt" >>"${m3cg_pass}/renames.txt"
 if [[ "${repetition}" != normal ]]; then printf 'm3cg-run et_m3cg_test_run_cabi_v1\n' >>"${m3cg_pass}/renames.txt"; fi
 objcopy --redefine-syms="${m3cg_pass}/renames.txt" "${m3cg_pass}/private.o"
 "${m3cg_env[@]}" "${m3cg_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" -I "${m3cg_source}/inc" \
 -MMD -MF "${m3cg_pass}/error.d" -c "${PROJECT_ROOT}/native/e1b_error_consumer_bridge.c" -o "${m3cg_pass}/error.o"
 "${m3cg_env[@]}" "${m3cg_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" -I "${m3cg_source}/inc" \
 -I "${PROJECT_ROOT}/src" -DET_M3T_PACKAGE_BUILD -MMD -MF "${m3cg_pass}/bridge.d" \
 -c "${PROJECT_ROOT}/${m3cg_bridge}" -o "${m3cg_pass}/bridge.o"
 m3cg_objects=("${m3cg_pass}/private.o" "${m3cg_pass}/error.o" "${m3cg_pass}/bridge.o")
 for index in "${!m3cg_sources[@]}"; do
  m3cg_compile_flags=("${m3cg_flags[@]}")
  if (( index >= 7 )); then
   m3cg_compile_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fPIC -O2 -I "${PROJECT_ROOT}/include")
   if (( index < 9 )); then m3cg_compile_flags+=(-fexcess-precision=standard -fno-fast-math -fstack-protector-all); fi
  fi
  "${m3cg_env[@]}" "${m3cg_cc}" "${m3cg_compile_flags[@]}" "${m3cg_test_flags[@]}" -MMD -MF "${m3cg_pass}/native-${index}.d" \
  -c "${PROJECT_ROOT}/${m3cg_sources[index]}" -o "${m3cg_pass}/native-${index}.o"
  m3cg_objects+=("${m3cg_pass}/native-${index}.o")
 done
 python3 "${PROJECT_ROOT}/tests/m3cg/check_closure.py" "${m3cg_pass}" "${m3cg_source}"
 "${m3cg_cxx}" -r "${m3cg_objects[@]}" -o "${m3cg_pass}/combined.o"
 { cat "${PROJECT_ROOT}/native/m3_package_defined_symbols.txt"; if [[ "${repetition}" != normal ]]; then printf 'et_m3cg_test_run_v1\n'; fi; } | LC_ALL=C sort >"${m3cg_pass}/exports.txt"
 nm -g --defined-only --format=posix "${m3cg_pass}/combined.o" | \
 awk 'NR==FNR {allowed[$1]=1;next} !($1 in allowed) {print $1}' "${m3cg_pass}/exports.txt" - >"${m3cg_pass}/localize.txt"
 objcopy --localize-symbols="${m3cg_pass}/localize.txt" "${m3cg_pass}/combined.o"
 nm -g --defined-only --format=posix "${m3cg_pass}/combined.o" | awk '{print $1}' | LC_ALL=C sort >"${m3cg_pass}/globals.txt"
 cmp "${m3cg_pass}/exports.txt" "${m3cg_pass}/globals.txt"
 nm -u --format=posix "${m3cg_pass}/combined.o" | awk '{print $1}' | LC_ALL=C sort >"${m3cg_pass}/undefined.txt"
 { cat "${PROJECT_ROOT}/native/m3_package_undefined_symbols.txt"; if [[ "${repetition}" != normal ]]; then printf 'eshkol_display_value\neshkol_runtime_current_output_fp\nfputc\n'; fi; } | LC_ALL=C sort -u >"${m3cg_pass}/expected-undefined.txt"
 cmp "${m3cg_pass}/expected-undefined.txt" "${m3cg_pass}/undefined.txt"
 for symbol in et_g3t_model_pins_begin_internal et_g3t_model_pins_check_internal et_g3t_model_pins_end_internal; do
  nm --format=posix "${m3cg_pass}/combined.o" | awk -v name="${symbol}" '$1==name && $2=="t" {found=1} END {exit !found}'
 done
 ar rcsD "${m3cg_pass}/libm3cg.a" "${m3cg_pass}/combined.o"
 if [[ "${repetition}" == normal ]]; then
  if nm --format=posix "${m3cg_pass}/combined.o" | grep -E 'et_m3cg_test_|et_f32_.*test_|et_m3t_test_|m3_call_test_'; then die "normal composition contains test hooks"; fi
  m3cg_compile -I "${PROJECT_ROOT}/lib" -L "${m3cg_pass}" --lib m3cg "${PROJECT_ROOT}/tests/m3cg/public_normal.esk" -o "${m3cg_pass}/caller"
  printf 'M3CG-NORMAL-PASS\n' >"${m3cg_pass}/expected"
 else
  for symbol in et_m3cg_test_acquire_v1 et_m3cg_test_check_v1 et_m3cg_test_end_v1; do
   nm --format=posix "${m3cg_pass}/combined.o" | awk -v name="${symbol}" '$1==name && $2=="t" {found=1} END {exit !found}'
  done
  m3cg_compile -L "${m3cg_pass}" --lib m3cg "${PROJECT_ROOT}/tests/m3cg/caller.esk" -o "${m3cg_pass}/caller"
  printf 'M3CG-SOURCE-AOT-PASS\n' >"${m3cg_pass}/expected"
 fi
 timeout --foreground --signal=TERM --kill-after=5s 180s "${m3cg_pass}/caller" >"${m3cg_pass}/stdout" 2>"${m3cg_pass}/stderr"
 cmp "${m3cg_pass}/expected" "${m3cg_pass}/stdout"
 [[ ! -s "${m3cg_pass}/stderr" ]] || die "M3CG witness emitted stderr"
done
for output in private.ll private.o combined.o libm3cg.a caller stdout; do
 cmp "${m3cg_dir}/fresh-1/${output}" "${m3cg_dir}/fresh-2/${output}"
done
# A post-localization caller cannot regain source-private or test-only authority.
m3cg_pass="${m3cg_dir}/fresh-normal"
for symbol in et_g3t_model_pins_begin_internal et_g3t_model_pins_check_internal et_g3t_model_pins_end_internal et_m3cg_test_run_v1; do
 printf '(extern i64 hidden :real %s)\n(display (hidden))\n' "${symbol}" >"${m3cg_dir}/negative.esk"
 m3cg_compile --compile-only "${m3cg_dir}/negative.esk" -o "${m3cg_dir}/negative.o"
 nm -u --format=posix "${m3cg_dir}/negative.o" | awk -v name="${symbol}" '$1==name {found=1} END {exit !found}'
 if m3cg_compile -L "${m3cg_pass}" --lib m3cg "${m3cg_dir}/negative.esk" -o "${m3cg_dir}/negative" >"${m3cg_dir}/negative.log" 2>&1; then die "private symbol escaped: ${symbol}"; fi
 grep -F "${symbol}" "${m3cg_dir}/negative.log" >/dev/null || die "private negative failed for another reason"
done
if "${m3cg_cxx}" -r "${m3cg_dir}/fresh-normal/combined.o" "${m3cg_dir}/fresh-1/combined.o" -o "${m3cg_dir}/duplicate.o" >"${m3cg_dir}/duplicate.log" 2>&1; then die "duplicate registry aggregates linked"; fi
grep -E 'multiple definition|duplicate symbol' "${m3cg_dir}/duplicate.log" >/dev/null || die "duplicate test failed for another reason"
printf 'M3CG private source/AOT/package gate passed\n'
