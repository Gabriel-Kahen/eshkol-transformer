#!/usr/bin/env bash
# One separately instrumented exact-wrapper aggregate. Never a production policy.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env nm objcopy python3 timeout; do require_command "${command}"; done
cd "${PROJECT_ROOT}"
build="$(project_build_dir)"
canonical="${build}/m3"
output="${build}/m3-numerical"
[[ -s "${canonical}/m3_package.o" && -s "${build}/m3-reference/development.json" ]] || die "M3 numerical gate requires canonical package and generated reference prerequisites"
mkdir -p "${build}"
tmp="$(mktemp -d "${build}/.m3-numerical.XXXXXX")"
trap 'status=$?; if [[ "$status" != 0 ]]; then printf "M3 numerical failure evidence: %s\n" "${tmp}" >&2; else rm -rf -- "${tmp}"; fi' EXIT
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compile_timeout="${M3_COMPILER_TIMEOUT_SECONDS:-600}"
[[ "${compile_timeout}" =~ ^[1-9][0-9]*$ ]] || die "M3 compiler timeout must be positive"
clean=(env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH
       -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX
       -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE
       -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX)
compile() {
  local libraries=$1
  shift
  "${clean[@]}" -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${tmp}/cache" ESHKOL_LIB_DIR="${libraries}" ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compile_timeout}s" "${runner}" "$@"
}
# The canonical builder remains closed. Reuse its fixed source-path validator,
# supplying only the exact checked-in tuple; no caller-selected source or flags.
raw_private_root="${PROJECT_ROOT}/native/m3_package_root.esk"
raw_package_bridge="${PROJECT_ROOT}/native/m3_package_bridge.c"
raw_package_renames="${PROJECT_ROOT}/native/m3_package_private_renames.txt"
raw_public_exports="${PROJECT_ROOT}/native/m3_package_public_exports.txt"
raw_include_dirs=("${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib"
                  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src")
source "${PROJECT_ROOT}/scripts/m3-package-policy.sh"
nm -g --defined-only --format=posix "${canonical}/m3_package.o" | awk '{print $1}' | LC_ALL=C sort -u >"${tmp}/canonical-globals.txt"
cmp "${PROJECT_ROOT}/native/m3_package_defined_symbols.txt" "${tmp}/canonical-globals.txt"
if nm -a "${canonical}/m3_package.o" | grep -E 'et_m3_test_|et_f32_test_|et_i64_test_'; then
  die "test observer leaked into canonical M3 package"
fi
(cd "${tmp}" && compile "${PROJECT_ROOT}/lib" --strict-types --no-stdlib \
  -I "${raw_include_dirs[0]}" -I "${raw_include_dirs[1]}" -I "${raw_include_dirs[2]}" \
  -I "${raw_include_dirs[3]}" -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
  --shared-lib --dump-ir --emit-depfile "${tmp}/private.d" "${raw_private_root}" -o private \
  >"${tmp}/private-compile.log" 2>&1)
sed -e 's/^[^:]*://' -e 's/\\//g' "${tmp}/private.d" | tr -s '[:space:]' '\n' | \
  m3_normalize_source_dependencies >"${tmp}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/m3_package_source_closure.txt" "${tmp}/source-closure.txt"
"${clean[@]}" "${cc}" -c -x ir "${tmp}/private.ll" -o "${tmp}/private.o"
cat "${PROJECT_ROOT}/native/"{e1b_private_renames,x1_config_private_renames,i2_wave2_p1_renames,d1_e1b_private_renames,t1_wave1_private_renames,i2_wave2_private_renames,m3_package_private_renames}.txt >"${tmp}/renames.txt"
objcopy --redefine-syms="${tmp}/renames.txt" "${tmp}/private.o"
"${clean[@]}" "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all \
  -I "$(eshkol_source_dir)/inc" -I "${PROJECT_ROOT}/native" -MMD -MF "${tmp}/bridge.d" \
  -c "${PROJECT_ROOT}/native/e1b_error_consumer_bridge.c" -o "${tmp}/bridge.o"
"${clean[@]}" "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -DET_M3T_PACKAGE_BUILD -DET_F32_TENSOR_TESTING \
  -I "$(eshkol_source_dir)/inc" -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -I "${PROJECT_ROOT}/src" -MMD -MF "${tmp}/package-bridge.d" \
  -c "${raw_package_bridge}" -o "${tmp}/package-bridge.o"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
        -fPIC -fvisibility=hidden -fno-common -ffp-contract=off -fexcess-precision=standard -frounding-math
        -DET_M3_TESTING -DET_M3_PACKAGE_TESTING -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
        -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
         src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
         src/eshkol_transformer/m3t_f32_integration.c src/eshkol_transformer/m3_model.c)
objects=("${tmp}/private.o" "${tmp}/bridge.o" "${tmp}/package-bridge.o")
for index in "${!sources[@]}"; do
  "${clean[@]}" "${cc}" "${cflags[@]}" -MMD -MF "${tmp}/native-${index}.d" \
    -c "${PROJECT_ROOT}/${sources[$index]}" -o "${tmp}/native-${index}.o"
  objects+=("${tmp}/native-${index}.o")
done
for provider in n2 n3k a2; do
  "${clean[@]}" /usr/bin/bash "${PROJECT_ROOT}/scripts/build-${provider}.sh" "${tmp}/providers/${provider}" normal
done
while IFS= read -r object; do objects+=("${tmp}/providers/${object}"); done <"${PROJECT_ROOT}/native/m3_package_native_objects.txt"
depfiles=("${tmp}/bridge.d" "${tmp}/package-bridge.d")
for index in "${!sources[@]}"; do depfiles+=("${tmp}/native-${index}.d"); done
while IFS= read -r object; do depfiles+=("${tmp}/providers/${object%.o}.d"); done <"${PROJECT_ROOT}/native/m3_package_native_objects.txt"
verified_source="$(realpath -- "$(eshkol_source_dir)")"
{
  for depfile in "${depfiles[@]}"; do
    sed -e 's/^[^:]*://' -e 's/\\//g' "${depfile}" | tr -s '[:space:]' '\n' | while IFS= read -r dependency; do
      [[ -n "${dependency}" ]] || continue
      m3_check_native_dependency_path "${dependency}"
      dependency="$(realpath -- "${dependency}")"
      case "${dependency}" in
        "${verified_source}/"*) printf '.deps/eshkol-src/%s\n' "${dependency#"${verified_source}/"}" ;;
        "${PROJECT_ROOT}/"*) printf '%s\n' "${dependency#"${PROJECT_ROOT}/"}" ;;
        *) die "instrumented native dependency is outside reviewed roots" ;;
      esac
    done
  done
} | awk '!seen[$0]++' >"${tmp}/native-source-closure.txt"
cmp "${PROJECT_ROOT}/native/m3_package_native_source_closure.txt" "${tmp}/native-source-closure.txt"
"${clean[@]}" "${cxx}" -r "${objects[@]}" -o "${tmp}/instrumented.raw.o"
{
  cat "${PROJECT_ROOT}/native/m3_package_defined_symbols.txt"
  printf '%s\n' et_m3_test_last_gradient_copy_v1 et_m3_test_last_parameter_copy_v1 \
    et_m3_test_last_gradient_metadata_v1 et_m3_test_report_counts_v1
} | LC_ALL=C sort -u >"${tmp}/test-globals.txt"
[[ "$(wc -l <"${tmp}/test-globals.txt")" == 97 ]] || die "instrumented M3 must expose canonical93 plus4 exact observers"
cp "${tmp}/instrumented.raw.o" "${tmp}/m3_numerical.o"
objcopy --keep-global-symbols="${tmp}/test-globals.txt" "${tmp}/m3_numerical.o"
nm -g --defined-only --format=posix "${tmp}/m3_numerical.o" | awk '{print $1}' | LC_ALL=C sort -u >"${tmp}/actual-globals.txt"
cmp "${tmp}/test-globals.txt" "${tmp}/actual-globals.txt"
{ cat "${PROJECT_ROOT}/native/m3_package_undefined_symbols.txt"; printf 'printf\n'; } | LC_ALL=C sort -u >"${tmp}/expected-undefined.txt"
nm -u --format=posix "${tmp}/m3_numerical.o" | awk '{print $1}' | LC_ALL=C sort -u >"${tmp}/actual-undefined.txt"
cmp "${tmp}/expected-undefined.txt" "${tmp}/actual-undefined.txt"
ar rcsD "${tmp}/libeshkol_transformer_m3.a" "${tmp}/m3_numerical.o"
[[ "$(ar t "${tmp}/libeshkol_transformer_m3.a")" == m3_numerical.o ]] || die "instrumented archive must have one registry-owning member"
mkdir -p "${output}"
cp -a "${canonical}/facades" "${tmp}/facades"
python3 -m tests.m3.generate_instrumented_caller --output "${tmp}/numerical.esk"
compile_public() {
  local source=$1 binary=$2
  shift 2
  compile "${tmp}/facades" --strict-types --no-stdlib -I "${tmp}/facades" \
    "$@" --compile-only --emit-depfile "${binary}.d" "${source}" -o "${binary}.o" >"${binary}.compile.log" 2>&1
  python3 "${PROJECT_ROOT}/tests/m3/check_public_closure.py" "${tmp}" "${source}" "${binary}.d"
  compile "${tmp}/facades" --strict-types --no-stdlib -I "${tmp}/facades" \
    "$@" -L "${tmp}" --lib eshkol_transformer_m3 "${source}" -o "${binary}" >>"${binary}.compile.log" 2>&1
}
compile_public "${tmp}/numerical.esk" "${tmp}/numerical"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s "${tmp}/numerical" >"${tmp}/numerical.stdout" 2>"${tmp}/numerical.stderr"
python3 -m tests.m3.check_public_parity --reference "${build}/m3-reference/development.json" \
  --public-output "${tmp}/numerical.stdout" --instrumented >"${tmp}/parity.txt"
tail -1 "${tmp}/parity.txt"
printf 'M3 instrumented early numerical evidence: %s\n' "${tmp}"
cat "${PROJECT_ROOT}/tests/m3/arena_retention.esk" >"${tmp}/arena.esk"
printf '\n(extern i64 m3-test-native-counts :real et_m3_test_report_counts_v1)\n(check (= (m3-test-native-counts) 0))\n' >>"${tmp}/arena.esk"
compile_public "${tmp}/arena.esk" "${tmp}/arena" -O 2
for mode in forward logits vjp reset failure; do
  for horizon in 1024 8192; do
    ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 600s \
      "${tmp}/arena" "${horizon}" "${mode}" >"${tmp}/arena-${mode}-${horizon}.stdout" 2>"${tmp}/arena-${mode}-${horizon}.stderr"
    grep -Fx "M3-ARENA-RETENTION-PASS ${mode} ${horizon}" "${tmp}/arena-${mode}-${horizon}.stdout" >/dev/null
    grep '^M3 native ' "${tmp}/arena-${mode}-${horizon}.stdout" >/dev/null
  done
done
python3 "${PROJECT_ROOT}/tests/m3/check_native_retention.py" "${tmp}"
# Publish test evidence/artifacts only; normal package remains untouched.
for artifact in libeshkol_transformer_m3.a m3_numerical.o numerical numerical.esk numerical.stdout numerical.stderr parity.txt arena arena.esk source-closure.txt native-source-closure.txt actual-globals.txt actual-undefined.txt; do
  cp "${tmp}/${artifact}" "${output}/${artifact}"
done
cp -a "${tmp}/facades" "${output}/"
cp "${tmp}"/arena-*.stdout "${tmp}"/arena-*.stderr "${output}/"
cat "${tmp}/parity.txt"
printf 'M3 instrumented public-wrapper numerical/arena evidence: %s\n' "${output}"
