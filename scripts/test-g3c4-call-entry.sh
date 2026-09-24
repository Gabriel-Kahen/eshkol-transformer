#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar python3 timeout; do require_command "${command}"; done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-call-entry.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-call-entry-test}"
test_source="${G3C4_CALL_ENTRY_TEST_SOURCE:-${PROJECT_ROOT}/tests/g3c4/call_entry_test.esk}"
mkdir -p "${evidence}"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-call.XXXXXX")"
cleanup() {
  local result=$?
  if (( result != 0 )); then
    cat "${temporary_dir}/compile.log" >&2 || true
    cat "${temporary_dir}/runtime.stdout" >&2 || true
  fi
  rm -rf -- "${temporary_dir}"
  return "${result}"
}
trap cleanup EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
  -I "$(eshkol_source_dir)/inc"
)
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_G3C4_I2_CONSTRUCTION_PRIVATE -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${temporary_dir}/i2_native_helpers.o"
"${cc}" "${cflags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
  -o "${temporary_dir}/m3t_f32_integration.o"
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/g3c4/call_entry_native.c" \
  -o "${temporary_dir}/g3c4_call_entry_native.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${temporary_dir}/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell \
              n3k_primitives_provider n2_primitives_provider \
              a2_attention_provider a2_kv_cache; do
  "${cc}" "${cflags[@]}" -DET_A2_KV_CACHE_TESTING \
    -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
for source in m3_i64_integration m3_model; do
  "${cc}" "${cflags[@]}" \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
ar rcsD "${temporary_dir}/libg3c4_call_entry.a" \
  "${temporary_dir}"/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${temporary_dir}/cache" ESHKOL_CXX_COMPILER="${cxx}" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
    --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" \
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/native" \
    -L "${temporary_dir}" --lib g3c4_call_entry \
    "${test_source}" \
    -o "${temporary_dir}/call-entry" \
    >"${temporary_dir}/compile.log" 2>&1

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${temporary_dir}/failstop-cache" \
  ESHKOL_CXX_COMPILER="${cxx}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
    --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" \
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/native" \
    -L "${temporary_dir}" --lib g3c4_call_entry \
    "${PROJECT_ROOT}/tests/g3c4/call_entry_failstop_test.esk" \
    -o "${temporary_dir}/call-entry-failstop" \
    >"${temporary_dir}/failstop-compile.log" 2>&1

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "${temporary_dir}/call-entry" \
  >"${temporary_dir}/runtime.stdout"
grep -E '^G3-C4 call entry PASS: checks=[1-9][0-9]*$' \
  "${temporary_dir}/runtime.stdout" >/dev/null
for mode in abort finish; do
  status=0
  timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${temporary_dir}/call-entry-failstop" "${mode}" \
    >"${temporary_dir}/failstop-${mode}.stdout" \
    2>"${temporary_dir}/failstop-${mode}.stderr" || status=$?
  [[ "${status}" == 134 ]] || \
    die "${mode} cleanup defect returned ${status}, expected fail-stop 134"
done
cp "${temporary_dir}/compile.log" "${evidence}/compile.log"
cp "${temporary_dir}/runtime.stdout" "${evidence}/runtime.stdout"
cp "${temporary_dir}/failstop-compile.log" "${evidence}/failstop-compile.log"
cp "${temporary_dir}"/failstop-*.std{out,err} "${evidence}/"
cp "${PROJECT_ROOT}/native/g3c4_call_entry_source_closure.txt" \
  "${evidence}/source-closure.txt"
while IFS= read -r source; do
  printf '%s\t%s\n' "$(sha256sum "${PROJECT_ROOT}/${source}" | awk '{print $1}')" \
    "${source}"
done <"${PROJECT_ROOT}/native/g3c4_call_entry_source_closure.txt" \
  >"${evidence}/source-sha256.tsv"
for artifact in "${temporary_dir}"/*.o "${temporary_dir}"/*.a \
                "${temporary_dir}/call-entry" \
                "${temporary_dir}/call-entry-failstop"; do
  printf '%s\t%s\n' "$(sha256sum "${artifact}" | awk '{print $1}')" \
    "$(basename "${artifact}")"
done | LC_ALL=C sort -k2 >"${evidence}/object-sha256.tsv"
{
  source_commit="${G3C4_SOURCE_COMMIT:-$(git -C "${PROJECT_ROOT}" rev-parse HEAD)}"
  source_tree="${G3C4_SOURCE_TREE:-$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')}"
  [[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid source commit"
  [[ "${source_tree}" =~ ^[0-9a-f]{40}$ ]] || die "invalid source tree"
  printf 'commit\t%s\n' "${source_commit}"
  printf 'tree\t%s\n' "${source_tree}"
  printf 'container_image\t%s\n' "${G3C4_CONTAINER_IMAGE:-unreported}"
  printf 'cc\t%s\n' "$("${cc}" --version | head -n 1)"
  printf 'cxx\t%s\n' "$("${cxx}" --version | head -n 1)"
  printf 'eshkol\t%s\n' "$("${runner}" --version 2>&1)"
} >"${evidence}/environment.tsv"
sha256sum "${PROJECT_ROOT}/scripts/test-g3c4-call-entry.sh" \
  "${PROJECT_ROOT}/scripts/check-g3c4-call-entry.py" \
  >"${evidence}/runner-sha256.txt"
(cd "${evidence}" && sha256sum \
  compile.log environment.tsv failstop-abort.stderr failstop-abort.stdout \
  failstop-compile.log failstop-finish.stderr failstop-finish.stdout \
  object-sha256.tsv runner-sha256.txt runtime.stdout source-closure.txt \
  source-sha256.tsv >evidence-manifest.tsv)
sha256sum "${evidence}/evidence-manifest.tsv" \
  >"${evidence}/evidence-manifest.sha256"
cat "${evidence}/runtime.stdout"
printf 'G3-C4 call entry evidence: %s\n' "${evidence}"
