#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env nm python3 timeout; do require_command "${command}"; done
[[ $# -ge 1 && $# -le 2 ]] || die "usage: $0 EVIDENCE_DIR [BUILT_ARTIFACT]"
g3g_evidence="$1"
g3g_artifact="${2:-${g3g_evidence}/installed}"
mkdir -p "${g3g_evidence}"
if [[ $# == 1 ]]; then
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3g-g0.sh" "${g3g_artifact}"
fi
python3 "${PROJECT_ROOT}/scripts/check-g3g-g0-package.py" >"${g3g_evidence}/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"${g3g_evidence}/q0.stdout" 2>"${g3g_evidence}/q0.stderr"
g3g_prefix="${PROJECT_ROOT}/native/g3g_g0_package"
if /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${g3g_prefix}_root.esk" \
    "${PROJECT_ROOT}/native/m3_package_bridge.c" \
    "${g3g_prefix}_private_renames.txt" \
    "${g3g_prefix}_public_exports.txt" \
    "${g3g_evidence}/mixed-package.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/src" \
    >"${g3g_evidence}/mixed-package.stdout" \
    2>"${g3g_evidence}/mixed-package.stderr"; then
  die "G3-G G0 accepted a mixed private/public package tuple"
fi
grep -F 'G3-G G0 requires the exact repository tuple' \
  "${g3g_evidence}/mixed-package.stderr" >/dev/null
for g3g_pair in 'defined_symbols global-defined' 'public_exports package-exports' \
    'public_strings public-strings' 'undefined_symbols undefined' \
    'source_closure source-closure' 'native_source_closure native-source-closure' \
    'native_objects native-objects'; do
  read -r g3g_expected g3g_observed <<<"${g3g_pair}"
  cmp "${g3g_prefix}_${g3g_expected}.txt" \
    "${g3g_artifact}/g3g_g0_package.o.evidence/${g3g_observed}.txt"
done
(cd "${g3g_artifact}/facades" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
  >"${g3g_evidence}/facades.txt"
cmp "${g3g_prefix}_facades.txt" "${g3g_evidence}/facades.txt"
ar t "${g3g_artifact}/libeshkol_transformer_g3g_g0.a" \
  >"${g3g_evidence}/archive-members.txt"
cmp "${g3g_prefix}_archive_members.txt" "${g3g_evidence}/archive-members.txt"
if nm -g --defined-only "${g3g_artifact}/g3g_g0_package.o" | \
    grep -E 'et_g3t_test_|et_e1b_private_|g3m-generate'; then
  die "G3-G G0 production package exported a private or test seam"
fi
g3g_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
g3g_cc="$(tsv_value "${g3g_provenance}" cc_path)"
g3g_cxx="$(tsv_value "${g3g_provenance}" cxx_path)"
g3g_runner="$(eshkol_build_dir)/eshkol-run"
"${g3g_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/g3t_transport.c" \
  -o "${g3g_evidence}/g3t-feature-off.o"
if nm --defined-only "${g3g_evidence}/g3t-feature-off.o" | \
    grep -E 'et_g3t_(test_|private_(sample|zero|p2_zero|prompt_preflight|input_from_pair|full_request_preflight|input_from_t1|output_(ids|lengths|cache_lengths|rng)_clone))'; then
  die "G3-G G0 feature-off transport exposed a gated seam"
fi
g3g_compile() {
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${g3g_evidence}/cache" \
    ESHKOL_LIB_DIR="${g3g_artifact}/facades" \
    ESHKOL_CXX_COMPILER="${g3g_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${g3g_runner}" --strict-types --no-stdlib \
    -I "${g3g_artifact}/facades" "$@"
}
for g3g_source in compile_api public_runtime; do
  g3g_compile --compile-only --emit-depfile "${g3g_evidence}/${g3g_source}.d" \
    "${PROJECT_ROOT}/tests/g3g_g0/${g3g_source}.esk" \
    -o "${g3g_evidence}/${g3g_source}.o" \
    >"${g3g_evidence}/${g3g_source}.compile-only.stdout" \
    2>"${g3g_evidence}/${g3g_source}.compile-only.stderr"
  python3 "${PROJECT_ROOT}/tests/g3g_g0/check_public_closure.py" \
    "${g3g_artifact}" "${PROJECT_ROOT}/tests/g3g_g0/${g3g_source}.esk" \
    "${g3g_evidence}/${g3g_source}.d" >>"${g3g_evidence}/static.stdout"
done
if nm -u --format=posix "${g3g_evidence}/compile_api.o" | \
    grep -E 'et_e1b_private_|et_g3t_private_|et_g3t_test_'; then
  die "G3-G G0 public caller retained private authority"
fi
nm -u --format=posix "${g3g_evidence}/compile_api.o" | \
  awk '$1 ~ /^et_e1b_public_g3_/ {print $1}' | LC_ALL=C sort -u \
  >"${g3g_evidence}/api-g3-undefined.txt"
grep '^et_e1b_public_g3_' "${g3g_prefix}_public_exports.txt" \
  >"${g3g_evidence}/expected-api-g3-undefined.txt"
cmp "${g3g_evidence}/expected-api-g3-undefined.txt" \
  "${g3g_evidence}/api-g3-undefined.txt"
g3g_compile -O 0 -L "${g3g_artifact}" --lib eshkol_transformer_g3g_g0 \
  "${PROJECT_ROOT}/tests/g3g_g0/public_runtime.esk" \
  -o "${g3g_evidence}/public-runtime" \
  >"${g3g_evidence}/public.compile.stdout" \
  2>"${g3g_evidence}/public.compile.stderr"
for g3g_mode in normal repeat; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${g3g_evidence}/public-runtime" \
    >"${g3g_evidence}/${g3g_mode}.stdout" \
    2>"${g3g_evidence}/${g3g_mode}.stderr"
  test ! -s "${g3g_evidence}/${g3g_mode}.stderr"
  grep -Fx 'G3G-PUBLIC-G0G1-PASS' "${g3g_evidence}/${g3g_mode}.stdout" >/dev/null
done
cmp "${g3g_evidence}/normal.stdout" "${g3g_evidence}/repeat.stdout"
git -C "${PROJECT_ROOT}" diff --check
cat "${g3g_evidence}/static.stdout"
cat "${g3g_evidence}/normal.stdout"
