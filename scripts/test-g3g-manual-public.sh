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
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-g3g-manual.sh" "${g3g_artifact}"
fi
python3 "${PROJECT_ROOT}/scripts/check-g3g-manual-package.py" >"${g3g_evidence}/static.stdout"
if git -C "${PROJECT_ROOT}" ls-files --error-unmatch \
    tests/q0/test_python_isolation.py >/dev/null 2>&1; then
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
    >"${g3g_evidence}/q0.stdout" 2>"${g3g_evidence}/q0.stderr"
else
  printf 'Docker worktree git metadata inaccessible; host Q0 and diff check required\n' \
    >"${g3g_evidence}/git-metadata.stdout"
fi
g3g_prefix="${PROJECT_ROOT}/native/g3g_manual_package"
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
  die "G3-G manual accepted a mixed private/public package tuple"
fi
grep -F 'G3-G manual requires the exact repository tuple' \
  "${g3g_evidence}/mixed-package.stderr" >/dev/null
for g3g_pair in 'defined_symbols global-defined' 'public_exports package-exports' \
    'public_strings public-strings' 'undefined_symbols undefined' \
    'source_closure source-closure' 'native_source_closure native-source-closure' \
    'native_objects native-objects'; do
  read -r g3g_expected g3g_observed <<<"${g3g_pair}"
  cmp "${g3g_prefix}_${g3g_expected}.txt" \
    "${g3g_artifact}/g3g_manual_package.o.evidence/${g3g_observed}.txt"
done
(cd "${g3g_artifact}/facades" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
  >"${g3g_evidence}/facades.txt"
cmp "${g3g_prefix}_facades.txt" "${g3g_evidence}/facades.txt"
ar t "${g3g_artifact}/libeshkol_transformer_g3g_manual.a" \
  >"${g3g_evidence}/archive-members.txt"
cmp "${g3g_prefix}_archive_members.txt" "${g3g_evidence}/archive-members.txt"
if nm -g --defined-only "${g3g_artifact}/g3g_manual_package.o" | \
    grep -E 'et_g3t_test_|et_e1b_private_|g3m-generate|et_t1_i64_shell_private_c4_read_v1|et_i64_tensor_private_t1_pair_validate_v1|et_g3c4_private_input_from_t1_p2_v1'; then
  die "G3-G manual production package exported a private or test seam"
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
    grep -E 'et_g3t_(test_|private_(sample|zero|p2_zero|prompt_preflight|input_from_pair|full_request_preflight|input_from_t1|output_(ids|lengths|cache_lengths|rng)_clone|logits_copy_bits|logits_reserve|manual_))'; then
  die "G3-G manual feature-off transport exposed a gated seam"
fi
"${g3g_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer" \
  -I "$(eshkol_source_dir)/inc" \
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE \
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE \
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${g3g_evidence}/g3c4-feature-off.o"
if nm -u --format=posix "${g3g_evidence}/g3c4-feature-off.o" | \
    grep -F 'et_t1_i64_shell_private_c4_read_v1'; then
  die "C4 owner feature-off path requires exact-pair authority"
fi
if nm -g --defined-only "${g3g_evidence}/g3c4-feature-off.o" | \
    grep -E 'et_g3c4_private_(input_from_t1_p2|p2g1_(preflight|run)|result_(tensor|rng)_create|output_device_state)_v1'; then
  die "C4 owner feature-off path exposed a gated public P2/G1 dependency"
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
    "${PROJECT_ROOT}/tests/g3g_manual/${g3g_source}.esk" \
    -o "${g3g_evidence}/${g3g_source}.o" \
    >"${g3g_evidence}/${g3g_source}.compile-only.stdout" \
    2>"${g3g_evidence}/${g3g_source}.compile-only.stderr"
  python3 "${PROJECT_ROOT}/tests/g3g_manual/check_public_closure.py" \
    "${g3g_artifact}" "${PROJECT_ROOT}/tests/g3g_manual/${g3g_source}.esk" \
    "${g3g_evidence}/${g3g_source}.d" >>"${g3g_evidence}/static.stdout"
done
if nm -u --format=posix "${g3g_evidence}/compile_api.o" | \
    grep -E 'et_e1b_private_|et_g3t_private_|et_g3t_test_'; then
  die "G3-G manual public caller retained private authority"
fi
nm -u --format=posix "${g3g_evidence}/compile_api.o" | \
  awk '$1 ~ /^et_e1b_public_g3(_|c4)/ {print $1}' | LC_ALL=C sort -u \
  >"${g3g_evidence}/api-g3-undefined.txt"
grep -E '^et_e1b_public_g3(_|c4)' "${g3g_prefix}_public_exports.txt" \
  >"${g3g_evidence}/expected-api-g3-undefined.txt"
cmp "${g3g_evidence}/expected-api-g3-undefined.txt" \
  "${g3g_evidence}/api-g3-undefined.txt"
g3g_compile -O 0 -L "${g3g_artifact}" --lib eshkol_transformer_g3g_manual \
  "${PROJECT_ROOT}/tests/g3g_manual/public_runtime.esk" \
  -o "${g3g_evidence}/public-runtime" \
  >"${g3g_evidence}/public.compile.stdout" \
  2>"${g3g_evidence}/public.compile.stderr"
for g3g_mode in normal repeat; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${g3g_evidence}/public-runtime" \
    >"${g3g_evidence}/${g3g_mode}.stdout" \
    2>"${g3g_evidence}/${g3g_mode}.stderr"
  test ! -s "${g3g_evidence}/${g3g_mode}.stderr"
  grep -Fx 'G3G-MANUAL-PUBLIC-PASS' "${g3g_evidence}/${g3g_mode}.stdout" >/dev/null
done
cmp "${g3g_evidence}/normal.stdout" "${g3g_evidence}/repeat.stdout"
if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree \
    >/dev/null 2>&1; then
  git -C "${PROJECT_ROOT}" diff --check
fi
cat "${g3g_evidence}/static.stdout"
cat "${g3g_evidence}/normal.stdout"
