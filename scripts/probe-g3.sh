#!/usr/bin/env bash
# Development reachability only. Never builds prerequisites or installs a G3 API.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 2 ]] || die "usage: $0 [M3_ARTIFACT_DIR [EVIDENCE_DIR]]"
verify_toolchain
for command in ar cmp nm timeout; do require_command "${command}"; done
g3_package="$(realpath "${1:-$(project_build_dir)/m3}")"
g3_output="${2:-$(project_build_dir)/g3-probe}"
mkdir -p "${g3_output}"
g3_output="$(realpath "${g3_output}")"
g3_install="${g3_output}/installed"
g3_timeout="${G3_COMPILER_TIMEOUT_SECONDS:-120}"
[[ "${g3_timeout}" =~ ^[1-9][0-9]*$ ]] || die "G3 compiler timeout must be positive"
[[ -s "${g3_package}/libeshkol_transformer_m3.a" ]] || die "build M3 first: /usr/bin/bash scripts/build-m3.sh"
for pair in 'defined_symbols global-defined' 'public_exports package-exports' \
    'public_strings public-strings' 'undefined_symbols undefined' \
    'source_closure source-closure' 'native_source_closure native-source-closure' \
    'native_objects native-objects'; do
  read -r expected observed <<<"${pair}"
  cmp "${PROJECT_ROOT}/native/m3_package_${expected}.txt" \
    "${g3_package}/m3_package.o.evidence/${observed}.txt"
done
python3 "${PROJECT_ROOT}/tests/probes/g3/check_package.py" \
  "${PROJECT_ROOT}" "${g3_package}" "${G3_PACKAGE_SOURCE_DIR:-${PROJECT_ROOT}}" "${g3_output}"
mkdir -p "${g3_install}/facades/transformer"
while IFS= read -r facade; do
  cmp "${PROJECT_ROOT}/lib/${facade}" "${g3_package}/facades/${facade}"
  cp "${g3_package}/facades/${facade}" "${g3_install}/facades/${facade}"
done <"${PROJECT_ROOT}/native/m3_package_facades.txt"
cp "${g3_package}/libeshkol_transformer_m3.a" "${g3_install}/"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "${g3_output}/toolchain-provenance.tsv"
sha256sum "${g3_install}/libeshkol_transformer_m3.a" >"${g3_output}/package.sha256"
g3_compile() {
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
    -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${g3_output}/cache" ESHKOL_LIB_DIR="${g3_install}/facades" \
    ESHKOL_CXX_COMPILER="${ESHKOL_CXX_COMPILER}" \
    timeout --foreground --signal=TERM --kill-after=5s "${g3_timeout}s" \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --no-stdlib \
    -I "${g3_install}/facades" "$@"
}
g3_source="${PROJECT_ROOT}/tests/probes/g3/m3_reachability.esk"
g3_compile --compile-only --emit-depfile "${g3_output}/m3.d" \
  "${g3_source}" -o "${g3_output}/m3.o" >"${g3_output}/m3-compile.log" 2>&1
python3 "${PROJECT_ROOT}/tests/m3/check_public_closure.py" \
  "${g3_install}" "${g3_source}" "${g3_output}/m3.d"
if nm -u --format=posix "${g3_output}/m3.o" | \
    grep -E 'et_(m3|m3t|i2|f32|p1|kernel)_private|et_e1b_private_'; then
  die "G3 public probe retained private authority"
fi
g3_compile -L "${g3_install}" --lib eshkol_transformer_m3 \
  "${g3_source}" -o "${g3_output}/m3" >"${g3_output}/m3-link.log" 2>&1
for run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s "${g3_output}/m3" \
    >"${g3_output}/m3-${run}.stdout" 2>"${g3_output}/m3-${run}.stderr"
done
cmp "${g3_output}/m3-1.stdout" "${g3_output}/m3-2.stdout"
grep -Fx 'G3-M3-REACHABILITY-PASS' "${g3_output}/m3-1.stdout"
if g3_compile --compile-only \
    "${PROJECT_ROOT}/tests/probes/g3/production_generation_missing.esk" \
    -o "${g3_output}/generation.o" >"${g3_output}/generation-missing.log" 2>&1; then
  die "production generation facade unexpectedly available; revisit G3 proposal"
fi
grep -F "Module 'transformer.generation' not found" "${g3_output}/generation-missing.log" >/dev/null || \
  die "generation negative failed for unrelated reason"
# Deliberately separate declaration-only search path; no A0 stub is executed.
g3_compile -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/tests/fixtures/a0" \
  --compile-only "${PROJECT_ROOT}/tests/probes/g3/a0_declarations.esk" \
  -o "${g3_output}/a0-declarations.o" >"${g3_output}/a0-declarations.log" 2>&1
for entry in 'generator-create 3' 'generator-prefill! 2' \
    'generator-decode-step! 2' 'generator-generate! 2' \
    'generation-output-ids 1' 'generation-output-lengths 1' \
    'generation-output-text 1' 'generation-output-rng 1' \
    'generation-output-cache-lengths 1'; do
  read -r operation arity <<<"${entry}"
  printf '(require transformer.generation)\n(%s)\n' "${operation}" \
    >"${g3_output}/arity.esk"
  if g3_compile -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/tests/fixtures/a0" \
      --compile-only "${g3_output}/arity.esk" -o "${g3_output}/arity.o" \
      >"${g3_output}/arity-${operation}.log" 2>&1; then
    die "A0 generation arity unexpectedly accepted: ${operation}"
  fi
  grep -F "Arity mismatch: ${operation} expects ${arity} arguments but got 0" \
    "${g3_output}/arity-${operation}.log" >/dev/null || \
    die "A0 generation arity fixture failed for unrelated reason: ${operation}"
done
printf 'G3 declaration-only arities compiled; production generation facade absent\n'
printf 'G3 probe evidence: %s\n' "${g3_output}"
