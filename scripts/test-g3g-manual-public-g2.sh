#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env nm python3 sha256sum timeout; do require_command "${command}"; done
[[ $# -ge 1 && $# -le 2 ]] || die "usage: $0 EVIDENCE_DIR [normal|sanitize|all]"
evidence="$1"
selection="${2:-all}"
case "$selection" in
  normal) modes=(normal) ;;
  sanitize) modes=(sanitize) ;;
  all) modes=(normal sanitize) ;;
  *) die "G2 selection must be normal, sanitize or all" ;;
esac
[[ ! -e "${evidence}" && ! -L "${evidence}" ]] ||
  die "G2 evidence directory already exists"
mkdir -p "${evidence}"
for mode in "${modes[@]}"; do mkdir "${evidence}/${mode}"; done
source_pin="${evidence}/source.before.sha256"
source_after="${evidence}/source.after.sha256"
source_inputs=(scripts/common.sh scripts/build-e1b-consumer.sh
  scripts/build-g3g-manual.sh scripts/test-g3g-manual-public.sh
  scripts/test-g3g-manual-public-g2.sh
  tests/g3g_manual/public_g2_runtime.esk
  tests/g3g_manual/check_public_closure.py
  native/g3g_manual_package_root.esk native/g3g_manual_package_source_closure.txt
  native/g3g_manual_package_native_source_closure.txt
  native/g3c4_public_g2_port_source_closure.txt
  scripts/g3g-manual-native-inputs.sh
  tests/e3_private/asan_ir_sources.sha256
  tests/shared/prepare_eshkol_asan_ir.py tests/e3_private/check_asan_ir.py)
for name in "${source_inputs[@]}"; do
  [[ -f "${PROJECT_ROOT}/${name}" && ! -L "${PROJECT_ROOT}/${name}" ]] ||
    die "G2 selected source is not ordinary: ${name}"
done
sha256sum "${source_inputs[@]/#/${PROJECT_ROOT}/}" >"${source_pin}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
printf 'G3G-MANUAL-PUBLIC-G2-PASS\n' >"${evidence}/expected.stdout"
for mode in "${modes[@]}"; do
  directory="${evidence}/${mode}"
  if [[ "$mode" == normal ]]; then
    G3G_MANUAL_SANITIZE=0 /usr/bin/bash \
      "${PROJECT_ROOT}/scripts/build-g3g-manual.sh" "${directory}/installed" \
      >"${directory}/build.stdout" 2>"${directory}/build.stderr"
  else
    G3G_MANUAL_SANITIZE=1 /usr/bin/bash \
      "${PROJECT_ROOT}/scripts/build-g3g-manual.sh" "${directory}/installed" \
      >"${directory}/build.stdout" 2>"${directory}/build.stderr"
    for measured in private.asan.ll private.asan.json private.asan-imports.txt \
        noninstrumentation-undefined.txt sanitizer-undefined.txt; do
      test -s "${directory}/installed/g3g_manual_package.o.evidence/${measured}"
    done
    cmp "${PROJECT_ROOT}/native/g3g_manual_package_undefined_symbols.txt" \
        "${directory}/installed/g3g_manual_package.o.evidence/noninstrumentation-undefined.txt"
  fi
  ar t "${directory}/installed/libeshkol_transformer_g3g_manual.a" \
    >"${directory}/archive-members.txt"
  cmp "${PROJECT_ROOT}/native/g3g_manual_package_archive_members.txt" \
      "${directory}/archive-members.txt"
  compiler="${cxx}"
  if [[ "${mode}" == sanitize ]]; then
    cat >"${directory}/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
    chmod 0500 "${directory}/cxx-wrap"
    compiler="${directory}/cxx-wrap"
  fi
  mkdir -p "${directory}/cache"
  g2_compile() {
    env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
      -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
      -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
      -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
      -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${directory}/cache" \
      ESHKOL_LIB_DIR="${directory}/installed/facades" \
      ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${runner}" --strict-types --no-stdlib \
      -I "${directory}/installed/facades" "$@"
  }
  g2_compile --compile-only --emit-depfile "${directory}/public_g2.d" \
    "${PROJECT_ROOT}/tests/g3g_manual/public_g2_runtime.esk" \
    -o "${directory}/public_g2.o" \
    >"${directory}/compile-only.stdout" 2>"${directory}/compile-only.stderr"
  python3 "${PROJECT_ROOT}/tests/g3g_manual/check_public_closure.py" \
    "${directory}/installed" \
    "${PROJECT_ROOT}/tests/g3g_manual/public_g2_runtime.esk" \
    "${directory}/public_g2.d" >"${directory}/source-closure.stdout"
  g2_compile -O 0 -L "${directory}/installed" \
    --lib eshkol_transformer_g3g_manual \
    "${PROJECT_ROOT}/tests/g3g_manual/public_g2_runtime.esk" \
    -o "${directory}/public-g2-runtime" \
    >"${directory}/compile.stdout" 2>"${directory}/compile.stderr"
  runtime=(env ESHKOL_ARENA_POISON=1)
  if [[ "${mode}" == sanitize ]]; then
    runtime=(env ESHKOL_ARENA_POISON=1 \
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  for attempt in first repeat; do
    "${runtime[@]}" timeout --foreground --signal=TERM --kill-after=5s 180s \
      "${directory}/public-g2-runtime" \
      >"${directory}/${attempt}.stdout" 2>"${directory}/${attempt}.stderr"
    cmp "${evidence}/expected.stdout" "${directory}/${attempt}.stdout"
    test ! -s "${directory}/${attempt}.stderr" ||
      die "G2 ${mode}/${attempt} emitted stderr"
  done
done
if [[ "$selection" == all ]]; then
  cmp "${evidence}/normal/installed/g3g_manual_package.o.evidence/global-defined.txt" \
      "${evidence}/sanitize/installed/g3g_manual_package.o.evidence/global-defined.txt"
  cmp "${evidence}/normal/first.stdout" "${evidence}/sanitize/first.stdout"
fi
if [[ "$selection" == all ]]; then
  /usr/bin/bash "${PROJECT_ROOT}/scripts/test-g3g-manual-public.sh" \
    "${evidence}/p2g1-regression" "${evidence}/normal/installed" \
    >"${evidence}/p2g1.stdout" 2>"${evidence}/p2g1.stderr"
else
  printf 'NOT RUN: selected-mode G2 smoke; installed P2/G1/C2 regression is a separate gate\n' \
    >"${evidence}/p2g1-regression.status"
fi
sha256sum "${source_inputs[@]/#/${PROJECT_ROOT}/}" >"${source_after}"
cmp "${source_pin}" "${source_after}"
printf 'G3G-MANUAL-PUBLIC-G2-GATE-PASS mode=%s\n' "$selection"
