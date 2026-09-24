#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp env grep python3 timeout; do require_command "${command}"; done

e3_artifact="${1:-$(project_build_dir)/e3-private}"
e3_output="${2:-$(project_build_dir)/e3-private-runtime}"
[[ -s "${e3_artifact}/libeshkol_transformer_e3_private.a" ]] || \
  die "E3 private archive prerequisite is missing"
mkdir -p "${e3_output}"

e3_corpus="${e3_output}/corpus"
if [[ ! -s "${e3_corpus}/corpus-reference.json" ]]; then
  python3 -m tests.e3_reference.corpus --output "${e3_corpus}"
fi
e3_reference="${E3_REFERENCE_BUNDLE:-}"
[[ "${e3_reference}" == /* && -f "${e3_reference}" ]] || \
  die "E3_REFERENCE_BUNDLE must name the accepted absolute reference bundle"
python3 - "${e3_corpus}/corpus-reference.json" "${e3_reference}" <<'PY'
from pathlib import Path
import sys
from tests.e3_reference.transcript import decode

corpus = decode(Path(sys.argv[1]).read_bytes())
reference = decode(Path(sys.argv[2]).read_bytes())
expected = {case["name"]: case for case in corpus["cases"]}
observed = {case["name"]: case["corpus"] for case in reference["cases"]}
if expected != observed:
    raise SystemExit("accepted E3 reference does not bind the generated corpus")
expected_final = {
    "packed-single": (["40b16271", "40a00000", "437f82e7", "00000000"], [5, 3]),
    "packed-sharded": (["40b16271", "40a00000", "437f82e7", "00000000"], [5, 3]),
    "packed-suffix": (["40b12fa5", "40400000", "437dee8f", "00000000"], [3, 2]),
    "unpacked-three": (["40b19c48", "40800000", "4380a930", "00000000"], [4, 2]),
    "unpacked-pairs": (["40b1892d", "40400000", "43805c75", "00000000"], [3, 3]),
}
actual_final = {
    case["name"]: (case["synthetic"]["final"]["metrics_f32"],
                   case["synthetic"]["final"]["counters"])
    for case in reference["cases"]
}
if expected_final != actual_final:
    raise SystemExit("private runtime expectations differ from the accepted E3 reference")
PY

e3_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
e3_cxx="$(tsv_value "${e3_provenance}" cxx_path)"
e3_runner="$(eshkol_build_dir)/eshkol-run"
e3_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${e3_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E3 compiler timeout must be a positive integer"

env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
  -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
  -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${e3_output}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${e3_cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s "${e3_timeout}s" \
  "${e3_runner}" --strict-types --no-stdlib -O 2 \
    -L "${e3_artifact}" --lib eshkol_transformer_e3_private \
    "${PROJECT_ROOT}/tests/e3_private/runtime.esk" \
    -o "${e3_output}/runtime" \
    >"${e3_output}/compile.stdout" 2>"${e3_output}/compile.stderr"

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 600s \
  "${e3_output}/runtime" \
  "${e3_corpus}/packed-single" "${e3_corpus}/packed-sharded" \
  "${e3_corpus}/packed-suffix" "${e3_corpus}/unpacked-three" \
  "${e3_corpus}/unpacked-pairs" \
  >"${e3_output}/runtime.stdout" 2>"${e3_output}/runtime.stderr"
grep -Fx "E3-PRIVATE-RUNTIME-PASS" "${e3_output}/runtime.stdout" >/dev/null
printf 'E3 private package runtime passed: %s\n' "${e3_output}/runtime"
