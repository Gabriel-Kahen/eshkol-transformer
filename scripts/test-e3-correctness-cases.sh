#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 realpath strings timeout; do require_command "${command}"; done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-correctness-cases}")"
[[ ! -e "${out}" ]] || die "E3 correctness output directory must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || die "E3 correctness requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || die "E3 correctness oracle versions are not pinned"
for variant in a b; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_reference.correctness --output "${out}/corpus-${variant}" \
    >"${out}/corpus-${variant}.stdout" 2>"${out}/corpus-${variant}.stderr"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_reference.correctness --corpus "${out}/corpus-${variant}" \
    --reference "${out}/reference-${variant}.json" --proof "${out}/selection-${variant}.json" \
    >"${out}/reference-${variant}.stdout" 2>"${out}/reference-${variant}.stderr"
done
diff -r "${out}/corpus-a" "${out}/corpus-b"
cmp "${out}/reference-a.json" "${out}/reference-b.json"
cmp "${out}/selection-a.json" "${out}/selection-b.json"

prefix="${PROJECT_ROOT}/tests/e3_private/numerical_package"
for variant in a b sanitized wrong-vocab-axis; do
  mode=normal
  [[ "${variant}" == sanitized ]] && mode=sanitize
  [[ "${variant}" == wrong-vocab-axis ]] && mode=wrong-vocab-axis
  "${PROJECT_ROOT}/scripts/build-e3-numerical-test.sh" "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" 2>"${out}/package-${variant}.stderr"
  artifact="${out}/package-${variant}/e3_numerical_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    cmp "${prefix}_${pair%%:*}.txt" "${artifact}/${pair#*:}.txt"
  done
  if [[ "${variant}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${artifact}/undefined.txt"
  fi
  grep -F 'E3ROW' "${artifact}/strings.txt" >/dev/null || die "${variant}: numerical observer absent"
done
cmp "${out}/package-a/e3_numerical_package.o" "${out}/package-b/e3_numerical_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_numerical.a" "${out}/package-b/libeshkol_transformer_e3_numerical.a"
cmp "${out}/package-a/e3_numerical_package.o.evidence/private.d" \
    "${out}/package-b/e3_numerical_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_numerical_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_numerical_package.o.evidence/native-depfiles"
generated_source="${out}/.e3-numerical-generated/source/e3_d2_dataset.esk"
for variant in a b; do
  object="${out}/package-${variant}/e3_numerical_package.o"
  strings -a "${object}" | grep -Fx "${generated_source}" >/dev/null || \
    die "${variant}: stable generated source path absent"
  if strings -a "${object}" | grep -F '/eshkol-transformer-e1b.' >/dev/null; then
    die "${variant}: random generated source path retained"
  fi
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "E3 compiler timeout must be positive"
for variant in a b sanitized wrong-vocab-axis; do
  cxx_selected="${cxx}"
  [[ "${variant}" == sanitized ]] && cxx_selected="${out}/cxx-sanitized"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx_selected}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_numerical \
      "${PROJECT_ROOT}/tests/e3_private/correctness_runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" 2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

for case_name in all-correct all-wrong; do
  for variant in a b sanitized poison; do
    runtime="${variant}" poison=0
    [[ "${variant}" == poison ]] && runtime=a && poison=1
    if [[ "${variant}" == sanitized ]]; then
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
        ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/runtime-${runtime}" "${case_name}" "${out}/corpus-a/${case_name}" \
        >"${out}/run-${variant}-${case_name}.stdout" 2>"${out}/run-${variant}-${case_name}.stderr"
    else
      ESHKOL_ARENA_POISON="${poison}" timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/runtime-${runtime}" "${case_name}" "${out}/corpus-a/${case_name}" \
        >"${out}/run-${variant}-${case_name}.stdout" 2>"${out}/run-${variant}-${case_name}.stderr"
    fi
    stdout="${out}/run-${variant}-${case_name}.stdout"
    [[ ! -s "${out}/run-${variant}-${case_name}.stderr" ]] || die "${variant}-${case_name}: runtime diagnostics"
    python3 -B -m tests.e3_private.parse_numerical_observation \
      --stdout "${stdout}" --batch-count 3 \
      --observed "${out}/observed-${variant}-${case_name}.json" \
      --roles "${out}/roles-${variant}-${case_name}.tsv"
    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
      -m tests.e3_native_parity.verify_observation \
      --reference "${out}/reference-a.json" --case "${case_name}" \
      --observed "${out}/observed-${variant}-${case_name}.json" \
      --roles "${out}/roles-${variant}-${case_name}.tsv" \
      --transcript "${out}/transcript-${variant}-${case_name}.json" \
      >"${out}/verify-${variant}-${case_name}.stdout" \
      2>"${out}/verify-${variant}-${case_name}.stderr"
    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
      -m tests.e3_reference.correctness --corpus "${out}/corpus-a" \
      --case "${case_name}" --observed "${out}/observed-${variant}-${case_name}.json" \
      >"${out}/class-${variant}-${case_name}.stdout" \
      2>"${out}/class-${variant}-${case_name}.stderr"
  done
  for variant in b poison; do
    cmp "${out}/run-a-${case_name}.stdout" "${out}/run-${variant}-${case_name}.stdout"
    cmp "${out}/transcript-a-${case_name}.json" "${out}/transcript-${variant}-${case_name}.json"
  done
done
ESHKOL_ARENA_POISON=0 timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${out}/runtime-wrong-vocab-axis" all-correct "${out}/corpus-a/all-correct" \
  >"${out}/run-wrong-vocab-axis-all-correct.stdout" \
  2>"${out}/run-wrong-vocab-axis-all-correct.stderr"
[[ ! -s "${out}/run-wrong-vocab-axis-all-correct.stderr" ]] || \
  die "wrong-vocab-axis: runtime diagnostics"
python3 -B -m tests.e3_private.parse_numerical_observation \
  --stdout "${out}/run-wrong-vocab-axis-all-correct.stdout" --batch-count 3 \
  --observed "${out}/observed-wrong-vocab-axis-all-correct.json" \
  --roles "${out}/roles-wrong-vocab-axis-all-correct.tsv"
cmp "${out}/roles-a-all-correct.tsv" \
    "${out}/roles-wrong-vocab-axis-all-correct.tsv"
python3 -B - "${out}/observed-a-all-correct.json" \
  "${out}/observed-wrong-vocab-axis-all-correct.json" <<'PY'
import json
from pathlib import Path
import sys

normal, mutant = (json.loads(Path(path).read_bytes()) for path in sys.argv[1:])
def require(condition, label):
    if not condition:
        raise SystemExit(f"wrong-vocab-axis {label} differs")

require([batch["inputs"] for batch in normal["batches"]] ==
        [[3, 14], [31, 14], [31, 0]], "normal D2 inputs")
require([batch["mask"] for batch in normal["batches"]] ==
        [[True, True], [True, True], [True, False]], "normal D2 masks")
require(len(mutant["batches"]) == 3, "batch count")
for baseline, changed in zip(normal["batches"], mutant["batches"]):
    for key in ("ordinal", "inputs", "targets", "mask", "logits_f32", "ce_f32",
                "reduction_f32", "counters"):
        require(baseline[key] == changed[key], key)
    require(baseline["state_f32"][:2] == changed["state_f32"][:2], "N/W state")
    require(changed["correct_f32"] == "00000000", "batch correct")
    require(changed["state_f32"][2] == "00000000", "C state")
require(normal["final"]["counters"] == mutant["final"]["counters"] == [5, 3],
        "final counters")
require(normal["final"]["metrics_f32"][:3] == mutant["final"]["metrics_f32"][:3],
        "final loss/weight/perplexity")
require(normal["final"]["metrics_f32"][3] == "3f800000", "normal accuracy")
require(mutant["final"]["metrics_f32"][3] == "00000000", "mutant accuracy")
PY
set +e
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  timeout --foreground --signal=TERM --kill-after=5s 120s "${q0_python}" -B \
  "${PROJECT_ROOT}/tests/e3_private/wrong_vocab_axis_rejection.py" run \
  --reference "${out}/reference-a.json" \
  --observed "${out}/observed-wrong-vocab-axis-all-correct.json" \
  --roles "${out}/roles-wrong-vocab-axis-all-correct.tsv" \
  --transcript "${out}/transcript-wrong-vocab-axis-all-correct.json" \
  --traceback "${out}/verify-wrong-vocab-axis-all-correct.traceback" \
  >"${out}/verify-wrong-vocab-axis-all-correct.stdout" \
  2>"${out}/verify-wrong-vocab-axis-all-correct.stderr"
verify_status=$?
set -e
printf '%d\n' "${verify_status}" >"${out}/verify-wrong-vocab-axis-all-correct.exit"
"${q0_python}" -B \
  "${PROJECT_ROOT}/tests/e3_private/wrong_vocab_axis_rejection.py" check \
  --status "${out}/verify-wrong-vocab-axis-all-correct.exit" \
  --stdout "${out}/verify-wrong-vocab-axis-all-correct.stdout" \
  --stderr "${out}/verify-wrong-vocab-axis-all-correct.stderr" \
  --traceback "${out}/verify-wrong-vocab-axis-all-correct.traceback" \
  --transcript "${out}/transcript-wrong-vocab-axis-all-correct.json"
printf 'E3 compiled all-correct/all-wrong transcripts and wrong-vocab-axis mutant passed\n'
