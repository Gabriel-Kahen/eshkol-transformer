#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp python3 realpath sha256sum timeout; do require_command "${command}"; done
[[ $# == 1 ]] || die "usage: $0 DISK_BACKED_EVIDENCE_DIR"
evidence="$(realpath -m -- "$1")"
mkdir -p -- "${evidence}"
scratch="$(mktemp -d "${evidence}/.build.XXXXXX")"
printf '%s\n' "${scratch}" >"${evidence}/scratch-path.txt"
prefix="${PROJECT_ROOT}/tests/cli3_generate/aggregate"
TMPDIR="${evidence}" E1B_COMPILER_TIMEOUT_SECONDS=1200 \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${prefix}_root.esk" "${prefix}_bridge.c" \
    "${prefix}_private_renames.txt" "${prefix}_public_exports.txt" \
    "${scratch}/aggregate.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t2/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/internal/d2/lib" \
    "${PROJECT_ROOT}/internal/e3/lib" \
    "${PROJECT_ROOT}/src"
ar rcsD "${scratch}/libcli3_generate_prerequisite.a" "${scratch}/aggregate.o"
runner="$(eshkol_build_dir)/eshkol-run"
cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
compile() {
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${scratch}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${runner}" --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -L "${scratch}" --lib cli3_generate_prerequisite "$@"
}
compile "${PROJECT_ROOT}/tests/cli3_generate/cli.esk" \
  -o "${scratch}/eshkol-transformer"
compile "${PROJECT_ROOT}/tests/cli3_generate/public_runtime.esk" \
  -o "${scratch}/public-runtime"
cat >"${scratch}/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
(root / 'train.txt').write_bytes(bytes((3, 197, 41, 7, 11, 19)))
PY
"${scratch}/eshkol-transformer" tokenizer byte \
  --config "${scratch}/config.json" --output "${scratch}/byte.t1" \
  >"${evidence}/tokenizer.json"
mkdir "${scratch}/train"
"${scratch}/eshkol-transformer" corpus build \
  --tokenizer "${scratch}/byte.t1" --document "${scratch}/train.txt" \
  --output-directory "${scratch}/train" --shard-token-limit 4 \
  >"${evidence}/corpus.json"
"${scratch}/eshkol-transformer" pretrain \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2" \
  --max-updates 1 >"${evidence}/pretrain.json"
cp "${scratch}/first.c2" "${scratch}/before.c2"
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
source = (root / 'first.c2').read_bytes()
corrupt = bytearray(source)
corrupt[-1] ^= 1
(root / 'corrupt.c2').write_bytes(corrupt)
(root / 'torn.c2').write_bytes(source[:17])
PY
for run in normal repeat; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${scratch}/public-runtime" \
    "${scratch}/byte.t1" "${scratch}/train" "${scratch}/first.c2" \
    "${scratch}/corrupt.c2" "${scratch}/torn.c2" \
    >"${evidence}/${run}.stdout" 2>"${evidence}/${run}.stderr"
  [[ ! -s "${evidence}/${run}.stderr" ]] || die "${run} public runtime wrote stderr"
  grep -E '^CLI3-GENERATE-CHECKPOINT-PREREQUISITE-PASS [0-9]+ [0-9]+$' \
    "${evidence}/${run}.stdout" >/dev/null
done
cmp "${evidence}/normal.stdout" "${evidence}/repeat.stdout"
cmp "${scratch}/before.c2" "${scratch}/first.c2"
generate_base=("${scratch}/eshkol-transformer" generate
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1"
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2"
  --prompt-hex 41 --seed 1729)
for mode in greedy categorical; do
  for repeat in first second; do
    "${generate_base[@]}" --sampling "${mode}" \
      >"${evidence}/generate-${mode}-${repeat}.json" \
      2>"${evidence}/generate-${mode}-${repeat}.stderr"
    [[ ! -s "${evidence}/generate-${mode}-${repeat}.stderr" ]]
  done
  cmp "${evidence}/generate-${mode}-first.json" \
      "${evidence}/generate-${mode}-second.json"
done
python3 - "${evidence}" <<'PY'
from pathlib import Path
import json
import sys
root = Path(sys.argv[1])
parts = (root / 'normal.stdout').read_text().split()
assert parts[0] == 'CLI3-GENERATE-CHECKPOINT-PREREQUISITE-PASS'
for mode, expected in [('greedy', int(parts[1])),
                       ('categorical', int(parts[2]))]:
    record = json.loads((root / f'generate-{mode}-first.json').read_text())
    assert record == {'artifact': 'generation', 'generated_hex': f'{expected:02x}',
                      'prompt_hex': '41', 'sampling': mode, 'seed': 1729}
PY
run_negative() {
  local name=$1 expected=$2
  shift 2
  set +e
  "$@" >"${evidence}/${name}.stdout" 2>"${evidence}/${name}.stderr"
  local status=$?
  set -e
  [[ "${status}" == "${expected}" ]] || \
    die "${name}: expected status ${expected}, got ${status}"
  [[ ! -s "${evidence}/${name}.stdout" ]]
  [[ -s "${evidence}/${name}.stderr" ]]
  if [[ "${expected}" == 2 ]]; then
    grep -F 'eshkol-transformer: usage message=' \
      "${evidence}/${name}.stderr" >/dev/null
  else
    grep -F 'category="corrupt-data"' \
      "${evidence}/${name}.stderr" >/dev/null
  fi
}
run_negative malformed-prompt 2 "${scratch}/eshkol-transformer" generate \
  --prompt-hex x --sampling greedy --seed 1729
run_negative wrong-sampling 2 "${scratch}/eshkol-transformer" generate \
  --prompt-hex 41 --sampling top-p --seed 1729
run_negative duplicate-seed 2 "${generate_base[@]}" --seed 1729 \
  --sampling greedy
run_negative corrupt-checkpoint 12 "${scratch}/eshkol-transformer" generate \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/corrupt.c2" \
  --prompt-hex 41 --sampling greedy --seed 1729
run_negative malformed-checkpoint 12 "${scratch}/eshkol-transformer" generate \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/torn.c2" \
  --prompt-hex 41 --sampling greedy --seed 1729
run_negative wrong-kind-checkpoint 12 "${scratch}/eshkol-transformer" generate \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/byte.t1" \
  --prompt-hex 41 --sampling greedy --seed 1729
cmp "${scratch}/before.c2" "${scratch}/first.c2"
CLI3_GENERATE_EXPECTED=1 TMPDIR="${evidence}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/test-cli3.sh" "${scratch}" \
  >"${evidence}/six-command-suite.stdout" \
  2>"${evidence}/six-command-suite.stderr"
cmp "${prefix}_source_closure.txt" \
  "${scratch}/aggregate.o.evidence/source-closure.txt"
cmp "${prefix}_native_source_closure.txt" \
  "${scratch}/aggregate.o.evidence/native-source-closure.txt"
sha256sum "${scratch}/first.c2" "${scratch}/before.c2" \
  >"${evidence}/checkpoint-sha256.txt"
git -C "${PROJECT_ROOT}" diff --check
printf 'CLI3-GENERATE-CHECKPOINT-PREREQUISITE-GATE-PASS\n'
