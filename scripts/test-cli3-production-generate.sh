#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp python3 realpath sha256sum; do require_command "${command}"; done
[[ $# == 1 ]] || die "usage: $0 DISK_BACKED_EVIDENCE_DIR"
evidence="$(realpath -m -- "$1")"
mkdir -p -- "${evidence}"
scratch="$(mktemp -d "${evidence}/.build.XXXXXX")"
printf '%s\n' "${scratch}" >"${evidence}/scratch-path.txt"
TMPDIR="${evidence}" CLI3_COMPILER_TIMEOUT_SECONDS=1200 \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-cli3.sh" "${scratch}"
cmp "${PROJECT_ROOT}/native/cli3_generate_source_closure.txt" \
  "${scratch}/cli3.o.evidence/source-closure.txt"
cmp "${PROJECT_ROOT}/native/cli3_generate_native_source_closure.txt" \
  "${scratch}/cli3.o.evidence/native-source-closure.txt"
cmp "${PROJECT_ROOT}/native/cli3_generate_public_strings.txt" \
  "${scratch}/cli3.o.evidence/public-strings.txt"
cmp "${PROJECT_ROOT}/native/cli3_generate_public_exports.txt" \
  "${scratch}/cli3.o.evidence/package-exports.txt"
cmp "${PROJECT_ROOT}/native/cli3_generate_undefined_symbols.txt" \
  "${scratch}/cli3.o.evidence/undefined.txt"
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
base=("${scratch}/eshkol-transformer" generate
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1"
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2"
  --prompt-hex 41 --seed 1729)
for mode in greedy categorical; do
  for repetition in first second; do
    "${base[@]}" --sampling "${mode}" \
      >"${evidence}/${mode}-${repetition}.json" \
      2>"${evidence}/${mode}-${repetition}.stderr"
    [[ ! -s "${evidence}/${mode}-${repetition}.stderr" ]]
  done
  cmp "${evidence}/${mode}-first.json" "${evidence}/${mode}-second.json"
done
python3 - "${evidence}" <<'PY'
from pathlib import Path
import json
import sys
root = Path(sys.argv[1])
# Exact independent public-runtime oracle from the sealed test-installed gate.
for mode, expected in [('greedy', 'c5'), ('categorical', '80')]:
    record = json.loads((root / f'{mode}-first.json').read_text())
    assert record == {'artifact': 'generation', 'generated_hex': expected,
                      'prompt_hex': '41', 'sampling': mode, 'seed': 1729}
PY
reject() {
  local name=$1 expected=$2 category=$3
  shift 3
  set +e
  "$@" >"${evidence}/${name}.stdout" 2>"${evidence}/${name}.stderr"
  local status=$?
  set -e
  [[ "${status}" == "${expected}" && ! -s "${evidence}/${name}.stdout" ]]
  grep -F "${category}" "${evidence}/${name}.stderr" >/dev/null
}
reject malformed-prompt 2 'usage message=' \
  "${scratch}/eshkol-transformer" generate --prompt-hex x \
  --sampling greedy --seed 1729
reject wrong-sampling 2 'usage message=' \
  "${scratch}/eshkol-transformer" generate --prompt-hex 41 \
  --sampling top-p --seed 1729
reject duplicate-seed 2 'usage message=' "${base[@]}" --seed 1729 \
  --sampling greedy
for pair in corrupt.c2:corrupt-checkpoint torn.c2:malformed-checkpoint \
            byte.t1:wrong-kind-checkpoint; do
  checkpoint="${pair%%:*}"
  name="${pair#*:}"
  reject "${name}" 12 'category="corrupt-data"' \
    "${scratch}/eshkol-transformer" generate \
    --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
    --train-corpus "${scratch}/train" \
    --checkpoint "${scratch}/${checkpoint}" --prompt-hex 41 \
    --sampling greedy --seed 1729
done
cmp "${scratch}/before.c2" "${scratch}/first.c2"
CLI3_GENERATE_EXPECTED=1 TMPDIR="${evidence}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/test-cli3.sh" "${scratch}" \
  >"${evidence}/six-command-suite.stdout" \
  2>"${evidence}/six-command-suite.stderr"
printf 'CLI3-PRODUCTION-GENERATE-PASS\n'
