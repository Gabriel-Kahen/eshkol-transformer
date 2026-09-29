#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

binary="${1:-$(project_build_dir)/cli3/eshkol-transformer}"
[[ -x "${binary}" ]] || die "CLI3 executable unavailable: ${binary}"
for command in cmp python3; do require_command "${command}"; done
scratch="$(mktemp -d "${TMPDIR:-/tmp}/cli3-b.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cat >"${scratch}/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
# Three D2 rows have mask weights 2, 2, 1; update two crosses finite EOS.
(root / 'train.txt').write_bytes(bytes((3, 197, 41, 7, 11, 19)))
(root / 'validation.txt').write_bytes(bytes((20, 21, 22, 23, 24, 25)))
config = (root / 'config.json').read_text()
(root / 'wrong-profile.json').write_text(
    config.replace('"model.context-length":2', '"model.context-length":4'))
PY
"${binary}" tokenizer byte --config "${scratch}/config.json" \
  --output "${scratch}/byte.t1" >"${scratch}/tokenizer.json"
"${binary}" corpus build --tokenizer "${scratch}/byte.t1" \
  --document "${scratch}/train.txt" --output-directory "${scratch}/train" \
  --shard-token-limit 4 >"${scratch}/train.json"
"${binary}" corpus build --tokenizer "${scratch}/byte.t1" \
  --document "${scratch}/validation.txt" \
  --output-directory "${scratch}/validation" --shard-token-limit 4 \
  >"${scratch}/validation.json"

set +e
"${binary}" pretrain --config "${scratch}/wrong-profile.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/wrong-profile.c2" --max-updates 1 \
  >"${scratch}/wrong-profile.stdout" 2>"${scratch}/wrong-profile.stderr"
status=$?
set -e
[[ "${status}" == 11 ]] || die "wrong profile returned ${status}, expected 11"
[[ ! -s "${scratch}/wrong-profile.stdout" && ! -e "${scratch}/wrong-profile.c2" ]]

"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/first.c2" --max-updates 1 \
  >"${scratch}/first.json"
"${binary}" evaluate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/validation" \
  --checkpoint "${scratch}/first.c2" >"${scratch}/evaluation.json"
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/resumed.c2" --resume "${scratch}/first.c2" \
  --max-updates 1 >"${scratch}/resumed.json"
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/uninterrupted.c2" --max-updates 2 \
  >"${scratch}/uninterrupted.json"
cmp "${scratch}/resumed.c2" "${scratch}/uninterrupted.c2"
cp "${scratch}/first.c2" "${scratch}/in-place.c2"
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/in-place.c2" --resume "${scratch}/in-place.c2" \
  --max-updates 1 --force >"${scratch}/in-place.json"
cmp "${scratch}/in-place.c2" "${scratch}/uninterrupted.c2"

python3 - "${scratch}" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
first = json.loads((root / 'first.json').read_text())
resumed = json.loads((root / 'resumed.json').read_text())
whole = json.loads((root / 'uninterrupted.json').read_text())
evaluation = json.loads((root / 'evaluation.json').read_text())
assert first['artifact'] == resumed['artifact'] == whole['artifact'] == 'checkpoint'
assert evaluation['artifact'] == 'evaluation'
assert first['updates'] == resumed['updates'] == 1
assert whole['updates'] == 2
assert whole['loss_f32_bits'] == '40aef60e'
assert first['tokens'] > 0 and resumed['tokens'] > 0
assert evaluation['tokens'] > 0 and evaluation['batches'] > 0
for record in (first, resumed, whole, evaluation):
    for key in ('loss_f32_bits', 'mask_weight_f32_bits'):
        assert re.fullmatch(r'[0-9a-f]{8}', record[key]), (key, record[key])
PY

cp "${scratch}/first.c2" "${scratch}/first-before.c2"
set +e
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/first.c2" --max-updates 1 \
  >"${scratch}/existing.stdout" 2>"${scratch}/existing.stderr"
status=$?
set -e
[[ "${status}" == 14 ]] || die "existing checkpoint returned ${status}, expected 14"
[[ ! -s "${scratch}/existing.stdout" ]]
cmp "${scratch}/first-before.c2" "${scratch}/first.c2"
set +e
"${binary}" evaluate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/train" \
  --checkpoint "${scratch}/first.c2" \
  >"${scratch}/same.stdout" 2>"${scratch}/same.stderr"
status=$?
set -e
[[ "${status}" == 10 ]] || die "same directory returned ${status}, expected 10"
[[ ! -s "${scratch}/same.stdout" ]]
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
source = bytearray((root / 'first.c2').read_bytes())
source[-1] ^= 1
(root / 'corrupt.c2').write_bytes(source)
PY
set +e
"${binary}" evaluate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/validation" \
  --checkpoint "${scratch}/corrupt.c2" \
  >"${scratch}/corrupt.stdout" 2>"${scratch}/corrupt.stderr"
status=$?
set -e
[[ "${status}" == 12 ]] || die "corrupt checkpoint returned ${status}, expected 12"
[[ ! -s "${scratch}/corrupt.stdout" ]]
cmp "${scratch}/first-before.c2" "${scratch}/first.c2"
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for name in ('wrong-profile', 'existing', 'same', 'corrupt'):
    line = (root / f'{name}.stderr').read_bytes()
    assert line.startswith(b'eshkol-transformer: ') and line.endswith(b'\n'), name
    assert line.count(b'\n') == 1 and len(line) <= 4096, name
PY
printf 'CLI3-B-FIXED-PROFILE-PASS\n'
