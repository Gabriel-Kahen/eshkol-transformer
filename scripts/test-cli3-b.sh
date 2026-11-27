#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

binary="${1:-$(project_build_dir)/cli3/eshkol-transformer}"
[[ -x "${binary}" ]] || die "CLI3 executable unavailable: ${binary}"
for command in cmp python3; do require_command "${command}"; done
scratch="$(mktemp -d "${TMPDIR:-/tmp}/cli3-b.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

bash "${PROJECT_ROOT}/scripts/prepare-cli3-b-fixed-training.sh" "${binary}" "${scratch}"
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
(root / 'validation.txt').write_bytes(bytes((20, 21, 22, 23, 24, 25)))
config = (root / 'config.json').read_text()
(root / 'wrong-profile.json').write_text(
    config.replace('"model.context-length":2', '"model.context-length":4'))
PY
mkdir "${scratch}/validation"
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
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/four.c2" --max-updates 4 \
  >"${scratch}/four.json"
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/resumed-three.c2" --resume "${scratch}/first.c2" \
  --max-updates 3 >"${scratch}/resumed-three.json"
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/chain-three.c2" --resume "${scratch}/resumed.c2" \
  --max-updates 1 >"${scratch}/chain-three.json"
"${binary}" pretrain --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/chain-four.c2" --resume "${scratch}/chain-three.c2" \
  --max-updates 1 >"${scratch}/chain-four.json"
cmp "${scratch}/four.c2" "${scratch}/resumed-three.c2"
cmp "${scratch}/four.c2" "${scratch}/chain-four.c2"

python3 - "${scratch}" <<'PY'
import json
from pathlib import Path
import re
import struct
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
assert (first['tokens'], resumed['tokens'], whole['tokens']) == (4, 3, 7)
assert (first['mask_weight_f32_bits'], resumed['mask_weight_f32_bits'],
        whole['mask_weight_f32_bits']) == ('40800000', '40400000', '40e00000')
# Independent raw-observation binary32 replay in observation_reference.c.
assert (first['loss_f32_bits'], resumed['loss_f32_bits'],
        whole['loss_f32_bits']) == ('40b21550', '40aacc63', '40aef60f')
# Cross-check aggregate reporting against the two public update metrics,
# weighting by their D2 mask counts and rounding once to f32.
def f32(word):
    return struct.unpack('>f', bytes.fromhex(word))[0]

weighted = (4 * f32(first['loss_f32_bits']) +
            3 * f32(resumed['loss_f32_bits'])) / 7
expected_loss_bits = struct.pack('>f', weighted).hex()
assert whole['loss_f32_bits'] == expected_loss_bits
assert evaluation['tokens'] > 0 and evaluation['batches'] > 0
records = [first, resumed, whole, evaluation]
for name, updates, tokens, epochs, total_tokens, total_updates, total_epochs in (
        ('first', 1, 4, 0, 4, 1, 0),
        ('resumed', 1, 3, 1, 7, 2, 1),
        ('uninterrupted', 2, 7, 1, 7, 2, 1),
        ('in-place', 1, 3, 1, 7, 2, 1),
        ('chain-three', 1, 3, 0, 10, 3, 1),
        ('chain-four', 1, 4, 1, 14, 4, 2),
        ('resumed-three', 3, 10, 2, 14, 4, 2),
        ('four', 4, 14, 2, 14, 4, 2)):
    record = json.loads((root / f'{name}.json').read_text())
    records.append(record)
    assert record['artifact'] == 'checkpoint', name
    assert (record['updates'], record['tokens'], record['epochs']) == (
        updates, tokens, epochs), name
    # C2 1.0 header offsets 184, 192, 200: cumulative counters.
    checkpoint = (root / f'{name}.c2').read_bytes()
    assert struct.unpack_from('<QQQ', checkpoint, 184) == (
        total_tokens, total_updates, total_epochs), name
for record in records:
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
