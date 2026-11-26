#!/usr/bin/env bash
set -euo pipefail

[[ $# == 1 && -x $1 ]] || { echo 'usage: test-cli3-downloaded-fixture.sh BINARY' >&2; exit 2; }
binary=$1
scratch=$(mktemp -d)
trap 'rm -rf -- "${scratch}"' EXIT
cat >"${scratch}/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
printf '\003\305\051\007\013\023' >"${scratch}/train.txt"
printf '\024\025\026\027\030\031' >"${scratch}/heldout.txt"
"${binary}" --version >"${scratch}/version.txt"
cmp -s "${scratch}/version.txt" <(printf 'eshkol-transformer 0.0.1\n')
"${binary}" tokenizer byte --config "${scratch}/config.json" \
  --output "${scratch}/byte.t1" >"${scratch}/tokenizer.json"
mkdir "${scratch}/train" "${scratch}/heldout"
"${binary}" corpus prepare --tokenizer "${scratch}/byte.t1" \
  --source "train=${scratch}/train.txt" --minimum-bytes 6 --maximum-bytes 6 \
  --output-directory "${scratch}/train" --shard-token-limit 4 >"${scratch}/prepare.json"
"${binary}" corpus report-validate --input-directory "${scratch}/train" \
  >"${scratch}/report-validate.json"
cmp "${scratch}/prepare.json" "${scratch}/report-validate.json"
"${binary}" corpus build --tokenizer "${scratch}/byte.t1" \
  --document "${scratch}/heldout.txt" --output-directory "${scratch}/heldout" \
  --shard-token-limit 4 >"${scratch}/heldout.json"
"${binary}" pretrain --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2" \
  --max-updates 1 >"${scratch}/first.json"
"${binary}" checkpoint inspect --input "${scratch}/first.c2" >"${scratch}/inspect.json"
"${binary}" pretrain --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/resumed.c2" \
  --resume "${scratch}/first.c2" --max-updates 1 >"${scratch}/resumed.json"
"${binary}" pretrain --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/uninterrupted.c2" \
  --max-updates 2 >"${scratch}/uninterrupted.json"
cmp "${scratch}/resumed.c2" "${scratch}/uninterrupted.c2"
"${binary}" evaluate --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --validation-corpus "${scratch}/heldout" \
  --checkpoint "${scratch}/first.c2" >"${scratch}/evaluate.json"
"${binary}" generate --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2" \
  --prompt-hex 41 --sampling greedy --seed 1729 >"${scratch}/generate.json"
python3 -E - "${scratch}" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
def record(name):
    raw = (root / f'{name}.json').read_bytes()
    assert raw.endswith(b'\n') and raw.count(b'\n') == 1, name
    return json.loads(raw)
report = (root / 'train/data4-report.tsv').read_bytes()
prefix, checksum = report.rsplit(b'checksum\t', 1)
assert checksum == hashlib.sha256(prefix).hexdigest().encode() + b'\n'
lines = prefix.splitlines()
assert lines[1] == b'version\t2\t0'
assert lines[15] == b'manifest-sha256\t' + hashlib.sha256(
    (root / 'train/manifest.etm').read_bytes()).hexdigest().encode()
assert record('prepare') == record('report-validate')
assert record('prepare')['total_tokens'] == 6
assert record('first')['updates'] == 1
assert record('inspect')['format_id'] == 'eshkol-training-state'
assert record('resumed')['updates'] == 1
assert record('uninterrupted')['updates'] == 2
assert record('evaluate')['artifact'] == 'evaluation'
assert record('generate') == {'artifact': 'generation', 'generated_hex': 'c5',
    'prompt_hex': '41', 'sampling': 'greedy', 'seed': 1729}
PY
printf 'CLI3 downloaded fixed-profile fixture: PASS\n'
