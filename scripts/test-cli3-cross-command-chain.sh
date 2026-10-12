#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Run only when the production CLI3 root is ready for a fresh compiled gate.
[[ $# == 1 ]] || die "usage: $0 DISK_BACKED_EVIDENCE_DIR"
verify_toolchain
for command in cmp python3 realpath; do require_command "${command}"; done
evidence="$(realpath -m -- "$1")"
mkdir -p -- "${evidence}"
scratch="$(mktemp -d "${evidence}/.cli3-cross-command.XXXXXX")"
printf '%s\n' "${scratch}" >"${evidence}/cross-command-scratch-path.txt"
TMPDIR="${evidence}" CLI3_COMPILER_TIMEOUT_SECONDS=1200 \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-cli3.sh" "${scratch}"
binary="${scratch}/eshkol-transformer"
[[ -x "${binary}" ]] || die "production CLI3 executable was not built"

run_ok() {
  local name=$1
  shift
  "$@" >"${evidence}/${name}.json" 2>"${evidence}/${name}.stderr"
  [[ -s "${evidence}/${name}.json" && ! -s "${evidence}/${name}.stderr" ]] || \
    die "${name}: expected JSON stdout and empty stderr"
}

run_rejected() {
  local name=$1 expected_status=$2 expected_category=$3
  shift 3
  local status
  set +e
  "$@" >"${evidence}/${name}.stdout" 2>"${evidence}/${name}.stderr"
  status=$?
  set -e
  [[ "${status}" == "${expected_status}" ]] || \
    die "${name}: expected status ${expected_status}, got ${status}"
  [[ ! -s "${evidence}/${name}.stdout" ]] || \
    die "${name}: rejected command wrote stdout"
  python3 - "${evidence}/${name}.stderr" "${expected_category}" <<'PY'
from pathlib import Path
import sys

line = Path(sys.argv[1]).read_bytes()
assert line.startswith(b'eshkol-transformer: '), line
assert line.endswith(b'\n') and line.count(b'\n') == 1, line
assert len(line) <= 4096 and line.isascii(), line
assert ('category="' + sys.argv[2] + '"').encode() in line, line
PY
}

cat >"${scratch}/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
(root / 'train.txt').write_bytes(bytes((3, 197, 41, 7, 11, 19)))
(root / 'heldout.txt').write_bytes(bytes((20, 21, 22, 23, 24, 25)))
config = (root / 'config.json').read_text()
assert config.count('"run.seed":1729') == 1
(root / 'other-config.json').write_text(
    config.replace('"run.seed":1729', '"run.seed":1730'))
PY

run_ok tokenizer-create "${binary}" tokenizer byte \
  --config "${scratch}/config.json" --output "${scratch}/byte.t1"
run_ok tokenizer-inspect "${binary}" tokenizer inspect \
  --input "${scratch}/byte.t1"
cmp "${evidence}/tokenizer-create.json" "${evidence}/tokenizer-inspect.json"

mkdir "${scratch}/train" "${scratch}/heldout"
run_ok train-build "${binary}" corpus build \
  --tokenizer "${scratch}/byte.t1" --document "${scratch}/train.txt" \
  --output-directory "${scratch}/train" --shard-token-limit 4
run_ok train-inspect "${binary}" corpus inspect \
  --input-directory "${scratch}/train"
cmp "${evidence}/train-build.json" "${evidence}/train-inspect.json"
run_ok heldout-build "${binary}" corpus build \
  --tokenizer "${scratch}/byte.t1" --document "${scratch}/heldout.txt" \
  --output-directory "${scratch}/heldout" --shard-token-limit 4
run_ok heldout-inspect "${binary}" corpus inspect \
  --input-directory "${scratch}/heldout"
cmp "${evidence}/heldout-build.json" "${evidence}/heldout-inspect.json"

run_ok pretrain "${binary}" pretrain \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2" \
  --max-updates 1
cp "${scratch}/first.c2" "${scratch}/first-before.c2"
run_ok checkpoint-inspect "${binary}" checkpoint inspect \
  --input "${scratch}/first.c2"
run_ok evaluate "${binary}" evaluate \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/heldout" \
  --checkpoint "${scratch}/first.c2"
run_ok generate "${binary}" generate \
  --config "${scratch}/config.json" --tokenizer "${scratch}/byte.t1" \
  --train-corpus "${scratch}/train" --checkpoint "${scratch}/first.c2" \
  --prompt-hex 41 --sampling greedy --seed 1729
cmp "${scratch}/first-before.c2" "${scratch}/first.c2"

# This valid 256-token BPE is rejected by the fixed byte-profile admission.
run_ok other-tokenizer-create "${binary}" tokenizer train-bpe \
  --document "${scratch}/train.txt" --maximum-merges 0 \
  --minimum-frequency 2 --utf8-policy raw \
  --output "${scratch}/other.t1"
run_ok other-tokenizer-inspect "${binary}" tokenizer inspect \
  --input "${scratch}/other.t1"
run_rejected evaluate-mismatched-tokenizer 11 unsupported \
  "${binary}" evaluate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/other.t1" --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/heldout" \
  --checkpoint "${scratch}/first.c2"
run_rejected generate-mismatched-tokenizer 11 unsupported \
  "${binary}" generate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/other.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/first.c2" --prompt-hex 41 \
  --sampling greedy --seed 1729

# The alternate config passes the fixed profile and differs only in run.seed.
# Restore must reject its identity against the C2 produced above.
run_rejected evaluate-config-identity 14 invalid-state \
  "${binary}" evaluate --config "${scratch}/other-config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/heldout" \
  --checkpoint "${scratch}/first.c2"
run_rejected generate-config-identity 14 invalid-state \
  "${binary}" generate --config "${scratch}/other-config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/first.c2" --prompt-hex 41 \
  --sampling greedy --seed 1729
for name in evaluate-config-identity generate-config-identity; do
  grep -F 'operation="trainer-load-state!" message="detached controls fail cross-state validation"' \
    "${evidence}/${name}.stderr" >/dev/null || \
    die "${name}: expected checkpoint/config cross-state diagnostic"
done

python3 - "${scratch}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
corrupt = bytearray((root / 'first.c2').read_bytes())
assert corrupt
corrupt[-1] ^= 1
(root / 'corrupt.c2').write_bytes(corrupt)
PY
run_rejected checkpoint-inspect-corrupt 12 corrupt-data \
  "${binary}" checkpoint inspect --input "${scratch}/corrupt.c2"
run_rejected evaluate-corrupt 12 corrupt-data \
  "${binary}" evaluate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --validation-corpus "${scratch}/heldout" \
  --checkpoint "${scratch}/corrupt.c2"
run_rejected generate-corrupt 12 corrupt-data \
  "${binary}" generate --config "${scratch}/config.json" \
  --tokenizer "${scratch}/byte.t1" --train-corpus "${scratch}/train" \
  --checkpoint "${scratch}/corrupt.c2" --prompt-hex 41 \
  --sampling greedy --seed 1729
cmp "${scratch}/first-before.c2" "${scratch}/first.c2"

python3 - "${evidence}" "${scratch}" <<'PY'
from pathlib import Path
import json
import re
import sys

evidence, scratch = map(Path, sys.argv[1:])

def record(name):
    raw = (evidence / f'{name}.json').read_bytes()
    assert raw.endswith(b'\n') and raw.count(b'\n') == 1, name
    value = json.loads(raw)
    assert isinstance(value, dict), name
    return value

tokenizer = record('tokenizer-inspect')
other = record('other-tokenizer-inspect')
train = record('train-inspect')
heldout = record('heldout-inspect')
pretrain = record('pretrain')
checkpoint = record('checkpoint-inspect')
evaluation = record('evaluate')
generation = record('generate')

assert tokenizer == {
    'artifact': 'tokenizer', 'family': 'byte',
    'fingerprint': 'sha256:eshkol-byte-tokenizer-v1:' +
        'aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704',
    'vocabulary_size': 256,
}
assert other['artifact'] == 'tokenizer' and other['family'] == 'byte-bpe'
assert other['vocabulary_size'] == 256
assert other['fingerprint'] != tokenizer['fingerprint']
for corpus in (train, heldout):
    assert set(corpus) == {'artifact', 'shard_count', 'shard_token_limit',
                           'tokenizer_fingerprint', 'total_shard_bytes',
                           'total_tokens', 'vocabulary_size'}
    assert corpus['artifact'] == 'token-corpus'
    assert corpus['tokenizer_fingerprint'] == tokenizer['fingerprint']
    assert corpus['vocabulary_size'] == tokenizer['vocabulary_size']
    assert corpus['total_tokens'] == 6
    assert corpus['shard_count'] == 2 and corpus['shard_token_limit'] == 4
assert train['total_shard_bytes'] == heldout['total_shard_bytes'] > 0
assert pretrain['artifact'] == checkpoint['artifact'] == 'checkpoint'
assert pretrain['checkpoint'] == str(scratch / 'first.c2')
assert pretrain['updates'] == 1 and pretrain['tokens'] == 4
assert set(checkpoint) == {
    'api_version', 'artifact', 'checksum_algorithm', 'config_fingerprint',
    'config_schema_version', 'format_id', 'format_version', 'payload_bytes',
    'required_features', 'tensor_count', 'tokenizer_fingerprint',
}
assert checkpoint['tokenizer_fingerprint'] == tokenizer['fingerprint']
# Independent X1 canonical-manifest SHA-256 oracle: exact input provenance,
# default CPU/f32/KV heads/determinism, derived head size, and final LF.
assert checkpoint['config_fingerprint'] == (
    'sha256:eshkol-config-json-v1:'
    '2c78042e68cc6bf3e328f68b5cce471e4b164df2ada4c6b8ca393dd286919449'
)
assert checkpoint['format_id'] == 'eshkol-training-state'
assert checkpoint['format_version'] == [1, 0]
assert checkpoint['config_schema_version'] == [1, 0]
assert checkpoint['checksum_algorithm'] == 'sha256'
assert checkpoint['required_features'] == []
assert checkpoint['tensor_count'] > 0 and checkpoint['payload_bytes'] > 0
assert evaluation['artifact'] == 'evaluation'
assert evaluation['batches'] > 0 and evaluation['tokens'] > 0
for result in (pretrain, evaluation):
    for key in ('loss_f32_bits', 'mask_weight_f32_bits'):
        assert re.fullmatch(r'[0-9a-f]{8}', result[key]), (key, result)
assert generation == {
    'artifact': 'generation', 'generated_hex': 'c5',
    'prompt_hex': '41', 'sampling': 'greedy', 'seed': 1729,
}
PY
printf 'CLI3-CROSS-COMMAND-CHAIN-PASS\n'
