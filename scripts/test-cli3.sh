#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

artifact_dir="${1:-$(project_build_dir)/cli3}"
executable="${artifact_dir}/eshkol-transformer"
[[ -x "${executable}" ]] || die "CLI3 executable not found: ${executable}"
for command in cmp python3 sha256sum; do
  require_command "${command}"
done

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-cli3.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

run_exact() {
  local name=$1 expected_status=$2
  shift 2
  set +e
  "$@" >"${temporary_dir}/${name}.stdout" \
    2>"${temporary_dir}/${name}.stderr"
  local status=$?
  set -e
  [[ "${status}" == "${expected_status}" ]] || \
    die "${name}: expected status ${expected_status}, got ${status}"
}

printf 'eshkol-transformer 0.0.1\n' >"${temporary_dir}/version.expected"
run_exact version 0 "${executable}" --version
cmp "${temporary_dir}/version.expected" "${temporary_dir}/version.stdout"
[[ ! -s "${temporary_dir}/version.stderr" ]]

cat >"${temporary_dir}/help.expected" <<'EOF'
Usage: eshkol-transformer GROUP COMMAND [OPTIONS]

Groups:
  tokenizer   Create and inspect tokenizer artifacts
  corpus      Build and inspect D1 token corpora
  checkpoint  Inspect C2 checkpoints
EOF
run_exact help 0 "${executable}" --help
cmp "${temporary_dir}/help.expected" "${temporary_dir}/help.stdout"
[[ ! -s "${temporary_dir}/help.stderr" ]]

run_exact help-extra 2 "${executable}" tokenizer byte --help extra
[[ ! -s "${temporary_dir}/help-extra.stdout" ]]
grep -Fx 'eshkol-transformer: usage message="unknown option --help"' \
  "${temporary_dir}/help-extra.stderr" >/dev/null
run_exact nonascii-option 2 "${executable}" tokenizer byte --é
grep -Fx 'eshkol-transformer: usage message="unknown option --\u00e9"' \
  "${temporary_dir}/nonascii-option.stderr" >/dev/null

byte_path="${temporary_dir}/byte.tsv"
run_exact byte 0 "${executable}" tokenizer byte \
  --output "${byte_path}" \
  --config "${PROJECT_ROOT}/tests/x1/fixtures/minimal_config_v1.json"
grep -Fx '{"artifact":"tokenizer","family":"byte","fingerprint":"sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704","vocabulary_size":256}' \
  "${temporary_dir}/byte.stdout" >/dev/null
[[ ! -s "${temporary_dir}/byte.stderr" ]]
run_exact byte-existing 13 "${executable}" tokenizer byte \
  --config "${PROJECT_ROOT}/tests/x1/fixtures/minimal_config_v1.json" \
  --output "${byte_path}"
[[ ! -s "${temporary_dir}/byte-existing.stdout" ]]
grep -F 'category="io"' "${temporary_dir}/byte-existing.stderr" >/dev/null

run_exact byte-inspect 0 "${executable}" tokenizer inspect \
  --max-metadata-bytes 1048576 --input "${byte_path}"
cmp "${temporary_dir}/byte.stdout" "${temporary_dir}/byte-inspect.stdout"

printf 'banana banana' >"${temporary_dir}/document-a"
printf 'bandana' >"${temporary_dir}/document-b"
printf 'banana' >"${temporary_dir}/document-c"
bpe_path="${temporary_dir}/bpe.tsv"
run_exact bpe 0 "${executable}" tokenizer train-bpe \
  --document "${temporary_dir}/document-a" --maximum-merges 8 \
  --special invalid=error --document "${temporary_dir}/document-b" \
  --prefix bos --minimum-frequency 2 --special bos=omit \
  --document "${temporary_dir}/document-c" --suffix eos \
  --utf8-policy raw --special eos=omit --output "${bpe_path}"
cmp "${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv" "${bpe_path}"
grep -F '"fingerprint":"sha256:eshkol-bpe-tokenizer-v1:1866ebedd76bf7d0e8e111ab25603a99aee927ea341e1305e4bb2c145648eb72"' \
  "${temporary_dir}/bpe.stdout" >/dev/null

run_exact bpe-reversed 0 "${executable}" tokenizer train-bpe \
  --output "${temporary_dir}/bpe-reversed.tsv" --document "${temporary_dir}/document-c" \
  --document "${temporary_dir}/document-b" --document "${temporary_dir}/document-a" \
  --minimum-frequency 2 --maximum-merges 8 --utf8-policy raw \
  --special eos=omit --special invalid=error --special bos=omit \
  --prefix bos --suffix eos
cmp "${bpe_path}" "${temporary_dir}/bpe-reversed.tsv"

run_exact specials 0 "${executable}" tokenizer inspect \
  --special eos --input "${bpe_path}" --special bos
grep -Fx '{"artifact":"tokenizer","family":"byte-bpe","fingerprint":"sha256:eshkol-bpe-tokenizer-v1:1866ebedd76bf7d0e8e111ab25603a99aee927ea341e1305e4bb2c145648eb72","vocabulary_size":263,"specials":{"bos":260,"eos":261}}' \
  "${temporary_dir}/specials.stdout" >/dev/null
run_exact duplicate-special 2 "${executable}" tokenizer inspect \
  --input "${bpe_path}" --special bos --special bos
[[ ! -s "${temporary_dir}/duplicate-special.stdout" ]]

mkdir "${temporary_dir}/corpus"
printf 'banana bandana' >"${temporary_dir}/corpus-input"
run_exact corpus-build 0 "${executable}" corpus build \
  --document "${temporary_dir}/corpus-input" --shard-token-limit 3 \
  --tokenizer "${bpe_path}" --output-directory "${temporary_dir}/corpus"
run_exact corpus-inspect 0 "${executable}" corpus inspect \
  --max-total-tokens 7 --input-directory "${temporary_dir}/corpus"
cmp "${temporary_dir}/corpus-build.stdout" \
  "${temporary_dir}/corpus-inspect.stdout"
grep -F '"shard_count":3' "${temporary_dir}/corpus-build.stdout" >/dev/null
grep -F '"total_tokens":7' "${temporary_dir}/corpus-build.stdout" >/dev/null
python3 - "${temporary_dir}/corpus" <<'PY'
import pathlib, struct, sys
directory = pathlib.Path(sys.argv[1])
values = []
for path in sorted(directory.glob("shard-*.ets")):
    data = path.read_bytes()
    header = struct.unpack_from("<I", data, 12)[0]
    count = struct.unpack_from("<Q", data, 40)[0]
    values.extend(struct.unpack_from(f"<{count}q", data, header))
assert values == [260, 259, 32, 257, 100, 258, 261], values
PY

checkpoint_dir="${temporary_dir}/checkpoint"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.c2.prepare_checkpoint_load_fixtures \
  "${checkpoint_dir}"
run_exact checkpoint 0 "${executable}" checkpoint inspect \
  --input "${checkpoint_dir}/valid.c2"
grep -Fx '{"api_version":"0.1.0-draft","artifact":"checkpoint","checksum_algorithm":"sha256","config_fingerprint":"sha256:eshkol-config-json-v1:a27ed00686df9879a06d4cb177c65807b3443a41463e8a397f8f1672562242b0","config_schema_version":[1,0],"format_id":"eshkol-training-state","format_version":[1,0],"payload_bytes":24,"required_features":[],"tensor_count":3,"tokenizer_fingerprint":"sha256:eshkol-byte-tokenizer-v1:0000000000000000000000000000000000000000000000000000000000000000"}' \
  "${temporary_dir}/checkpoint.stdout" >/dev/null
run_exact checkpoint-limit 12 "${executable}" checkpoint inspect \
  --max-file-bytes 1 --input "${checkpoint_dir}/valid.c2"
grep -F 'category="corrupt-data"' \
  "${temporary_dir}/checkpoint-limit.stderr" >/dev/null
inspect_negative_dir="${temporary_dir}/checkpoint-negatives"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.c2.prepare_checkpoint_inspect_fixtures \
  "${inspect_negative_dir}"
run_exact checkpoint-checksum 12 "${executable}" checkpoint inspect \
  --input "${inspect_negative_dir}/checksum.c2"
grep -F 'category="corrupt-data"' \
  "${temporary_dir}/checkpoint-checksum.stderr" >/dev/null
run_exact checkpoint-unsupported 11 "${executable}" checkpoint inspect \
  --input "${inspect_negative_dir}/compiler.c2"
grep -F 'category="unsupported"' \
  "${temporary_dir}/checkpoint-unsupported.stderr" >/dev/null

cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -shared -fPIC \
  "${PROJECT_ROOT}/tests/cli3/write_fault.c" -ldl \
  -o "${temporary_dir}/write-fault.so"
run_exact stdout-short 0 env LD_PRELOAD="${temporary_dir}/write-fault.so" \
  CLI3_TEST_WRITE_MODE=short-once "${executable}" tokenizer inspect \
  --input "${byte_path}"
cmp "${temporary_dir}/byte.stdout" "${temporary_dir}/stdout-short.stdout"
[[ ! -s "${temporary_dir}/stdout-short.stderr" ]]

fault_artifact="${temporary_dir}/fault-published.tsv"
run_exact stdout-error 13 env LD_PRELOAD="${temporary_dir}/write-fault.so" \
  CLI3_TEST_WRITE_MODE=error-after-prefix "${executable}" tokenizer byte \
  --config "${PROJECT_ROOT}/tests/x1/fixtures/minimal_config_v1.json" \
  --output "${fault_artifact}"
[[ "$(wc -c <"${temporary_dir}/stdout-error.stdout")" == 7 ]]
cmp "${byte_path}" "${fault_artifact}"
grep -F 'category="io"' "${temporary_dir}/stdout-error.stderr" >/dev/null
grep -F '"published?":true' "${temporary_dir}/stdout-error.stderr" >/dev/null

if ldd "${executable}" | grep -Ei 'python|torch' >/dev/null; then
  die "CLI3 executable links a Python/PyTorch runtime"
fi
printf 'CLI3 TARGETED PASS: six commands, grammar, artifacts, and stdout faults\n'
