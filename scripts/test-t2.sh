#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp diff grep ldd nm ps python3 rg sha256sum sleep strings timeout; do
  require_command "${command}"
done

t2_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-t2.XXXXXX")"
trap 'rm -rf -- "${t2_tmp}"' EXIT
t2_fixture="${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv"
t1_fixture="${PROJECT_ROOT}/tests/t1/fixtures/byte_tokenizer_v1.tsv"
t2_runner="$(eshkol_build_dir)/eshkol-run"
t2_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
t2_cxx="$(tsv_value "${t2_provenance}" cxx_path)"

(
  cd -- "${PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
    -s tests/t2 -p 'test_*.py'
  for repetition in 1 2; do
    PYTHONDONTWRITEBYTECODE=1 python3 -m tests.t2.generate_fixture \
      --output "${t2_tmp}/fixture-${repetition}.tsv"
    PYTHONDONTWRITEBYTECODE=1 python3 -m \
      tests.t2.generate_adversarial_fixtures \
      --output-directory "${t2_tmp}/adversarial-${repetition}"
  done
)
cmp "${t2_tmp}/fixture-1.tsv" "${t2_tmp}/fixture-2.tsv"
cmp "${t2_tmp}/fixture-1.tsv" "${t2_fixture}"
diff -ru "${t2_tmp}/adversarial-1" "${t2_tmp}/adversarial-2"

E1B_COMPILER_TIMEOUT_SECONDS="${T2_COMPILER_TIMEOUT_SECONDS:-300}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-t2.sh"
t2_dir="$(project_build_dir)/t2"
t2_object="${t2_dir}/wave2.o"
t2_archive="${t2_dir}/libeshkol_transformer_wave2.a"
[[ -r "${t2_object}" && -r "${t2_archive}" ]] || \
  die "T2 Wave 2 aggregate was not published"
[[ "$(nm -g --defined-only --format=posix "${t2_object}" | wc -l)" == 46 ]] || \
  die "T2 aggregate does not expose exactly 46 globals"
cmp "${PROJECT_ROOT}/native/t2_wave2_defined_symbols.txt" \
  "${t2_object}.evidence/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_undefined_symbols.txt" \
  "${t2_object}.evidence/undefined.txt"

# Exercise the localized T2-to-D1 reader through a separate, noninstallable seam.
E1B_COMPILER_TIMEOUT_SECONDS="${T2_COMPILER_TIMEOUT_SECONDS:-300}" \
  "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/t2_wave2_d1_test_root.esk" \
  "${PROJECT_ROOT}/native/t2_wave2_d1_test_package_bridge.c" \
  "${PROJECT_ROOT}/native/t2_wave2_d1_test_private_renames.txt" \
  "${PROJECT_ROOT}/native/t2_wave2_d1_test_public_exports.txt" \
  "${t2_tmp}/wave2-d1-test.o" \
  "${PROJECT_ROOT}/internal/p1/lib" \
  "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t2/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" \
  "${PROJECT_ROOT}/src"
[[ "$(nm -g --defined-only --format=posix \
       "${t2_tmp}/wave2-d1-test.o" | wc -l)" == 47 ]] || \
  die "T2 D1 test aggregate must expose exactly 47 globals"
sed -e 's/^[^:]*://' -e 's/\\//g' \
  "${t2_tmp}/wave2-d1-test.o.evidence/private.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${t2_tmp}/d1-source-closure.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_d1_test_source_closure.txt" \
  "${t2_tmp}/d1-source-closure.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_d1_test_defined_symbols.txt" \
  "${t2_tmp}/wave2-d1-test.o.evidence/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_undefined_symbols.txt" \
  "${t2_tmp}/wave2-d1-test.o.evidence/undefined.txt"
ar rcsD "${t2_tmp}/libeshkol_transformer_wave2_d1_test.a" \
  "${t2_tmp}/wave2-d1-test.o"

compile_core() {
  local source="$1" output="$2"
  env -u ESHKOL_PATH XDG_CACHE_HOME="${output}.cache" \
    ESHKOL_CXX_COMPILER="${t2_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 300s \
    "${t2_runner}" --strict-types --no-stdlib -I "${PROJECT_ROOT}" \
    "${source}" -o "${output}"
}

run_bounded() {
  local label="$1" stdout="$2" stderr="$3"
  shift 3
  local process_id state rss status max_rss=0 start_seconds="${SECONDS}"
  "$@" >"${stdout}" 2>"${stderr}" &
  process_id=$!
  while kill -0 "${process_id}" 2>/dev/null; do
    read -r state rss \
      < <(ps -o stat=,rss= -p "${process_id}" 2>/dev/null || true) || true
    [[ "${state:-}" == Z* ]] && break
    if [[ "${rss:-}" =~ ^[0-9]+$ && "${rss}" -gt "${max_rss}" ]]; then
      max_rss="${rss}"
    fi
    if (( SECONDS - start_seconds >= 60 )); then
      kill -TERM "${process_id}" 2>/dev/null || true
      sleep 1
      kill -KILL "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
      die "${label} exceeded 60 seconds"
    fi
    sleep 0.05
  done
  if wait "${process_id}"; then status=0; else status=$?; fi
  [[ "${status}" == 0 ]] || die "${label} failed: ${status}"
  [[ "${max_rss}" -gt 0 && "${max_rss}" -le 524288 ]] || \
    die "${label} exceeded 524288 KiB RSS: ${max_rss} KiB"
  if grep -F 'Heap usage at' "${stderr}" >/dev/null; then
    die "${label} emitted a heap-pressure warning"
  fi
  printf '%s\t%s KiB peak RSS\n' "${label}" "${max_rss}"
}

for repetition in 1 2; do
  compile_core "${PROJECT_ROOT}/tests/t2/core_smoke.esk" \
    "${t2_tmp}/core-smoke-${repetition}"
  compile_core "${PROJECT_ROOT}/tests/t2/training_stream_smoke.esk" \
    "${t2_tmp}/training-stream-${repetition}"
  compile_core "${PROJECT_ROOT}/tests/t2/decoder_partition_smoke.esk" \
    "${t2_tmp}/decoder-partition-${repetition}"
  compile_core "${PROJECT_ROOT}/tests/t2/utf8_stream_smoke.esk" \
    "${t2_tmp}/utf8-stream-${repetition}"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${t2_tmp}/core-smoke-${repetition}" \
    >"${t2_tmp}/core-smoke-${repetition}.stdout"
  run_bounded "T2 training/stream ${repetition}" \
    "${t2_tmp}/training-stream-${repetition}.stdout" \
    "${t2_tmp}/training-stream-${repetition}.stderr" \
    "${t2_tmp}/training-stream-${repetition}"
  run_bounded "T2 decoder partition ${repetition}" \
    "${t2_tmp}/decoder-partition-${repetition}.stdout" \
    "${t2_tmp}/decoder-partition-${repetition}.stderr" \
    "${t2_tmp}/decoder-partition-${repetition}"
  run_bounded "T2 UTF-8 stream ${repetition}" \
    "${t2_tmp}/utf8-stream-${repetition}.stdout" \
    "${t2_tmp}/utf8-stream-${repetition}.stderr" \
    "${t2_tmp}/utf8-stream-${repetition}"
done
cmp "${t2_tmp}/core-smoke-1.stdout" "${t2_tmp}/core-smoke-2.stdout"
cmp "${t2_tmp}/training-stream-1.stdout" \
  "${t2_tmp}/training-stream-2.stdout"
cmp "${t2_tmp}/decoder-partition-1.stdout" \
  "${t2_tmp}/decoder-partition-2.stdout"
cmp "${t2_tmp}/utf8-stream-1.stdout" \
  "${t2_tmp}/utf8-stream-2.stdout"
grep -Fx 'T2 CORE SMOKE PASS: 6 checks' \
  "${t2_tmp}/core-smoke-1.stdout" >/dev/null
grep -Fx 'T2 TRAINING STREAM PASS: 91 checks' \
  "${t2_tmp}/training-stream-1.stdout" >/dev/null
grep -Fx 'T2 DECODER PARTITION PASS: 73728 one-ID omit chunks' \
  "${t2_tmp}/decoder-partition-1.stdout" >/dev/null
grep -Fx 'T2 UTF8 STREAM PASS: 60 checks' \
  "${t2_tmp}/utf8-stream-1.stdout" >/dev/null

for repetition in 1 2; do
  env -u ESHKOL_PATH XDG_CACHE_HOME="${t2_tmp}/public-cache-${repetition}" \
    ESHKOL_CXX_COMPILER="${t2_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 300s \
    "${t2_runner}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" -L "${t2_dir}" \
    --lib eshkol_transformer_wave2 \
    "${PROJECT_ROOT}/tests/t2/public_runtime.esk" \
    -o "${t2_tmp}/public-runtime-${repetition}"
  run_bounded "T2 public runtime ${repetition}" \
    "${t2_tmp}/public-${repetition}.stdout" \
    "${t2_tmp}/public-${repetition}.stderr" \
    "${t2_tmp}/public-runtime-${repetition}" \
    "${t2_fixture}" "${t2_tmp}/saved-${repetition}.tsv" \
    "${t2_tmp}/adversarial-${repetition}/truncated.tsv" \
    "${t2_tmp}/adversarial-${repetition}/bad-checksum.tsv" \
    "${t2_tmp}/adversarial-${repetition}/algorithm.tsv" \
    "${t2_tmp}/adversarial-${repetition}/oversized.tsv" \
    "${t2_tmp}/adversarial-${repetition}/forward-reference.tsv" \
    "${t2_tmp}/adversarial-${repetition}/rank-gap.tsv" \
    "${t2_tmp}/adversarial-${repetition}/merge-count-mismatch.tsv" \
    "${t2_tmp}/adversarial-${repetition}/special-id-collision.tsv" \
    "${t2_tmp}/adversarial-${repetition}/noncanonical-version.tsv"
  grep -Fx 'T2 PUBLIC PASS: 17 BPE aggregate checks' \
    "${t2_tmp}/public-${repetition}.stdout" >/dev/null
  cmp "${t2_fixture}" "${t2_tmp}/saved-${repetition}.tsv"
done
cmp "${t2_tmp}/public-1.stdout" "${t2_tmp}/public-2.stdout"

env -u ESHKOL_PATH XDG_CACHE_HOME="${t2_tmp}/parser-negatives-cache" \
  ESHKOL_CXX_COMPILER="${t2_cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
  "${t2_runner}" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -L "${t2_dir}" \
  --lib eshkol_transformer_wave2 \
  "${PROJECT_ROOT}/tests/t2/parser_negatives_runtime.esk" \
  -o "${t2_tmp}/parser-negatives-runtime"
run_bounded "T2 compiled parser negatives" \
  "${t2_tmp}/parser-negatives.stdout" \
  "${t2_tmp}/parser-negatives.stderr" \
  "${t2_tmp}/parser-negatives-runtime" "${t2_tmp}/adversarial-1"
grep -Fx 'T2 PARSER NEGATIVES PASS: 22 compiled rejection checks' \
  "${t2_tmp}/parser-negatives.stdout" >/dev/null

env -u ESHKOL_PATH XDG_CACHE_HOME="${t2_tmp}/d1-cache" \
  ESHKOL_CXX_COMPILER="${t2_cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
  "${t2_runner}" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -L "${t2_tmp}" \
  --lib eshkol_transformer_wave2_d1_test \
  "${PROJECT_ROOT}/tests/t2/d1_roundtrip_runtime.esk" \
  -o "${t2_tmp}/d1-roundtrip-runtime"
mkdir -p "${t2_tmp}/corpus"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${t2_tmp}/d1-roundtrip-runtime" "${t2_fixture}" \
  "${t2_tmp}/adversarial-1/alternate-same-vocab.tsv" \
  "${t2_tmp}/corpus" >"${t2_tmp}/d1-roundtrip.stdout"
grep -Fx 'T2 D1 ROUNDTRIP PASS: 6 checks' \
  "${t2_tmp}/d1-roundtrip.stdout" >/dev/null

env -u ESHKOL_PATH XDG_CACHE_HOME="${t2_tmp}/d1-negatives-cache" \
  ESHKOL_CXX_COMPILER="${t2_cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
  "${t2_runner}" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -L "${t2_tmp}" \
  --lib eshkol_transformer_wave2_d1_test \
  "${PROJECT_ROOT}/tests/t2/d1_negatives_runtime.esk" \
  -o "${t2_tmp}/d1-negatives-runtime"
mkdir -p "${t2_tmp}/d1-vocab-mismatch" \
  "${t2_tmp}/d1-negative-template"
run_bounded "T2 D1 negative setup" \
  "${t2_tmp}/d1-negatives-setup.stdout" \
  "${t2_tmp}/d1-negatives-setup.stderr" \
  "${t2_tmp}/d1-negatives-runtime" setup "${t2_fixture}" \
  "${t2_tmp}/d1-vocab-mismatch" "${t2_tmp}/d1-negative-template" \
  "${t2_tmp}/d1-malformed" "${t2_tmp}/d1-truncated" \
  "${t2_tmp}/d1-checksum-corrupt"
grep -Fx 'T2 D1 NEGATIVE SETUP PASS' \
  "${t2_tmp}/d1-negatives-setup.stdout" >/dev/null
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/t2/prepare_d1_negative_fixtures.py" \
  --template "${t2_tmp}/d1-negative-template" \
  --malformed "${t2_tmp}/d1-malformed" \
  --truncated "${t2_tmp}/d1-truncated" \
  --checksum-corrupt "${t2_tmp}/d1-checksum-corrupt" \
  >"${t2_tmp}/d1-negative-fixtures.stdout"
grep -Fx 'T2 D1 NEGATIVE FIXTURES PASS: 3 deterministic shard mutations' \
  "${t2_tmp}/d1-negative-fixtures.stdout" >/dev/null
run_bounded "T2 compiled D1 negatives" \
  "${t2_tmp}/d1-negatives.stdout" \
  "${t2_tmp}/d1-negatives.stderr" \
  "${t2_tmp}/d1-negatives-runtime" check "${t2_fixture}" \
  "${t2_tmp}/d1-vocab-mismatch" "${t2_tmp}/d1-negative-template" \
  "${t2_tmp}/d1-malformed" "${t2_tmp}/d1-truncated" \
  "${t2_tmp}/d1-checksum-corrupt"
grep -Fx 'T2 D1 NEGATIVES PASS: 4 compiled seam checks' \
  "${t2_tmp}/d1-negatives.stdout" >/dev/null

# The successor aggregate must run the accepted T1 artifact and exact public test.
env -u ESHKOL_PATH XDG_CACHE_HOME="${t2_tmp}/t1-cache" \
  ESHKOL_CXX_COMPILER="${t2_cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
  "${t2_runner}" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -L "${t2_dir}" \
  --lib eshkol_transformer_wave2 \
  "${PROJECT_ROOT}/tests/t1/public_runtime.esk" \
  -o "${t2_tmp}/t1-through-wave2"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${t2_tmp}/t1-through-wave2" "${t1_fixture}" \
  "${t2_tmp}/t1-saved.tsv" >"${t2_tmp}/t1-through-wave2.stdout"
grep -Fx 'T1 PUBLIC PASS: 23 aggregate tokenizer checks' \
  "${t2_tmp}/t1-through-wave2.stdout" >/dev/null
cmp "${t1_fixture}" "${t2_tmp}/t1-saved.tsv"

if rg -n -i '\b(import|from) (torch|pytorch|python)|python\.h|py_' \
    "${PROJECT_ROOT}/internal/t2" "${PROJECT_ROOT}/native/t2_wave2_root.esk" \
    "${PROJECT_ROOT}/native/t2_wave2_package_bridge.c" >/dev/null; then
  die "T2 production sources reference a Python runtime"
fi
if strings -a "${t2_object}" | \
   grep -E 'tests/t2|test_reference\.py|torch' >/dev/null; then
  die "T2 production aggregate contains a development-oracle path"
fi
for delivered in "${t2_object}" "${t2_tmp}/public-runtime-1"; do
  if nm -a "${delivered}" | grep -E '(^|[[:space:]])Py_|libpython' >/dev/null || \
     ldd "${delivered}" 2>/dev/null | grep -Ei 'python|torch' >/dev/null; then
    die "T2 delivered binary contains a development-oracle dependency"
  fi
done

T2_COMPILER_TIMEOUT_SECONDS="${T2_COMPILER_TIMEOUT_SECONDS:-300}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/test-t2-boundary.sh"

printf 'T2 PASS: deterministic BPE training, artifact, whole/stream runtime, T1 compatibility, and isolation gates\n'
