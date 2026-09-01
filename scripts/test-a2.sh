#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command cmp
require_command grep
require_command nm
require_command python3
require_command sha256sum
require_command timeout

A2_PROVENANCE="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
A2_CC="$(tsv_value "${A2_PROVENANCE}" cc_path)"
A2_CXX="$(tsv_value "${A2_PROVENANCE}" cxx_path)"
A2_TMP="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-a2.XXXXXX")"
trap 'rm -rf -- "${A2_TMP}"' EXIT
A2_ARTIFACT_DIR="$(project_build_dir)/a2"
A2_LIBRARY="${A2_ARTIFACT_DIR}/libeshkol_transformer_a2.a"
K1_LIBRARY="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
A2_RUNNER="$(eshkol_build_dir)/eshkol-run"
A2_CFLAGS=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -I "${PROJECT_ROOT}/include"
)

[[ -r "${A2_LIBRARY}" ]] || die "canonical A2 archive is missing"
[[ "$(ar t "${A2_LIBRARY}")" == $'a2_attention_provider.o\na2_kv_cache.o' ]] || \
  die "canonical A2 archive has unexpected members"
nm -g --defined-only "${A2_LIBRARY}" | \
  awk '$2 ~ /^[TDBR]$/ { print $3 }' | LC_ALL=C sort -u \
  >"${A2_TMP}/a2_defined_symbols.txt"
cmp "${PROJECT_ROOT}/tests/a2/expected/a2_defined_symbols.txt" \
  "${A2_TMP}/a2_defined_symbols.txt"
nm -u "${A2_LIBRARY}" | awk 'NF != 0 && $NF !~ /:$/ { print $NF }' | \
  grep -v '^__stack_chk_fail$' | LC_ALL=C sort -u \
  >"${A2_TMP}/a2_undefined_symbols.txt"
if grep -Fvx -f \
    "${PROJECT_ROOT}/tests/a2/expected/a2_allowed_undefined_symbols.txt" \
    "${A2_TMP}/a2_undefined_symbols.txt"; then
  die "A2 archive has an unexpected native dependency"
fi
provider_exports="$(nm -g --defined-only \
  "${A2_ARTIFACT_DIR}/a2_attention_provider.o" | \
  awk '$2 == "T" { print $3 }')"
[[ "${provider_exports}" == 'et_a2_kernel_provider_v1' ]] || \
  die "A2 numerical provider has an unexpected export boundary"
if nm -u "${A2_ARTIFACT_DIR}/a2_attention_provider.o" | \
    grep -E '(^|[[:space:]_])(malloc|calloc|realloc|free)(@|$)' >/dev/null; then
  die "A2 numerical provider has a hidden allocation dependency"
fi
if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/a2_attention_provider.c" \
    "${PROJECT_ROOT}/native/a2_kv_cache.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/a2_attention_abi.h" \
    "${PROJECT_ROOT}/include/eshkol_transformer/a2_kv_cache.h"; then
  die "A2 production path contains a forbidden Python/PyTorch reference"
fi

# Merely linking A2 must not alter K1's provider-free discovery path.
"${A2_CC}" "${A2_CFLAGS[@]}" \
  "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  "${A2_LIBRARY}" "${K1_LIBRARY}" -lm -o "${A2_TMP}/report-baseline-a2-linked"
LC_ALL=C "${A2_TMP}/report-baseline-a2-linked" \
  >"${A2_TMP}/baseline_report_v1.json"
(cd "${A2_TMP}" && sha256sum -c \
  "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256")

"${A2_CC}" "${A2_CFLAGS[@]}" \
  "${PROJECT_ROOT}/tests/a2/test_attention_provider.c" \
  "${A2_LIBRARY}" "${K1_LIBRARY}" -lm -o "${A2_TMP}/test_attention_provider"
"${A2_CC}" "${A2_CFLAGS[@]}" -DET_A2_KV_CACHE_TESTING \
  "${PROJECT_ROOT}/native/a2_kv_cache.c" \
  "${PROJECT_ROOT}/tests/a2/test_kv_cache.c" \
  "${K1_LIBRARY}" -lm -o "${A2_TMP}/test_kv_cache"
"${A2_CC}" "${A2_CFLAGS[@]}" -O2 \
  "${PROJECT_ROOT}/tests/a2/test_cache_attention.c" \
  "${A2_LIBRARY}" "${K1_LIBRARY}" -lm \
  -o "${A2_TMP}/test_cache_attention"
for source in test_attention_provider test_kv_cache test_cache_attention; do
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${A2_TMP}/${source}" >"${A2_TMP}/${source}-${run}.stdout"
  done
  cmp "${A2_TMP}/${source}-1.stdout" "${A2_TMP}/${source}-2.stdout"
done
grep -E '^A2 provider PASS \([0-9]+ checks\)$' \
  "${A2_TMP}/test_attention_provider-1.stdout" >/dev/null
grep -E '^A2 KV CACHE PASS: [0-9]+ ABI, transaction, lease, atomicity, and adversarial checks$' \
  "${A2_TMP}/test_kv_cache-1.stdout" >/dev/null
grep -E '^A2 CACHE ATTENTION PASS: [0-9]+ incremental/full parity, mask, tail, and determinism checks$' \
  "${A2_TMP}/test_cache_attention-1.stdout" >/dev/null

"${A2_CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests/a2/header_cpp.cpp" \
  "${A2_LIBRARY}" "${K1_LIBRARY}" -lm -o "${A2_TMP}/header-cpp"
"${A2_TMP}/header-cpp"

# This fixed-buffer Eshkol bridge is test-only. It links the real provider/cache
# objects without adding a transport, carrier, or generic resolver to A2.
"${A2_CC}" "${A2_CFLAGS[@]}" -c \
  "${PROJECT_ROOT}/tests/a2/aot_private_bridge.c" \
  -o "${A2_TMP}/aot_private_bridge.o"
a2_test_exports="$(nm -g --defined-only "${A2_TMP}/aot_private_bridge.o" | \
  awk '$2 == "T" { print $3 }' | LC_ALL=C sort)"
[[ "${a2_test_exports}" == \
  $'et_a2_test_cache_transport_v1\net_a2_test_provider_transport_v1' ]] || \
  die "A2 private AOT bridge has an unexpected export boundary"
nm -u "${A2_TMP}/aot_private_bridge.o" | \
  grep -F 'et_a2_kernel_provider_v1' >/dev/null
if nm -g --defined-only "${A2_LIBRARY}" | \
    grep -E 'et_a2_test_|eshkol_transformer_kernel_provider_v1' >/dev/null; then
  die "test transport or generic provider symbol leaked into the A2 archive"
fi
ar rcsD "${A2_TMP}/libeshkol_transformer_a2_private_aot.a" \
  "${A2_TMP}/aot_private_bridge.o" \
  "${A2_ARTIFACT_DIR}/a2_attention_provider.o" \
  "${A2_ARTIFACT_DIR}/a2_kv_cache.o" \
  "$(project_build_dir)/k1/kernel_abi.o"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${A2_RUNNER}" --strict-types --emit-object \
  --emit-depfile "${A2_TMP}/aot_private.d" --no-stdlib \
  "${PROJECT_ROOT}/tests/a2/aot_private.esk" \
  -o "${A2_TMP}/aot_private.o" \
  >"${A2_TMP}/aot-private-object.stdout" \
  2>"${A2_TMP}/aot-private-object.stderr"
grep -F 'tests/a2/aot_private.esk' "${A2_TMP}/aot_private.d" >/dev/null
for a2_aot_run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${A2_RUNNER}" --strict-types --no-stdlib \
    -L "${A2_TMP}" --lib eshkol_transformer_a2_private_aot \
    "${PROJECT_ROOT}/tests/a2/aot_private.esk" \
    -o "${A2_TMP}/aot-private-${a2_aot_run}" \
    >"${A2_TMP}/aot-private-compile-${a2_aot_run}.stdout" \
    2>"${A2_TMP}/aot-private-compile-${a2_aot_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s \
    "${A2_TMP}/aot-private-${a2_aot_run}" \
    >"${A2_TMP}/aot-private-${a2_aot_run}.stdout" \
    2>"${A2_TMP}/aot-private-${a2_aot_run}.stderr"
  cmp "${PROJECT_ROOT}/tests/a2/expected/aot_private.stdout" \
    "${A2_TMP}/aot-private-${a2_aot_run}.stdout"
  test ! -s "${A2_TMP}/aot-private-${a2_aot_run}.stderr"
done
cmp "${A2_TMP}/aot-private-1.stdout" \
  "${A2_TMP}/aot-private-2.stdout"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-a2.sh" "${A2_TMP}/sanitized" sanitize
"${A2_CC}" "${A2_CFLAGS[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${PROJECT_ROOT}/tests/a2/test_attention_provider.c" \
  "${A2_TMP}/sanitized/libeshkol_transformer_a2.a" "${K1_LIBRARY}" -lm \
  -o "${A2_TMP}/test_attention_provider-sanitized"
"${A2_CC}" "${A2_CFLAGS[@]}" -DET_A2_KV_CACHE_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/native/a2_kv_cache.c" \
  "${PROJECT_ROOT}/tests/a2/test_kv_cache.c" "${K1_LIBRARY}" -lm \
  -o "${A2_TMP}/test_kv_cache-sanitized"
"${A2_CC}" "${A2_CFLAGS[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/a2/test_cache_attention.c" \
  "${A2_TMP}/sanitized/libeshkol_transformer_a2.a" "${K1_LIBRARY}" -lm \
  -o "${A2_TMP}/test_cache_attention-sanitized"
for source in test_attention_provider test_kv_cache test_cache_attention; do
  ASAN_OPTIONS=detect_leaks="${A2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${A2_TMP}/${source}-sanitized" >/dev/null
done

python3 -m unittest -v tests.a2.test_reference

printf 'A2 PASS: provider, attention/RoPE numerics, transactional cache, private Eshkol AOT, oracle, C/C++ ABI, and sanitizers\n'
