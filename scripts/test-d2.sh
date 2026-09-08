#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar awk cmp find grep ldd nm python3 rg sha256sum sleep strings timeout; do
  require_command "${command}"
done

d2_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-d2.XXXXXX")"
trap 'rm -rf -- "${d2_tmp}"' EXIT
d2_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
d2_cc="$(tsv_value "${d2_provenance}" cc_path)"
d2_cxx="$(tsv_value "${d2_provenance}" cxx_path)"
d2_runner="$(eshkol_build_dir)/eshkol-run"
d2_dir="$(project_build_dir)/d2"
d2_wave2_library="${d2_dir}/libeshkol_transformer_wave2.a"
d2_fixture="${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv"
d2_library="${d2_dir}/libeshkol_transformer_d2_private.a"
i1_library="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
d2_timeout="${D2_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${d2_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "D2_COMPILER_TIMEOUT_SECONDS must be a positive integer"

[[ -r "${d2_library}" && -r "${d2_dir}/d2_native.o" ]] || \
  die "canonical D2 private archive or object is missing"
[[ -r "${d2_wave2_library}" && -r "${d2_dir}/d2_wave2.o" ]] || \
  die "canonical D2 aggregate archive or object is missing"
[[ "$(ar t "${d2_library}")" == "d2_native.o" ]] || \
  die "canonical D2 private archive has unexpected members"
[[ -r "${i1_library}" && -r "${k1_library}" ]] || \
  die "canonical I1/K1 dependencies are missing"
if nm -g --defined-only --format=posix "${d2_library}" | \
    grep -E '^eshkol_transformer_kernel_provider_v1[[:space:]]'; then
  die "D2 private archive defines the forbidden canonical K1 provider symbol"
fi
nm -g --defined-only --format=posix "${d2_dir}/d2_native.o" | \
  awk '{ print $1 }' >"${d2_tmp}/d2-defined.txt"
nm -u --format=posix "${d2_dir}/d2_native.o" | \
  awk '{ print $1 }' >"${d2_tmp}/d2-undefined.txt"
cmp "${PROJECT_ROOT}/native/d2_native_defined_symbols.txt" \
  "${d2_tmp}/d2-defined.txt"
cmp "${PROJECT_ROOT}/native/d2_native_undefined_symbols.txt" \
  "${d2_tmp}/d2-undefined.txt"

(
  cd -- "${PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
    -s tests/d2 -p 'test_*.py'
  for repetition in 1 2; do
    PYTHONDONTWRITEBYTECODE=1 python3 -m tests.d2.generate_batch_fixture \
      --output "${d2_tmp}/fixture-${repetition}.json"
  done
)
cmp "${d2_tmp}/fixture-1.json" "${d2_tmp}/fixture-2.json"
cmp "${d2_tmp}/fixture-1.json" \
  "${PROJECT_ROOT}/tests/d2/fixtures/shifted_batch_v1.json"

compile_semantic_core() {
  local label=$1
  mkdir -p "${d2_tmp}/${label}" "${d2_tmp}/cache-${label}"
  if ! env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${d2_tmp}/cache-${label}" \
      ESHKOL_CXX_COMPILER="${d2_cxx}" \
      timeout --foreground --signal=TERM --kill-after=5s "${d2_timeout}s" \
      "${d2_runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}" "${PROJECT_ROOT}/tests/d2/semantic_core.esk" \
      -o "${d2_tmp}/${label}/d2-semantic" \
      >"${d2_tmp}/${label}/compile.stdout" \
      2>"${d2_tmp}/${label}/compile.stderr"; then
    sed -n '1,240p' "${d2_tmp}/${label}/compile.stderr" >&2
    die "D2 strict AOT semantic-core compilation failed"
  fi
  [[ -x "${d2_tmp}/${label}/d2-semantic" ]] || \
    die "D2 strict AOT executable is missing"
  ! rg -F 'ERROR:' "${d2_tmp}/${label}/compile.stderr" >/dev/null || \
    die "D2 compiler reported ERROR while returning success"
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${d2_tmp}/${label}/d2-semantic" \
    >"${d2_tmp}/${label}/run.stdout" \
    2>"${d2_tmp}/${label}/run.stderr"
  test ! -s "${d2_tmp}/${label}/run.stderr"
  grep -E '^D2 SEMANTIC CORE PASS: [0-9]+ checks$' \
    "${d2_tmp}/${label}/run.stdout" >/dev/null
}

compile_semantic_core a
compile_semantic_core b
cmp "${d2_tmp}/a/d2-semantic" "${d2_tmp}/b/d2-semantic"
cmp "${d2_tmp}/a/run.stdout" "${d2_tmp}/b/run.stdout"
mapfile -t d2_direct_loads < <(
  sed -n 's/^(load "\([^"]*\)").*/\1/p' \
    "${PROJECT_ROOT}/tests/d2/semantic_core.esk"
)
[[ "${#d2_direct_loads[@]}" == 2 &&
   "${d2_direct_loads[0]}" == "internal/c1/lib/c1_sha256.esk" &&
   "${d2_direct_loads[1]}" == "internal/d2/lib/d2_semantic_core.esk" ]] || \
  die "D2 semantic executable source closure changed"
if rg -n '^\(load ' "${PROJECT_ROOT}/internal/d2/lib/d2_semantic_core.esk" \
    >/dev/null; then
  die "D2 semantic core gained an undeclared transitive source load"
fi

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-d2.sh" \
  "${d2_tmp}/native-test" test
d2_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
"${d2_cc}" "${d2_cflags[@]}" -DET_D2_NATIVE_TESTING \
  -DET_I64_TENSOR_TESTING \
  "${PROJECT_ROOT}/tests/d2/test_d2_native.c" \
  "${d2_tmp}/native-test/libeshkol_transformer_d2_private.a" \
  "${PROJECT_ROOT}/native/i64_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -o "${d2_tmp}/test-d2-native"
for repetition in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${d2_tmp}/test-d2-native" \
    >"${d2_tmp}/native-${repetition}.stdout"
done
cmp "${d2_tmp}/native-1.stdout" "${d2_tmp}/native-2.stdout"
grep -E '^D2 native PASS: [0-9]+ ' "${d2_tmp}/native-1.stdout" >/dev/null

"${d2_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/d2/header_cpp.cpp" \
  "${d2_library}" "${i1_library}" "${k1_library}" \
  -o "${d2_tmp}/header-cpp"
"${d2_tmp}/header-cpp"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-k1.sh" \
  "${d2_tmp}/sanitized-k1" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" \
  "${d2_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-d2.sh" \
  "${d2_tmp}/sanitized-d2" sanitize-test
"${d2_cc}" "${d2_cflags[@]}" -DET_D2_NATIVE_TESTING \
  -DET_I64_TENSOR_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/d2/test_d2_native.c" \
  "${d2_tmp}/sanitized-d2/libeshkol_transformer_d2_private.a" \
  "${d2_tmp}/sanitized-i1/libeshkol_transformer_i64.a" \
  "${d2_tmp}/sanitized-k1/libeshkol_transformer_k1.a" \
  -o "${d2_tmp}/test-d2-native-sanitized"
ASAN_OPTIONS=detect_leaks="${D2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${d2_tmp}/test-d2-native-sanitized" >/dev/null

(
  cd -- "${PROJECT_ROOT}"
  for repetition in 1 2; do
    PYTHONDONTWRITEBYTECODE=1 python3 -m tests.d2.prepare_public_resources \
      --output "${d2_tmp}/public-resources-${repetition}"
    PYTHONDONTWRITEBYTECODE=1 python3 -m \
      tests.t2.generate_adversarial_fixtures \
      --output-directory "${d2_tmp}/t2-adversarial-${repetition}" >/dev/null
  done
)
diff -ru "${d2_tmp}/public-resources-1" "${d2_tmp}/public-resources-2"
cmp "${d2_tmp}/t2-adversarial-1/alternate-same-vocab.tsv" \
  "${d2_tmp}/t2-adversarial-2/alternate-same-vocab.tsv"

compile_public_d2() {
  local source=$1 label=$2
  local optimize_args=(--optimize 0)
  [[ "${source}" == resource_runtime ]] && optimize_args=()
  mkdir -p "${d2_tmp}/${label}" "${d2_tmp}/cache-${label}"
  if ! env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${d2_tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${d2_cxx}" \
      timeout --foreground --signal=TERM --kill-after=5s "${d2_timeout}s" \
      "${d2_runner}" --strict-types "${optimize_args[@]}" --no-stdlib \
      -I "${PROJECT_ROOT}/lib" -L "${d2_dir}" \
      --lib eshkol_transformer_wave2 \
      "${PROJECT_ROOT}/tests/d2/${source}.esk" \
      -o "${d2_tmp}/${label}/${source}" \
      >"${d2_tmp}/${label}/compile.stdout" \
      2>"${d2_tmp}/${label}/compile.stderr"; then
    sed -n '1,260p' "${d2_tmp}/${label}/compile.stderr" >&2
    die "D2 public ${source} strict AOT compilation failed"
  fi
  [[ -x "${d2_tmp}/${label}/${source}" ]] || \
    die "D2 public ${source} executable is missing"
  ! rg -F 'ERROR:' "${d2_tmp}/${label}/compile.stderr" >/dev/null || \
    die "D2 public ${source} compiler reported ERROR while returning success"
}

for public_source in public_runtime public_errors_runtime resource_runtime; do
  compile_public_d2 "${public_source}" "public-a-${public_source}"
  compile_public_d2 "${public_source}" "public-b-${public_source}"
  cmp "${d2_tmp}/public-a-${public_source}/${public_source}" \
    "${d2_tmp}/public-b-${public_source}/${public_source}"
done

for repetition in a b; do
  resource_index=1
  [[ "${repetition}" == b ]] && resource_index=2
  mkdir -p "${d2_tmp}/public-write-${repetition}"
  if ! "${d2_tmp}/public-${repetition}-public_runtime/public_runtime" \
      "${d2_fixture}" \
      "${d2_tmp}/t2-adversarial-${resource_index}/alternate-same-vocab.tsv" \
      "${d2_tmp}/public-write-${repetition}" \
      "${d2_tmp}/missing-${repetition}" \
      >"${d2_tmp}/public-runtime-${repetition}.stdout" \
      2>"${d2_tmp}/public-runtime-${repetition}.stderr"; then
    sed -n '1,260p' "${d2_tmp}/public-runtime-${repetition}.stdout" >&2
    sed -n '1,260p' "${d2_tmp}/public-runtime-${repetition}.stderr" >&2
    die "D2 public runtime failed"
  fi
  test ! -s "${d2_tmp}/public-runtime-${repetition}.stderr"
  if ! ESHKOL_ARENA_POISON=1 \
      "${d2_tmp}/public-${repetition}-public_errors_runtime/public_errors_runtime" \
      "${d2_fixture}" "${d2_tmp}/public-resources-${resource_index}" \
      >"${d2_tmp}/public-errors-${repetition}.stdout" \
      2>"${d2_tmp}/public-errors-${repetition}.stderr"; then
    sed -n '1,260p' "${d2_tmp}/public-errors-${repetition}.stdout" >&2
    sed -n '1,260p' "${d2_tmp}/public-errors-${repetition}.stderr" >&2
    die "D2 public errors runtime failed"
  fi
  test ! -s "${d2_tmp}/public-errors-${repetition}.stderr"
done
cmp "${d2_tmp}/public-runtime-a.stdout" "${d2_tmp}/public-runtime-b.stdout"
cmp "${d2_tmp}/public-errors-a.stdout" "${d2_tmp}/public-errors-b.stdout"
grep -E '^D2 PUBLIC PASS: [0-9]+ compiled dataset/batch/cursor checks$' \
  "${d2_tmp}/public-runtime-a.stdout" >/dev/null
grep -E '^D2 PUBLIC ERRORS PASS: [0-9]+ compiled rejection/recovery checks$' \
  "${d2_tmp}/public-errors-a.stdout" >/dev/null

d2_runtime_dir="${d2_tmp}/private-view-runtime"
mkdir -p "${d2_runtime_dir}"
d2_runtime_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
  -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
"${d2_cc}" "${d2_runtime_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${d2_runtime_dir}/i2_wave2_native_bridge.o"
"${d2_cc}" "${d2_runtime_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${d2_runtime_dir}/p1_identity.o"
for runtime_source in data_io checkpoint_io kernel_abi t1_i64_shell f32_tensor; do
  "${d2_cc}" "${d2_runtime_cflags[@]}" \
    -c "${PROJECT_ROOT}/native/${runtime_source}.c" \
    -o "${d2_runtime_dir}/${runtime_source}.o"
done
"${d2_cc}" "${d2_runtime_cflags[@]}" -DET_I64_TENSOR_TESTING \
  -c "${PROJECT_ROOT}/native/i64_tensor.c" \
  -o "${d2_runtime_dir}/i64_tensor.o"
"${d2_cc}" "${d2_runtime_cflags[@]}" -DET_D2_NATIVE_TESTING \
  -DET_I64_TENSOR_TESTING -c "${PROJECT_ROOT}/native/d2_native.c" \
  -o "${d2_runtime_dir}/d2_native.o"
"${d2_cc}" "${d2_runtime_cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/d2/view_probe.c" \
  -o "${d2_runtime_dir}/view_probe.o"
ar rcsD "${d2_runtime_dir}/libeshkol_transformer_d2_test_runtime.a" \
  "${d2_runtime_dir}"/*.o

compile_private_view() {
  local label=$1
  mkdir -p "${d2_tmp}/${label}" "${d2_tmp}/cache-${label}"
  if ! env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${d2_tmp}/cache-${label}" \
      ESHKOL_CXX_COMPILER="${d2_cxx}" \
      timeout --foreground --signal=TERM --kill-after=5s "${d2_timeout}s" \
      "${d2_runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t2/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/internal/d2/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" -L "${d2_runtime_dir}" \
      --lib eshkol_transformer_d2_test_runtime \
      "${PROJECT_ROOT}/tests/d2/private_view_runtime.esk" \
      -o "${d2_tmp}/${label}/private-view" \
      >"${d2_tmp}/${label}/compile.stdout" \
      2>"${d2_tmp}/${label}/compile.stderr"; then
    sed -n '1,260p' "${d2_tmp}/${label}/compile.stderr" >&2
    die "D2 private-view strict AOT compilation failed"
  fi
}
compile_private_view private-a
compile_private_view private-b
cmp "${d2_tmp}/private-a/private-view" "${d2_tmp}/private-b/private-view"
for repetition in a b; do
  resource_index=1
  [[ "${repetition}" == b ]] && resource_index=2
  if ! timeout --foreground --signal=TERM --kill-after=5s 180s \
      "${d2_tmp}/private-${repetition}/private-view" "${d2_fixture}" \
      "${d2_tmp}/public-resources-${resource_index}" \
      >"${d2_tmp}/private-view-${repetition}.stdout" \
      2>"${d2_tmp}/private-view-${repetition}.stderr"; then
    sed -n '1,260p' "${d2_tmp}/private-view-${repetition}.stdout" >&2
    sed -n '1,260p' "${d2_tmp}/private-view-${repetition}.stderr" >&2
    die "D2 private-view runtime failed"
  fi
  test ! -s "${d2_tmp}/private-view-${repetition}.stderr"
done
cmp "${d2_tmp}/private-view-a.stdout" "${d2_tmp}/private-view-b.stdout"
grep -E '^D2 PRIVATE VIEW PASS: [0-9]+ compiled content/lifetime checks$' \
  "${d2_tmp}/private-view-a.stdout" >/dev/null

run_resource_probe() {
  local label=$1 directory=$2 expected=$3
  local pid rss=0 rss_max=0 fd_count=0 fd_max=0 started=$SECONDS
  "${d2_tmp}/public-a-resource_runtime/resource_runtime" "${d2_fixture}" \
    "${directory}" consume >"${d2_tmp}/resource-${label}.stdout" &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    if (( SECONDS - started > 300 )); then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      die "D2 ${label} resource probe timed out"
    fi
    if [[ -r "/proc/${pid}/status" ]]; then
      rss="$(awk '/^VmRSS:/ { print $2 }' "/proc/${pid}/status" 2>/dev/null)" || \
        rss=0
      [[ "${rss}" =~ ^[0-9]+$ ]] || rss=0
      (( rss > rss_max )) && rss_max=$rss
      fd_count="$(find "/proc/${pid}/fd" -mindepth 1 -maxdepth 1 \
        2>/dev/null | wc -l)" || fd_count=0
      (( fd_count > fd_max )) && fd_max=$fd_count
    fi
    sleep 0.01
  done
  if ! wait "${pid}"; then
    die "D2 ${label} resource probe failed"
  fi
  grep -Fx "D2 RESOURCE CONSUME PASS: ${expected} batches" \
    "${d2_tmp}/resource-${label}.stdout" >/dev/null
  printf '%s\n' "${rss_max}" >"${d2_tmp}/resource-${label}.rss"
  printf '%s\n' "${fd_max}" >"${d2_tmp}/resource-${label}.fds"
}
run_resource_probe small "${d2_tmp}/public-resources-1/small" 1
run_resource_probe large "${d2_tmp}/public-resources-1/large" 8192
small_rss="$(<"${d2_tmp}/resource-small.rss")"
large_rss="$(<"${d2_tmp}/resource-large.rss")"
[[ "${small_rss}" =~ ^[0-9]+$ && "${large_rss}" =~ ^[0-9]+$ ]] || \
  die "D2 resource RSS measurements are not integer KiB values"
small_fds="$(<"${d2_tmp}/resource-small.fds")"
large_fds="$(<"${d2_tmp}/resource-large.fds")"
rss_delta=$(( large_rss > small_rss ? large_rss - small_rss : small_rss - large_rss ))
(( small_fds <= 8 && large_fds <= 8 )) || \
  die "D2 resource probe exceeded the fixed eight-descriptor process ceiling"
printf 'D2 RESOURCE RSS/FD: small=%s KiB/%s fd large=%s KiB/%s fd delta=%s KiB\n' \
  "${small_rss}" "${small_fds}" "${large_rss}" "${large_fds}" "${rss_delta}"

for horizon in short long; do
  mode=consume-short
  expected_batches=1024
  if [[ "${horizon}" == long ]]; then
    mode=consume
    expected_batches=8192
  fi
  ESHKOL_ARENA_REPORT=1 \
    timeout --foreground --signal=TERM --kill-after=5s 300s \
    "${d2_tmp}/public-a-resource_runtime/resource_runtime" "${d2_fixture}" \
    "${d2_tmp}/public-resources-1/large" "${mode}" \
    >"${d2_tmp}/arena-${horizon}.stdout" \
    2>"${d2_tmp}/arena-${horizon}.stderr"
  grep -Fx "D2 RESOURCE CONSUME PASS: ${expected_batches} batches" \
    "${d2_tmp}/arena-${horizon}.stdout" >/dev/null
  arena_bytes="$(awk -F= '/global_total_allocated_bytes=/{print $2}' \
    "${d2_tmp}/arena-${horizon}.stderr" | tail -1)"
  [[ "${arena_bytes}" =~ ^[0-9]+$ ]] || \
    die "D2 ${horizon} run omitted the exact Eshkol arena retention counter"
  printf '%s\n' "${arena_bytes}" >"${d2_tmp}/arena-${horizon}.bytes"
done
short_arena="$(<"${d2_tmp}/arena-short.bytes")"
long_arena="$(<"${d2_tmp}/arena-long.bytes")"
[[ "${short_arena}" == "${long_arena}" ]] || \
  die "D2 optimized retained arena bytes grew from 1024 to 8192 batches"
printf 'D2 RESOURCE ARENA PASS: 1024=%s bytes 8192=%s bytes slope=0\n' \
  "${short_arena}" "${long_arena}"

declare -A topology_open topology_packed
for topology in one-shards many-shard; do
  for probe in open packed; do
    timeout --foreground --signal=TERM --kill-after=5s 300s \
      "${d2_tmp}/public-a-resource_runtime/resource_runtime" "${d2_fixture}" \
      "${d2_tmp}/public-resources-1/${topology}" "arena-${probe}" \
      >"${d2_tmp}/arena-${topology}-${probe}.stdout" \
      2>"${d2_tmp}/arena-${topology}-${probe}.stderr"
    test ! -s "${d2_tmp}/arena-${topology}-${probe}.stderr"
    measured="$(awk '/^D2 RESOURCE ARENA (OPEN|PACKED): [0-9]+ bytes$/ {print $5}' \
      "${d2_tmp}/arena-${topology}-${probe}.stdout")"
    [[ "${measured}" =~ ^[0-9]+$ ]] || \
      die "D2 ${topology} arena-${probe} omitted an exact byte count"
    if [[ "${probe}" == open ]]; then
      topology_open["${topology}"]="${measured}"
    else
      topology_packed["${topology}"]="${measured}"
    fi
  done
done
one_open="${topology_open[one-shards]}"
many_open="${topology_open[many-shard]}"
open_retained_delta=$(( many_open > one_open \
  ? many_open - one_open : one_open - many_open ))
printf 'D2 RESOURCE SHARD ARENA MEASURED: open one=%s/many=%s bytes retained-delta=%s packed one=%s/many=%s bytes\n' \
  "${topology_open[one-shards]}" "${topology_open[many-shard]}" \
  "${open_retained_delta}" "${topology_packed[one-shards]}" \
  "${topology_packed[many-shard]}"
(( open_retained_delta <= 83365 )) || \
  die "D2 equal-token open retention exceeded one admitted working payload"
[[ "${topology_packed[one-shards]}" == "${topology_packed[many-shard]}" ]] || \
  die "D2 equal-token packed traversal retained shard-topology-dependent arena bytes"

timeout --foreground --signal=TERM --kill-after=5s 180s \
  "${d2_tmp}/public-a-resource_runtime/resource_runtime" "${d2_fixture}" \
  "${d2_tmp}/public-resources-1/small" reopen \
  >"${d2_tmp}/resource-reopen.stdout"
grep -Fx 'D2 RESOURCE REOPEN PASS: 256 datasets' \
  "${d2_tmp}/resource-reopen.stdout" >/dev/null
timeout --foreground --signal=TERM --kill-after=5s 180s \
  "${d2_tmp}/public-a-resource_runtime/resource_runtime" "${d2_fixture}" \
  "${d2_tmp}/public-resources-1/small" accessors \
  >"${d2_tmp}/resource-accessors.stdout"
grep -Fx 'D2 RESOURCE ACCESSORS PASS: live=0 bytes released=0 bytes' \
  "${d2_tmp}/resource-accessors.stdout" >/dev/null

wave2_evidence="${d2_dir}/d2_wave2.o.evidence"
[[ "$(wc -l <"${wave2_evidence}/global-defined.txt")" == 58 ]] || \
  die "D2 aggregate must have exactly 58 globals"
[[ "$(wc -l <"${wave2_evidence}/package-exports.txt")" == 52 ]] || \
  die "D2 aggregate must have exactly 52 exports"
[[ "$(ar t "${d2_wave2_library}")" == "d2_wave2.o" ]] || \
  die "D2 aggregate archive must contain only d2_wave2.o"
sed -e 's/^[^:]*://' -e 's/\\//g' "${wave2_evidence}/private.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${d2_tmp}/wave2-source-closure.txt"
cmp "${PROJECT_ROOT}/native/d2_wave2_source_closure.txt" \
  "${d2_tmp}/wave2-source-closure.txt" || \
  die "D2 aggregate trusted source closure drifted"
mapfile -t d2_production_sources \
  <"${PROJECT_ROOT}/native/d2_wave2_source_closure.txt"
(( ${#d2_production_sources[@]} == 18 )) || \
  die "D2 production source closure must contain exactly 18 files"
for d2_source_index in "${!d2_production_sources[@]}"; do
  d2_production_sources[${d2_source_index}]="${PROJECT_ROOT}/${d2_production_sources[${d2_source_index}]}"
done

for evidence_name in global-defined.txt package-exports.txt undefined.txt \
    expected-undefined.txt public-strings.txt readelf-symbols.txt nm.txt \
    strings.txt private.d link.map allowlist-provenance.tsv; do
  [[ -s "${wave2_evidence}/${evidence_name}" ]] || \
    die "D2 aggregate evidence omits ${evidence_name}"
done
nm -s "${d2_wave2_library}" | \
  awk '/^Archive index:$/ { active = 1; next }
       active && /^$/ { active = 0; next }
       active {
         if (NF != 3 || $2 != "in" || $3 != "d2_wave2.o") exit 2
         print $1
       }' | LC_ALL=C sort >"${d2_tmp}/wave2-archive-index.txt" || \
  die "D2 aggregate archive index has an unexpected member or shape"
cmp "${wave2_evidence}/global-defined.txt" \
  "${d2_tmp}/wave2-archive-index.txt" || \
  die "D2 aggregate archive index differs from the exact 58-symbol manifest"
if grep -E 'et_(e1b_private|d2_|i2_|f32_|p1_private_)|d2-(native|dataset|batch|core)' \
    "${d2_tmp}/wave2-archive-index.txt" >/dev/null; then
  die "D2 aggregate archive index exposes a localized authority"
fi

mkdir -p "${d2_tmp}/shadow/transformer"
printf '(error "hostile D2 private root loaded")\n' \
  >"${d2_tmp}/shadow/d2_wave2_root.esk"
printf '(error "hostile D2 dataset implementation loaded")\n' \
  >"${d2_tmp}/shadow/d2_dataset.esk"
ESHKOL_PATH="${d2_tmp}/shadow" ESHKOL_LIB_DIR="${d2_tmp}/shadow" \
D2_COMPILER_TIMEOUT_SECONDS="${d2_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-d2.sh" \
    "${d2_tmp}/hostile-build"
cmp "${d2_dir}/d2_wave2.o" "${d2_tmp}/hostile-build/d2_wave2.o"
for deterministic_evidence in global-defined.txt package-exports.txt \
    undefined.txt expected-undefined.txt public-strings.txt private.d link.map; do
  cmp "${wave2_evidence}/${deterministic_evidence}" \
    "${d2_tmp}/hostile-build/d2_wave2.o.evidence/${deterministic_evidence}"
done
if grep -F "${d2_tmp}/shadow" \
    "${d2_tmp}/hostile-build/d2_wave2.o.evidence/private.d" >/dev/null; then
  die "D2 build admitted a hostile module path"
fi

reject_d2_builder_input() {
  local label=$1 expected=$2
  shift 2
  if E1B_COMPILER_TIMEOUT_SECONDS="${d2_timeout}" \
      "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" "$@" \
      >"${d2_tmp}/${label}.stdout" 2>"${d2_tmp}/${label}.stderr"; then
    die "D2 builder admitted ${label}"
  fi
  grep -F "${expected}" "${d2_tmp}/${label}.stderr" >/dev/null || \
    die "D2 ${label} rejection reported the wrong reason"
}
cp "${PROJECT_ROOT}/native/d2_wave2_root.esk" \
  "${d2_tmp}/copied-d2-root.esk"
for rejected_root in "${d2_tmp}/copied-d2-root.esk" \
    "${d2_dir}/d2_wave2.o"; do
  label="rejected-root-$(basename -- "${rejected_root}")"
  reject_d2_builder_input "${label}" \
    'repository package components require their exact repository-owned private root' \
    "${rejected_root}" \
    "${PROJECT_ROOT}/native/d2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/d2_wave2_private_renames.txt" \
    "${PROJECT_ROOT}/native/d2_wave2_public_exports.txt" \
    "${d2_tmp}/${label}.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t2/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/internal/d2/lib" \
    "${PROJECT_ROOT}/src"
  [[ ! -e "${d2_tmp}/${label}.o" && \
     ! -e "${d2_tmp}/${label}.o.evidence" ]] || \
    die "rejected D2 builder input published an artifact"
done
for copied_component in bridge renames exports; do
  case "${copied_component}" in
    bridge)
      source_component="${PROJECT_ROOT}/native/d2_wave2_package_bridge.c"
      expected_component='D2 aggregate policy requires the exact repository bridge'
      ;;
    renames)
      source_component="${PROJECT_ROOT}/native/d2_wave2_private_renames.txt"
      expected_component='D2 aggregate policy requires the exact repository rename map'
      ;;
    exports)
      source_component="${PROJECT_ROOT}/native/d2_wave2_public_exports.txt"
      expected_component='D2 aggregate policy requires the exact repository export list'
      ;;
  esac
  copied_path="${d2_tmp}/copied-$(basename -- "${source_component}")"
  cp "${source_component}" "${copied_path}"
  bridge="${PROJECT_ROOT}/native/d2_wave2_package_bridge.c"
  renames="${PROJECT_ROOT}/native/d2_wave2_private_renames.txt"
  exports="${PROJECT_ROOT}/native/d2_wave2_public_exports.txt"
  case "${copied_component}" in
    bridge) bridge="${copied_path}" ;;
    renames) renames="${copied_path}" ;;
    exports) exports="${copied_path}" ;;
  esac
  reject_d2_builder_input "rejected-copied-${copied_component}" \
    "${expected_component}" \
    "${PROJECT_ROOT}/native/d2_wave2_root.esk" \
    "${bridge}" "${renames}" "${exports}" \
    "${d2_tmp}/rejected-copied-${copied_component}.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t2/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/internal/d2/lib" \
    "${PROJECT_ROOT}/src"
  [[ ! -e "${d2_tmp}/rejected-copied-${copied_component}.o" && \
     ! -e "${d2_tmp}/rejected-copied-${copied_component}.o.evidence" ]] || \
    die "rejected copied D2 ${copied_component} published an artifact"
done
reject_d2_builder_input hostile-include \
  'D2 aggregate policy requires exact ordered trusted include roots' \
  "${PROJECT_ROOT}/native/d2_wave2_root.esk" \
  "${PROJECT_ROOT}/native/d2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/d2_wave2_private_renames.txt" \
  "${PROJECT_ROOT}/native/d2_wave2_public_exports.txt" \
  "${d2_tmp}/hostile-include.o" \
  "${PROJECT_ROOT}/internal/p1/lib" \
  "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t2/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" \
  "${PROJECT_ROOT}/internal/d2/lib" \
  "${PROJECT_ROOT}/src" "${d2_tmp}/shadow"
[[ ! -e "${d2_tmp}/hostile-include.o" && \
   ! -e "${d2_tmp}/hostile-include.o.evidence" ]] || \
  die "rejected hostile D2 include published an artifact"

if "${d2_cc}" -r -Wl,--whole-archive \
    "${d2_wave2_library}" "${d2_wave2_library}" \
    -Wl,--no-whole-archive -o "${d2_tmp}/duplicate-authority.o" \
    >"${d2_tmp}/duplicate-authority.stdout" \
    2>"${d2_tmp}/duplicate-authority.stderr"; then
  die "linker admitted duplicate D2 aggregate authority"
fi
grep -E 'multiple definition.*et_e1b_error_(predicate|category)_v1' \
  "${d2_tmp}/duplicate-authority.stderr" >/dev/null || \
  die "duplicate D2 authority rejection did not identify E1 ownership"
for private_binding in d2-token-dataset-open d2-token-tensor-with-view-internal \
    d2-core-normalize-config d1-sha256-prefix; do
  if env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${d2_tmp}/cache-private-${private_binding}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${d2_cxx}" \
      "${d2_runner}" --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "${d2_dir}" --lib eshkol_transformer_wave2 \
      -e "(begin (require transformer.data) ${private_binding})" \
      >"${d2_tmp}/private-${private_binding}.stdout" \
      2>"${d2_tmp}/private-${private_binding}.stderr"; then
    die "D2 aggregate exposed private binding ${private_binding}"
  fi
done

if rg -ni 'python|pytorch|torch|libpython|(^|[^[:alnum:]_])Py_' \
    "${d2_production_sources[@]}" \
    "${PROJECT_ROOT}/lib/transformer/data.esk" \
    "${PROJECT_ROOT}/native/d2_native.c" \
    "${PROJECT_ROOT}/native/d2_native.h" >/dev/null; then
  die "D2 production candidate references a Python runtime"
fi
for delivered in "${d2_dir}/d2_native.o" "${d2_dir}/d2_wave2.o" \
    "${d2_tmp}/a/d2-semantic" "${d2_tmp}/b/d2-semantic" \
    "${d2_tmp}/public-a-public_runtime/public_runtime" \
    "${d2_tmp}/public-b-public_runtime/public_runtime" \
    "${d2_tmp}/public-a-public_errors_runtime/public_errors_runtime" \
    "${d2_tmp}/public-b-public_errors_runtime/public_errors_runtime" \
    "${d2_tmp}/public-a-resource_runtime/resource_runtime" \
    "${d2_tmp}/public-b-resource_runtime/resource_runtime" \
    "${d2_tmp}/private-a/private-view" \
    "${d2_tmp}/private-b/private-view"; do
  if strings -a "${delivered}" | \
      grep -Ei 'python|pytorch|torch|libpython|(^|[^[:alnum:]_])Py_' >/dev/null || \
      nm -a "${delivered}" | \
        grep -Ei 'python|pytorch|torch|libpython|(^|[[:space:]])Py_' >/dev/null || \
      ldd "${delivered}" 2>/dev/null | \
        grep -Ei 'python|pytorch|torch|libpython' >/dev/null; then
    die "D2 delivered candidate contains a development-oracle dependency"
  fi
done

printf 'D2 PASS: carrier-neutral shift/shuffle/cursor semantics, frozen Q0 fixture, private carrier lifetime, determinism, resource, sanitizer, and isolation gates\n'
