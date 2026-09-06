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
  mkdir -p "${d2_tmp}/${label}" "${d2_tmp}/cache-${label}"
  if ! env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${d2_tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${d2_cxx}" \
      timeout --foreground --signal=TERM --kill-after=5s "${d2_timeout}s" \
      "${d2_runner}" --strict-types --optimize 0 --no-stdlib \
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
  "${d2_tmp}/public-${repetition}-public_runtime/public_runtime" \
    "${d2_fixture}" \
    "${d2_tmp}/t2-adversarial-${resource_index}/alternate-same-vocab.tsv" \
    "${d2_tmp}/public-write-${repetition}" \
    "${d2_tmp}/missing-${repetition}" \
    >"${d2_tmp}/public-runtime-${repetition}.stdout"
  "${d2_tmp}/public-${repetition}-public_errors_runtime/public_errors_runtime" \
    "${d2_fixture}" "${d2_tmp}/public-resources-${resource_index}" \
    >"${d2_tmp}/public-errors-${repetition}.stdout"
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
printf 'D2 RESOURCE MEASURED: small=%s KiB/%s fd large=%s KiB/%s fd delta=%s KiB\n' \
  "${small_rss}" "${small_fds}" "${large_rss}" "${large_fds}" "${rss_delta}"
(( small_rss <= 262144 && large_rss <= 262144 )) || \
  die "D2 resource probe exceeded the fixed 256 MiB RSS test ceiling"
(( rss_delta <= 65536 )) || \
  die "D2 RSS changed by more than 64 MiB when corpus grew to 8193 tokens"
(( small_fds <= 8 && large_fds <= 8 )) || \
  die "D2 resource probe exceeded the fixed eight-descriptor process ceiling"
printf 'D2 RESOURCE BOUNDS PASS: small=%s KiB/%s fd large=%s KiB/%s fd delta=%s KiB\n' \
  "${small_rss}" "${small_fds}" "${large_rss}" "${large_fds}" "${rss_delta}"
timeout --foreground --signal=TERM --kill-after=5s 180s \
  "${d2_tmp}/public-a-resource_runtime/resource_runtime" "${d2_fixture}" \
  "${d2_tmp}/public-resources-1/small" reopen \
  >"${d2_tmp}/resource-reopen.stdout"
grep -Fx 'D2 RESOURCE REOPEN PASS: 256 datasets' \
  "${d2_tmp}/resource-reopen.stdout" >/dev/null

wave2_evidence="${d2_dir}/d2_wave2.o.evidence"
[[ "$(wc -l <"${wave2_evidence}/global-defined.txt")" == 58 ]] || \
  die "D2 aggregate must have exactly 58 globals"
[[ "$(wc -l <"${wave2_evidence}/package-exports.txt")" == 52 ]] || \
  die "D2 aggregate must have exactly 52 exports"
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

if rg -n -i '\b(import|from) (torch|pytorch|python)|python\.h|py_' \
    "${PROJECT_ROOT}/internal/d2" \
    "${PROJECT_ROOT}/native/d2_native.c" \
    "${PROJECT_ROOT}/native/d2_native.h" >/dev/null; then
  die "D2 production candidate references a Python runtime"
fi
for delivered in "${d2_dir}/d2_native.o" "${d2_dir}/d2_wave2.o" \
    "${d2_tmp}/a/d2-semantic"; do
  if strings -a "${delivered}" | \
      grep -E 'tests/d2/(reference|generate)|torch' >/dev/null || \
      nm -a "${delivered}" | grep -E '(^|[[:space:]])Py_|libpython' >/dev/null || \
      ldd "${delivered}" 2>/dev/null | grep -Ei 'python|torch' >/dev/null; then
    die "D2 delivered candidate contains a development-oracle dependency"
  fi
done

printf 'D2 PASS: carrier-neutral shift/shuffle/cursor semantics, frozen Q0 fixture, private carrier lifetime, determinism, resource, sanitizer, and isolation gates\n'
