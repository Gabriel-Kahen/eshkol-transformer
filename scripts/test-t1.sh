#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp find grep nm ps python3 sleep strings timeout; do
  require_command "${command}"
done

t1_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-t1.XXXXXX")"
trap 'rm -rf -- "${t1_tmp}"' EXIT
t1_fixture="${PROJECT_ROOT}/tests/t1/fixtures/byte_tokenizer_v1.tsv"
t1_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
t1_cc="$(tsv_value "${t1_provenance}" cc_path)"
t1_cxx="$(tsv_value "${t1_provenance}" cxx_path)"

t1_native_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -DET_T1_I64_SHELL_TESTING -DET_I64_TENSOR_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
t1_native_sources=(
  "${PROJECT_ROOT}/tests/t1/test_i64_shell.c"
  "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/i64_tensor.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
)
"${t1_cc}" "${t1_native_flags[@]}" -fstack-protector-all \
  "${t1_native_sources[@]}" -o "${t1_tmp}/i64-shell"
for repetition in 1 2; do
  "${t1_tmp}/i64-shell" >"${t1_tmp}/i64-shell-${repetition}.stdout"
done
cmp "${t1_tmp}/i64-shell-1.stdout" "${t1_tmp}/i64-shell-2.stdout"
grep -Fx 'T1 I64 SHELL PASS: 110 admission, exact-i64, lifetime, and failure checks' \
  "${t1_tmp}/i64-shell-1.stdout" >/dev/null

"${t1_cc}" "${t1_native_flags[@]}" -fno-omit-frame-pointer \
  -fsanitize=address,undefined "${t1_native_sources[@]}" \
  -o "${t1_tmp}/i64-shell-sanitized"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${t1_tmp}/i64-shell-sanitized" >/dev/null

for source in t1_i64_shell i64_tensor kernel_abi; do
  "${t1_cc}" "${t1_native_flags[@]}" \
    -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${t1_tmp}/${source}.o"
done
"${t1_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -DET_T1_I64_SHELL_TESTING -DET_I64_TENSOR_TESTING \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -x c++ -c "${PROJECT_ROOT}/tests/t1/test_i64_shell.c" \
  -o "${t1_tmp}/i64-shell-cxx.o"
"${t1_cxx}" "${t1_tmp}/i64-shell-cxx.o" \
  "${t1_tmp}/t1_i64_shell.o" "${t1_tmp}/i64_tensor.o" \
  "${t1_tmp}/kernel_abi.o" -o "${t1_tmp}/i64-shell-cxx"
"${t1_tmp}/i64-shell-cxx" >/dev/null

(
  cd -- "${PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.t1.test_reference
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
    tests.q0.test_python_isolation
  PYTHONDONTWRITEBYTECODE=1 python3 -m tests.t1.generate_fixture \
    --output "${t1_tmp}/fixture-1.tsv"
  PYTHONDONTWRITEBYTECODE=1 python3 -m tests.t1.generate_fixture \
    --output "${t1_tmp}/fixture-2.tsv"
)
cmp "${t1_tmp}/fixture-1.tsv" "${t1_tmp}/fixture-2.tsv"
cmp "${t1_tmp}/fixture-1.tsv" "${t1_fixture}"

E1B_COMPILER_TIMEOUT_SECONDS="${T1_COMPILER_TIMEOUT_SECONDS:-300}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-t1.sh"
t1_archive="$(project_build_dir)/t1/libeshkol_transformer_wave1.a"
t1_object="$(project_build_dir)/t1/wave1.o"
[[ -r "${t1_archive}" && -r "${t1_object}" ]] || \
  die "T1 Wave 1 aggregate was not published"
[[ "$(nm -g --defined-only --format=posix "${t1_object}" | wc -l)" == 47 ]] || \
  die "T1 Wave 1 aggregate does not expose exactly 47 globals"
if nm -a "${t1_object}" | \
    grep -E 'et_t1_test_io_|T1_SAVE_FAILPOINT_TEST_ONLY' >/dev/null; then
  die "production T1 aggregate contains a test-only save failpoint"
fi
if strings -a "${t1_object}" | \
    grep -E 'et_t1_test_io_|T1_SAVE_FAILPOINT_TEST_ONLY' >/dev/null; then
  die "production T1 aggregate strings contain a test-only save failpoint"
fi
if grep -ER 'et_t1_test_io_|t1_save_failpoints|T1_SAVE_FAILPOINT_TEST_ONLY' \
    "${PROJECT_ROOT}/lib/transformer/tokenizer.esk" \
    "${PROJECT_ROOT}/lib/transformer/persistence.esk" \
    "${PROJECT_ROOT}/internal/t1" \
    "${PROJECT_ROOT}/native/t1_wave1_root.esk" \
    "${PROJECT_ROOT}/native/t1_wave1_package_bridge.c" \
    "${PROJECT_ROOT}/native/t1_i64_shell.c" \
    "${PROJECT_ROOT}/native/t1_i64_shell.h" \
    "${PROJECT_ROOT}/scripts/build-t1.sh" \
    "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" >/dev/null; then
  die "production T1 source or builder references a test-only save failpoint"
fi

"${t1_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -fstack-protector-all -fno-common \
  -c "${PROJECT_ROOT}/tests/t1/t1_save_failpoints.c" \
  -o "${t1_tmp}/t1-save-failpoints.o"
ar rcsD "${t1_tmp}/libeshkol_transformer_t1_save_failpoints.a" \
  "${t1_tmp}/t1-save-failpoints.o"

t1_runner="$(eshkol_build_dir)/eshkol-run"
ESHKOL_CXX_COMPILER="${t1_cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 240s \
  "${t1_runner}" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" \
  -L "$(project_build_dir)/t1" --lib eshkol_transformer_wave1 \
  -L "${t1_tmp}" --lib eshkol_transformer_t1_save_failpoints \
  "${PROJECT_ROOT}/tests/t1/save_failpoint_runtime.esk" \
  -o "${t1_tmp}/save-failpoint-runtime"
t1_save_failpoint_modes=(
  short eintr zero sync-temp close-temp publish
  sync-directory close-directory cleanup
)
for repetition in 1 2; do
  for mode in "${t1_save_failpoint_modes[@]}"; do
    case_dir="${t1_tmp}/save-failpoint-${repetition}-${mode}"
    mkdir -p "${case_dir}"
    target="${case_dir}/tokenizer.tsv"
    timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${t1_tmp}/save-failpoint-runtime" \
      "${t1_fixture}" "${target}" "${mode}" \
      >"${case_dir}/stdout" 2>"${case_dir}/stderr"
    grep -Fx "T1 SAVE FAILPOINT PASS: ${mode}" \
      "${case_dir}/stdout" >/dev/null
    [[ ! -s "${case_dir}/stderr" ]] || \
      die "T1 save failpoint ${mode} emitted stderr"
    temp_count="$({ find "${case_dir}" -maxdepth 1 -type f \
      -name '.et-c1-*.tmp' -print; } | wc -l)"
    if [[ "${mode}" == cleanup ]]; then
      [[ "${temp_count}" == 1 ]] || \
        die "T1 cleanup failure did not retain exactly one unpublished temp"
    else
      [[ "${temp_count}" == 0 ]] || \
        die "T1 save failpoint ${mode} left an unpublished temp"
    fi
  done
done

for repetition in 1 2; do
  ESHKOL_CXX_COMPILER="${t1_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${t1_runner}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -L "$(project_build_dir)/t1" --lib eshkol_transformer_wave1 \
    "${PROJECT_ROOT}/tests/t1/public_runtime.esk" \
    -o "${t1_tmp}/public-runtime-${repetition}"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${t1_tmp}/public-runtime-${repetition}" \
    "${t1_fixture}" "${t1_tmp}/saved-${repetition}.tsv" \
    >"${t1_tmp}/public-runtime-${repetition}.stdout"
  grep -Fx 'T1 PUBLIC PASS: 23 aggregate tokenizer checks' \
    "${t1_tmp}/public-runtime-${repetition}.stdout" >/dev/null
  cmp "${t1_fixture}" "${t1_tmp}/saved-${repetition}.tsv"
done
cmp "${t1_tmp}/public-runtime-1.stdout" \
  "${t1_tmp}/public-runtime-2.stdout"
for production_aot in "${t1_tmp}/public-runtime-1" \
    "${t1_tmp}/public-runtime-2"; do
  if nm -a "${production_aot}" | \
      grep -E 'et_t1_test_io_|T1_SAVE_FAILPOINT_TEST_ONLY' >/dev/null; then
    die "production T1 AOT executable defines a test-only save failpoint"
  fi
  if strings -a "${production_aot}" | \
      grep -E 'et_t1_test_io_|T1_SAVE_FAILPOINT_TEST_ONLY' >/dev/null; then
    die "production T1 AOT strings contain a test-only save failpoint"
  fi
done

for fixture_set in adversarial-1 adversarial-2; do
  (
    cd -- "${PROJECT_ROOT}"
    PYTHONDONTWRITEBYTECODE=1 python3 -m \
      tests.t1.generate_adversarial_fixtures \
      --output-directory "${t1_tmp}/${fixture_set}"
  ) >"${t1_tmp}/${fixture_set}.stdout"
done
diff -ru "${t1_tmp}/adversarial-1" "${t1_tmp}/adversarial-2"
cmp "${t1_tmp}/adversarial-1.stdout" "${t1_tmp}/adversarial-2.stdout"
grep -E '^T1 ADVERSARIAL FIXTURES PASS: 32 deterministic artifacts$' \
  "${t1_tmp}/adversarial-1.stdout" >/dev/null

(
  cd -- "${PROJECT_ROOT}"
  timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${t1_runner}" --strict-types --no-stdlib -I "${PROJECT_ROOT}" \
    "${PROJECT_ROOT}/tests/t1/core_adversarial.esk" \
    -o "${t1_tmp}/core-adversarial"
)
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${t1_tmp}/core-adversarial" >"${t1_tmp}/core-adversarial.stdout"
grep -E '^T1 CORE ADVERSARIAL PASS: 9 build-only semantic checks$' \
  "${t1_tmp}/core-adversarial.stdout" >/dev/null

t1_adversarial_paths=(
  valid-strict.tsv valid-overlap.tsv valid-optional.tsv
  corrupt-body.tsv truncated.tsv trailing.tsv bad-checksum.tsv
  major.tsv version.tsv required-feature.tsv optional-v1.tsv
  reserved-limit.tsv algorithm.tsv payload-overflow.tsv overlong-line.tsv
  non-ascii.tsv bad-special-name.tsv special-collision.tsv special-order.tsv
  unknown-insertion.tsv error-insertion.tsv decode-policy.tsv normalization.tsv
  payload-size-mismatch.tsv insertion-order.tsv unknown-suffix.tsv
  duplicate-optional.tsv special-count-limit.tsv prefix-count-limit.tsv
  name-limit.tsv optional-value-limit.tsv file-limit.tsv
)
for repetition in 1 2; do
  ESHKOL_CXX_COMPILER="${t1_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${t1_runner}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -L "$(project_build_dir)/t1" --lib eshkol_transformer_wave1 \
    "${PROJECT_ROOT}/tests/t1/adversarial_runtime.esk" \
    -o "${t1_tmp}/adversarial-runtime-${repetition}"
  t1_adversarial_arguments=()
  for fixture_name in "${t1_adversarial_paths[@]}"; do
    t1_adversarial_arguments+=(
      "${t1_tmp}/adversarial-${repetition}/${fixture_name}"
    )
  done
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${t1_tmp}/adversarial-runtime-${repetition}" \
    "${t1_adversarial_arguments[@]}" \
    >"${t1_tmp}/adversarial-runtime-${repetition}.stdout"
  grep -E '^T1 ADVERSARIAL PUBLIC PASS: 61 delivered-runtime checks$' \
    "${t1_tmp}/adversarial-runtime-${repetition}.stdout" >/dev/null
done
cmp "${t1_tmp}/adversarial-runtime-1.stdout" \
  "${t1_tmp}/adversarial-runtime-2.stdout"

for fixture_set in limits-1 limits-2; do
  (
    cd -- "${PROJECT_ROOT}"
    PYTHONDONTWRITEBYTECODE=1 python3 -m tests.t1.generate_limit_fixtures \
      --output-directory "${t1_tmp}/${fixture_set}"
  ) >"${t1_tmp}/${fixture_set}.stdout"
done
diff -ru "${t1_tmp}/limits-1" "${t1_tmp}/limits-2"
cmp "${t1_tmp}/limits-1.stdout" "${t1_tmp}/limits-2.stdout"
grep -Fx \
  'T1 LIMIT FIXTURES PASS: 6 deterministic artifacts; exact-max 472645 bytes; optional-max 98891 bytes' \
  "${t1_tmp}/limits-1.stdout" >/dev/null

run_bounded_runtime() {
  local label="$1" maximum_seconds="$2" maximum_rss="$3"
  local stdout="$4" stderr="$5"
  shift 5
  local process_id state rss status timed_out=0 max_rss=0
  local start_seconds="${SECONDS}" grace complete=0
  "$@" >"${stdout}" 2>"${stderr}" &
  process_id=$!
  while true; do
    if ! kill -0 "${process_id}" 2>/dev/null; then
      break
    fi
    state=
    rss=
    read -r state rss \
      < <(ps -o stat=,rss= -p "${process_id}" 2>/dev/null || true) || true
    if [[ "${state:-}" == Z* ]]; then
      break
    fi
    if [[ "${rss:-}" =~ ^[0-9]+$ && "${rss}" -gt "${max_rss}" ]]; then
      max_rss="${rss}"
    fi
    if (( SECONDS - start_seconds >= maximum_seconds )); then
      timed_out=1
      break
    fi
    sleep 0.05
  done
  if [[ "${timed_out}" == 1 ]]; then
    kill -TERM "${process_id}" 2>/dev/null || true
    for ((grace = 0; grace < 100; grace++)); do
      if ! kill -0 "${process_id}" 2>/dev/null; then
        complete=1
        break
      fi
      state="$(ps -o stat= -p "${process_id}" 2>/dev/null || true)"
      if [[ "${state:-}" == Z* ]]; then
        complete=1
        break
      fi
      sleep 0.05
    done
    if [[ "${complete}" == 0 ]]; then
      kill -KILL "${process_id}" 2>/dev/null || true
    fi
  fi
  if wait "${process_id}"; then status=0; else status=$?; fi
  [[ "${timed_out}" == 0 ]] || \
    die "${label} exceeded ${maximum_seconds} seconds"
  [[ "${status}" == 0 ]] || die "${label} failed: ${status}"
  [[ "${max_rss}" -gt 0 && "${max_rss}" -le "${maximum_rss}" ]] || \
    die "${label} exceeded ${maximum_rss} KiB RSS: ${max_rss} KiB"
  t1_measured_rss="${max_rss}"
}

for repetition in 1 2; do
  ESHKOL_CXX_COMPILER="${t1_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${t1_runner}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -L "$(project_build_dir)/t1" --lib eshkol_transformer_wave1 \
    "${PROJECT_ROOT}/tests/t1/limit_runtime.esk" \
    -o "${t1_tmp}/limit-runtime-${repetition}"
  run_bounded_runtime 'T1 exact-max runtime' 60 524288 \
    "${t1_tmp}/limit-runtime-${repetition}.stdout" \
    "${t1_tmp}/limit-runtime-${repetition}.stderr" \
    "${t1_tmp}/limit-runtime-${repetition}" \
    "${t1_tmp}/limits-${repetition}/exact-max.tsv" \
    "${t1_tmp}/limits-${repetition}/specials-one-over.tsv" \
    "${t1_tmp}/limits-${repetition}/prefix-one-over.tsv" \
    "${t1_tmp}/limits-${repetition}/suffix-one-over.tsv" \
    "${t1_tmp}/limits-${repetition}/optional-max.tsv" \
    "${t1_tmp}/limits-${repetition}/optional-one-over.tsv" \
    "${t1_tmp}/limit-saved-${repetition}.tsv" \
    "${t1_tmp}/limit-optional-saved-${repetition}.tsv"
  if grep -F 'Heap usage at' \
      "${t1_tmp}/limit-runtime-${repetition}.stderr" >/dev/null; then
    die "T1 exact-max runtime emitted a heap-pressure warning"
  fi
  printf 'T1 combined-limit measured max RSS: %s KiB\n' "${t1_measured_rss}"
  grep -Fx 'T1 LIMIT PUBLIC PASS: 12 exact/one-over runtime checks' \
    "${t1_tmp}/limit-runtime-${repetition}.stdout" >/dev/null
  cmp "${t1_tmp}/limits-${repetition}/exact-max.tsv" \
    "${t1_tmp}/limit-saved-${repetition}.tsv"
  cmp "${t1_tmp}/limits-${repetition}/optional-max.tsv" \
    "${t1_tmp}/limit-optional-saved-${repetition}.tsv"
done
cmp "${t1_tmp}/limit-runtime-1.stdout" \
  "${t1_tmp}/limit-runtime-2.stdout"

for repetition in 1 2; do
  ESHKOL_CXX_COMPILER="${t1_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${t1_runner}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -L "$(project_build_dir)/t1" --lib eshkol_transformer_wave1 \
    "${PROJECT_ROOT}/tests/t1/registry_lifetime.esk" \
    -o "${t1_tmp}/registry-lifetime-${repetition}"
  run_bounded_runtime 'T1 registry baseline runtime' 30 262144 \
    "${t1_tmp}/registry-baseline-${repetition}.stdout" \
    "${t1_tmp}/registry-baseline-${repetition}.stderr" \
    "${t1_tmp}/registry-lifetime-${repetition}" "${t1_fixture}" baseline
  t1_registry_baseline_rss="${t1_measured_rss}"
  grep -Fx \
    'T1 REGISTRY LIFETIME PASS: 4 oldest-identity checks after 16 growth cycles' \
    "${t1_tmp}/registry-baseline-${repetition}.stdout" >/dev/null
  run_bounded_runtime 'T1 registry growth runtime' 30 262144 \
    "${t1_tmp}/registry-growth-${repetition}.stdout" \
    "${t1_tmp}/registry-growth-${repetition}.stderr" \
    "${t1_tmp}/registry-lifetime-${repetition}" "${t1_fixture}" growth
  t1_registry_growth_rss="${t1_measured_rss}"
  grep -Fx \
    'T1 REGISTRY LIFETIME PASS: 4 oldest-identity checks after 128 growth cycles' \
    "${t1_tmp}/registry-growth-${repetition}.stdout" >/dev/null
  [[ "${t1_registry_growth_rss}" -gt "${t1_registry_baseline_rss}" ]] || \
    die "T1 retained registry growth did not increase measured RSS"
  printf 'T1 registry retained RSS: baseline=%s KiB growth=%s KiB delta=%s KiB\n' \
    "${t1_registry_baseline_rss}" "${t1_registry_growth_rss}" \
    "$((t1_registry_growth_rss - t1_registry_baseline_rss))"
done
cmp "${t1_tmp}/registry-baseline-1.stdout" \
  "${t1_tmp}/registry-baseline-2.stdout"
cmp "${t1_tmp}/registry-growth-1.stdout" \
  "${t1_tmp}/registry-growth-2.stdout"

/usr/bin/bash "${PROJECT_ROOT}/scripts/test-t1-boundary.sh"

printf 'T1 PASS: development oracle, native exact-i64 shell/sanitizers/C++, frozen and adversarial format, 47-global aggregate, public exact-byte/UTF-8 runtime, persistence, and deterministic AOT\n'
