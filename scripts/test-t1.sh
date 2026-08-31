#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp grep nm python3 timeout; do
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
grep -E '^T1 I64 SHELL PASS: [0-9]+ admission, exact-i64, lifetime, and failure checks$' \
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
[[ "$(nm -g --defined-only --format=posix "${t1_object}" | wc -l)" == 46 ]] || \
  die "T1 Wave 1 aggregate does not expose exactly 46 globals"

t1_runner="$(eshkol_build_dir)/eshkol-run"
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
  grep -E '^T1 PUBLIC PASS: [0-9]+ aggregate tokenizer checks$' \
    "${t1_tmp}/public-runtime-${repetition}.stdout" >/dev/null
  cmp "${t1_fixture}" "${t1_tmp}/saved-${repetition}.tsv"
done
cmp "${t1_tmp}/public-runtime-1.stdout" \
  "${t1_tmp}/public-runtime-2.stdout"

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

/usr/bin/bash "${PROJECT_ROOT}/scripts/test-t1-boundary.sh"

printf 'T1 PASS: development oracle, native exact-i64 shell/sanitizers/C++, frozen and adversarial format, 46-global aggregate, public exact-byte/UTF-8 runtime, persistence, and deterministic AOT\n'
