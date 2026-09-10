#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp nm readelf rg strings timeout; do
  require_command "${command}"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-p1-native.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

"${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check
for run in a b; do
  "${PROJECT_ROOT}/scripts/build-p1-identity.sh" \
    "${temporary_dir}/identity-${run}" normal all
done
public_archive="${temporary_dir}/identity-a/public/libeshkol_transformer_p1_identity.a"
trusted_archive="${temporary_dir}/identity-a/trusted/libeshkol_transformer_p1_identity.a"
cmp "${public_archive}" \
  "${temporary_dir}/identity-b/public/libeshkol_transformer_p1_identity.a"
cmp "${trusted_archive}" \
  "${temporary_dir}/identity-b/trusted/libeshkol_transformer_p1_identity.a"

nm -g --defined-only "${public_archive}" | awk '$2 ~ /^[A-Z]$/ { print $3 }' | \
  LC_ALL=C sort >"${temporary_dir}/public-symbols.txt"
cmp "${PROJECT_ROOT}/native/p1_identity_public_symbols.txt" \
  "${temporary_dir}/public-symbols.txt"
nm -g --defined-only "${trusted_archive}" | awk '$2 ~ /^[A-Z]$/ { print $3 }' | \
  LC_ALL=C sort >"${temporary_dir}/trusted-symbols.txt"
cmp "${PROJECT_ROOT}/native/p1_identity_trusted_symbols.txt" \
  "${temporary_dir}/trusted-symbols.txt"
if nm -a "${public_archive}" | rg 'et_p1_private_' >/dev/null || \
   strings -a "${public_archive}" | \
     rg -i 'module_internal|p1-native-|fixture|provider-seal|provider-abort' >/dev/null; then
  die "P1 public identity archive exposes a private authority"
fi

public_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native"
)
trusted_cflags=("${public_cflags[@]}" -DET_P1_PRIVATE_API=1)

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_header.cpp" \
  "${public_archive}" -o "${temporary_dir}/header-public"
"${temporary_dir}/header-public"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -DET_P1_CPP_TRUSTED_PROBE=1 -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_header.cpp" \
  "${trusted_archive}" -o "${temporary_dir}/header-trusted"
"${temporary_dir}/header-trusted"

"${cc}" "${trusted_cflags[@]}" "${PROJECT_ROOT}/tests/p1/test_p1_identity.c" \
  "${trusted_archive}" -o "${temporary_dir}/test-identity"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${temporary_dir}/test-identity" >"${temporary_dir}/identity.stdout"
grep -F 'P1 identity PASS: 274 checks' "${temporary_dir}/identity.stdout" >/dev/null

test_hook_dir="${temporary_dir}/identity-test"
mkdir -p "${test_hook_dir}"
"${cc}" "${public_cflags[@]}" -fPIC -fvisibility=hidden -fno-common \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" -o "${test_hook_dir}/p1_identity.o"
ar rcsD "${test_hook_dir}/libeshkol_transformer_p1_identity.a" \
  "${test_hook_dir}/p1_identity.o"
"${cc}" "${trusted_cflags[@]}" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_failpoints.c" \
  "${test_hook_dir}/libeshkol_transformer_p1_identity.a" \
  -Wl,--wrap=calloc -Wl,--wrap=getrandom -o "${temporary_dir}/test-failpoints"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${temporary_dir}/test-failpoints" >"${temporary_dir}/failpoints.stdout"
grep -F 'P1 failpoint PASS: 123 checks' "${temporary_dir}/failpoints.stdout" >/dev/null

for role in public trusted; do
  flags=("${public_cflags[@]}")
  [[ "${role}" == trusted ]] && flags=("${trusted_cflags[@]}")
  "${cc}" "${flags[@]}" -c \
    "${PROJECT_ROOT}/tests/p1/p1_${role}_identity_probe.c" \
    -o "${temporary_dir}/${role}-probe.o"
done
"${cc}" "${trusted_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/p1/test_p1_cross_role_identity.c" \
  -o "${temporary_dir}/cross-role.o"
"${cc}" "${temporary_dir}/public-probe.o" "${temporary_dir}/trusted-probe.o" \
  "${temporary_dir}/cross-role.o" "${trusted_archive}" \
  -o "${temporary_dir}/test-cross-role"
"${temporary_dir}/test-cross-role" >"${temporary_dir}/cross-role.stdout"
grep -F 'P1 cross-role PASS: 8 checks' "${temporary_dir}/cross-role.stdout" >/dev/null

"${PROJECT_ROOT}/scripts/build-p1-identity.sh" \
  "${temporary_dir}/identity-sanitized" sanitize trusted
sanitized_archive="${temporary_dir}/identity-sanitized/trusted/libeshkol_transformer_p1_identity.a"
sanitized_hook_dir="${temporary_dir}/identity-test-sanitized"
mkdir -p "${sanitized_hook_dir}"
"${cc}" "${public_cflags[@]}" -fPIC -fvisibility=hidden -fno-common \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1 \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${sanitized_hook_dir}/p1_identity.o"
ar rcsD "${sanitized_hook_dir}/libeshkol_transformer_p1_identity.a" \
  "${sanitized_hook_dir}/p1_identity.o"
"${cc}" "${trusted_cflags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${PROJECT_ROOT}/tests/p1/test_p1_identity.c" \
  "${sanitized_archive}" -o "${temporary_dir}/test-identity-sanitized"
ASAN_OPTIONS=detect_leaks="${P1_LSAN:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${temporary_dir}/test-identity-sanitized" >/dev/null
"${cc}" "${trusted_cflags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_failpoints.c" \
  "${sanitized_hook_dir}/libeshkol_transformer_p1_identity.a" \
  -Wl,--wrap=calloc -Wl,--wrap=getrandom \
  -o "${temporary_dir}/test-failpoints-sanitized"
ASAN_OPTIONS=detect_leaks="${P1_LSAN:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${temporary_dir}/test-failpoints-sanitized" >/dev/null

printf 'P1 NATIVE PASS: ABI, ownership, failpoints, cross-role identity, determinism, and sanitizers\n'
