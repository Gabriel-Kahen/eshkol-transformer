#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command ar
require_command cmp
require_command env
require_command nm
require_command readelf
require_command rg
require_command strings
require_command timeout
verify_toolchain

p1_runner="$(eshkol_build_dir)/eshkol-run"
p1_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
p1_cc="$(tsv_value "${p1_provenance}" cc_path)"
p1_cxx="$(tsv_value "${p1_provenance}" cxx_path)"
p1_timeout="${P1_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${p1_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "P1_COMPILER_TIMEOUT_SECONDS must be a positive integer"

p1_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-p1.XXXXXX")"
trap 'rm -rf -- "${p1_tmp}"' EXIT

p1_public_source="${PROJECT_ROOT}/lib/transformer/module.esk"
p1_trusted_root="${PROJECT_ROOT}/internal/p1/lib"
p1_trusted_source="${p1_trusted_root}/transformer/module.esk"
p1_test_source="${PROJECT_ROOT}/tests/p1/module_state_test.esk"
p1_registry_test_source="${PROJECT_ROOT}/tests/p1/registry_atomicity_test.esk"
p1_provider_source="${PROJECT_ROOT}/tests/p1/providers/p1_test/tensor_provider.esk"
p1_example_source="${PROJECT_ROOT}/examples/p1/module_state_inspection.esk"
p1_public_stub_source="${PROJECT_ROOT}/tests/p1/public_stub_unsupported.esk"
p1_public_reverse_source="${PROJECT_ROOT}/tests/p1/public_import_reverse.esk"
p1_public_link="${p1_tmp}/identity-a/public"
p1_trusted_link="${p1_tmp}/identity-a/trusted"
p1_test_trusted_link="${p1_tmp}/identity-test/trusted"
p1_public_archive="${p1_public_link}/libeshkol_transformer_p1_identity.a"
p1_trusted_archive="${p1_trusted_link}/libeshkol_transformer_p1_identity.a"
p1_test_trusted_archive="${p1_test_trusted_link}/libeshkol_transformer_p1_identity.a"
p1_package_link="${p1_tmp}/package"
p1_package_object="${p1_package_link}/eshkol_transformer_p1.o"
p1_package_archive="${p1_package_link}/libeshkol_transformer_p1.a"

run_compiler() {
  timeout --foreground --signal=TERM --kill-after=5s \
    "${p1_timeout}s" env -u ESHKOL_PATH "${p1_runner}" "$@"
}

run_fresh_compiler() {
  local cache="$1"
  shift
  mkdir -p "${cache}"
  timeout --foreground --signal=TERM --kill-after=5s \
    "${p1_timeout}s" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${cache}" "${p1_runner}" "$@"
}

compile_public_object() {
  local source="$1" output="$2" depfile="$3" log="$4"
  run_fresh_compiler "${p1_tmp}/cache-object-$(basename -- "${output}")" \
    --strict-types --emit-object --no-stdlib \
    --emit-depfile "${depfile}" -I "${PROJECT_ROOT}/lib" \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -s "${output}" && -s "${depfile}" ]] || \
    die "P1 public object or depfile is missing: ${source}"
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "P1 compiler reported ERROR while returning success: ${source}"
}

compile_trusted_object() {
  local source="$1" output="$2" depfile="$3" log="$4"
  run_compiler --strict-types --emit-object --no-stdlib \
    --emit-depfile "${depfile}" -I "${p1_trusted_root}" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -s "${output}" && -s "${depfile}" ]] || \
    die "P1 trusted object or depfile is missing: ${source}"
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "P1 compiler reported ERROR while returning success: ${source}"
}

compile_public_aot() {
  local source="$1" output="$2" log="$3"
  run_fresh_compiler "${p1_tmp}/cache-aot-$(basename -- "${output}")" \
    --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -L "${p1_package_link}" --lib eshkol_transformer_p1 \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -x "${output}" ]] || die "P1 public AOT executable is missing"
}

compile_trusted_aot() {
  local source="$1" output="$2" log="$3"
  local link_dir="${4:-${p1_trusted_link}}"
  run_compiler --strict-types --no-stdlib -I "${p1_trusted_root}" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    -L "${link_dir}" --lib eshkol_transformer_p1_identity \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -x "${output}" ]] || die "P1 trusted AOT executable is missing"
}

"${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check
"${PROJECT_ROOT}/scripts/build-p1-identity.sh" \
  "${p1_tmp}/identity-a" normal all
"${PROJECT_ROOT}/scripts/build-p1-identity.sh" \
  "${p1_tmp}/identity-b" normal all

cmp "${p1_public_archive}" \
  "${p1_tmp}/identity-b/public/libeshkol_transformer_p1_identity.a"
cmp "${p1_trusted_archive}" \
  "${p1_tmp}/identity-b/trusted/libeshkol_transformer_p1_identity.a"

mkdir -p "${p1_package_link}" "${p1_tmp}/package-b"
cp "${p1_trusted_source}" "${p1_tmp}/copied-p1-root.esk"
if E1B_PACKAGE_PROFILE=p1 "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${p1_tmp}/copied-p1-root.esk" \
    "${PROJECT_ROOT}/native/p1_package_bridge.c" \
    "${PROJECT_ROOT}/native/p1_package_renames.txt" \
    "${PROJECT_ROOT}/native/p1_package_public_exports.txt" \
    "${p1_tmp}/copied-root-package.o" \
    >"${p1_tmp}/copied-root-package.log" 2>&1; then
  die "P1 copied trusted root opted into the wider undefined policy"
fi
grep -F 'P1 wider undefined-symbol policy requires the exact reviewed input tuple' \
  "${p1_tmp}/copied-root-package.log" >/dev/null
test ! -e "${p1_tmp}/copied-root-package.o"
test ! -e "${p1_tmp}/copied-root-package.o.evidence"

cp "${PROJECT_ROOT}/native/p1_package_bridge.c" \
  "${p1_tmp}/p1_package_bridge.c"
cp "${PROJECT_ROOT}/native/p1_identity.c" "${p1_tmp}/p1_identity.c"
cp "${PROJECT_ROOT}/native/p1_identity_internal.h" \
  "${p1_tmp}/p1_identity_internal.h"
if "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${p1_trusted_source}" \
    "${p1_tmp}/p1_package_bridge.c" \
    "${PROJECT_ROOT}/native/p1_package_renames.txt" \
    "${PROJECT_ROOT}/native/p1_package_public_exports.txt" \
    "${p1_tmp}/mismatched-package.o" \
    >"${p1_tmp}/mismatched-package.log" 2>&1; then
  die "P1 copied bridge/native inputs opted into the wider undefined policy"
fi
grep -F 'P1 wider undefined-symbol policy requires the exact reviewed input tuple' \
  "${p1_tmp}/mismatched-package.log" >/dev/null
test ! -e "${p1_tmp}/mismatched-package.o"
test ! -e "${p1_tmp}/mismatched-package.o.evidence"

E1B_PACKAGE_PROFILE=attacker \
  "${PROJECT_ROOT}/scripts/build-p1-package.sh" "${p1_package_object}"
"${PROJECT_ROOT}/scripts/build-p1-package.sh" \
  "${p1_tmp}/package-b/eshkol_transformer_p1.o"
cmp "${p1_package_object}" \
  "${p1_tmp}/package-b/eshkol_transformer_p1.o"
for evidence in global-defined.txt undefined.txt expected-undefined.txt \
                package-exports.txt readelf-symbols.txt link.map \
                allowlist-provenance.tsv; do
  cmp "${p1_package_object}.evidence/${evidence}" \
    "${p1_tmp}/package-b/eshkol_transformer_p1.o.evidence/${evidence}"
done
ar rcsD "${p1_package_archive}" "${p1_package_object}"
nm -s "${p1_package_archive}" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${p1_tmp}/package-archive-index.txt"
if rg 'et_e1b_(consumer|private|box|ensure)|et_p1_(private|public)_' \
    "${p1_tmp}/package-archive-index.txt" >/dev/null; then
  die "P1 package archive index exposed a localized identity seam"
fi
for leaked_name in transformer-error-make transformer-error-raise \
                   transformer-error-wrap-foreign e1-internal-dispatch \
                   et-e1b-private-raise; do
  if run_fresh_compiler "${p1_tmp}/cache-public-surface-${leaked_name}" \
      --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "${p1_package_link}" --lib eshkol_transformer_p1 \
      -e "(begin (require transformer.module) ${leaked_name})" \
      >"${p1_tmp}/public-surface-${leaked_name}.log" 2>&1; then
    die "P1 public root leaked a privileged flattened binding: ${leaked_name}"
  fi
  rg -F "${leaked_name}" \
    "${p1_tmp}/public-surface-${leaked_name}.log" >/dev/null || \
    die "P1 public surface negative did not diagnose: ${leaked_name}"
done

LC_ALL=C nm -g --defined-only "${p1_public_archive}" | \
  awk '$2 ~ /^[A-Z]$/ { print $3 }' | LC_ALL=C sort \
  >"${p1_tmp}/public-symbols.actual"
cmp "${PROJECT_ROOT}/native/p1_identity_public_symbols.txt" \
  "${p1_tmp}/public-symbols.actual"

LC_ALL=C nm -g --defined-only "${p1_trusted_archive}" | \
  awk '$2 ~ /^[A-Z]$/ { print $3 }' | LC_ALL=C sort \
  >"${p1_tmp}/trusted-symbols.actual"
cmp "${PROJECT_ROOT}/native/p1_identity_trusted_symbols.txt" \
  "${p1_tmp}/trusted-symbols.actual"

if nm -a "${p1_public_archive}" | rg 'et_p1_private_' >/dev/null; then
  die "P1 public archive contains a privileged symbol"
fi
if readelf -Ws "${p1_public_link}/p1_identity.o" | \
    rg 'et_p1_private_' >/dev/null; then
  die "P1 public object contains a privileged symbol"
fi
if strings -a "${p1_public_archive}" | \
    rg -i 'module_internal|p1-native-|fixture|provider-seal|provider-abort' \
      >/dev/null; then
  die "P1 public archive contains a private source or operation marker"
fi

trusted_private_count="$(readelf -Ws \
  "${p1_trusted_link}/p1_identity.o" | \
  awk '/et_p1_private_/ { if ($5 != "GLOBAL" || $6 != "HIDDEN") exit 2; n++ }
       END { print n + 0 }')" || die "P1 private symbol visibility changed"
[[ "${trusted_private_count}" == 25 ]] || \
  die "P1 trusted ABI symbol count changed: ${trusted_private_count}"

p1_public_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow
  -I "${PROJECT_ROOT}/native"
)
p1_cflags=("${p1_public_cflags[@]}" -DET_P1_PRIVATE_API=1)
mkdir -p "${p1_test_trusted_link}"
"${p1_cc}" "${p1_public_cflags[@]}" -fPIC -fvisibility=hidden -fno-common \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${p1_test_trusted_link}/p1_identity.o"
ar rcsD "${p1_test_trusted_archive}" \
  "${p1_test_trusted_link}/p1_identity.o"
if nm -a "${p1_public_archive}" "${p1_trusted_archive}" | \
    rg 'et_p1_test_' >/dev/null; then
  die "P1 production identity archives contain a test hook"
fi
{
  cat "${PROJECT_ROOT}/native/p1_identity_trusted_symbols.txt"
  printf '%s\n' \
    et_p1_test_callback_fail_after_v1 \
    et_p1_test_live_entry_count_v1 \
    et_p1_test_tombstone_count_v1
} | LC_ALL=C sort >"${p1_tmp}/test-trusted-symbols.expected"
LC_ALL=C nm -g --defined-only "${p1_test_trusted_archive}" | \
  awk '$2 ~ /^[A-Z]$/ { print $3 }' | LC_ALL=C sort \
  >"${p1_tmp}/test-trusted-symbols.actual"
cmp "${p1_tmp}/test-trusted-symbols.expected" \
  "${p1_tmp}/test-trusted-symbols.actual"
test_hook_count="$(readelf -Ws "${p1_test_trusted_link}/p1_identity.o" | \
  awk '/et_p1_test_/ { if ($5 != "GLOBAL" || $6 != "HIDDEN") exit 2; n++ }
       END { print n + 0 }')" || die "P1 test-hook visibility changed"
[[ "${test_hook_count}" == 3 ]] || \
  die "P1 test-only hook count changed: ${test_hook_count}"
"${p1_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_header.cpp" \
  "${p1_public_archive}" -o "${p1_tmp}/test-p1-identity-cxx-public"
"${p1_tmp}/test-p1-identity-cxx-public"
"${p1_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -DET_P1_CPP_TRUSTED_PROBE=1 -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_header.cpp" \
  "${p1_trusted_archive}" -o "${p1_tmp}/test-p1-identity-cxx-trusted"
"${p1_tmp}/test-p1-identity-cxx-trusted"
"${p1_cc}" "${p1_cflags[@]}" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity.c" "${p1_trusted_archive}" \
  -o "${p1_tmp}/test-p1-identity"
"${p1_tmp}/test-p1-identity" >"${p1_tmp}/identity.stdout"
grep -F 'P1 identity PASS: 118 checks' "${p1_tmp}/identity.stdout" >/dev/null

"${p1_cc}" "${p1_cflags[@]}" \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_failpoints.c" \
  "${p1_trusted_archive}" -Wl,--wrap=calloc -Wl,--wrap=getrandom \
  -o "${p1_tmp}/test-p1-identity-failpoints"
"${p1_tmp}/test-p1-identity-failpoints" \
  >"${p1_tmp}/failpoints.stdout"
grep -F 'P1 failpoint PASS: 57 checks' \
  "${p1_tmp}/failpoints.stdout" >/dev/null

"${p1_cc}" "${p1_public_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/p1/p1_public_identity_probe.c" \
  -o "${p1_tmp}/p1-public-probe.o"
"${p1_cc}" "${p1_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/p1/p1_trusted_identity_probe.c" \
  -o "${p1_tmp}/p1-trusted-probe.o"
"${p1_cc}" "${p1_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/p1/test_p1_cross_role_identity.c" \
  -o "${p1_tmp}/p1-cross-role-main.o"
"${p1_cc}" "${p1_tmp}/p1-public-probe.o" \
  "${p1_tmp}/p1-trusted-probe.o" "${p1_tmp}/p1-cross-role-main.o" \
  "${p1_trusted_archive}" -o "${p1_tmp}/test-p1-cross-role"
"${p1_tmp}/test-p1-cross-role" >"${p1_tmp}/cross-role.stdout"
grep -F 'P1 cross-role PASS: 8 checks' \
  "${p1_tmp}/cross-role.stdout" >/dev/null

"${PROJECT_ROOT}/scripts/build-p1-identity.sh" \
  "${p1_tmp}/identity-sanitize" sanitize trusted
sanitized_archive="${p1_tmp}/identity-sanitize/trusted/libeshkol_transformer_p1_identity.a"
"${p1_cc}" "${p1_cflags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${PROJECT_ROOT}/tests/p1/test_p1_identity.c" \
  "${sanitized_archive}" -o "${p1_tmp}/test-p1-identity-sanitized"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  "${p1_tmp}/test-p1-identity-sanitized" >/dev/null
"${p1_cc}" "${p1_cflags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_failpoints.c" \
  "${sanitized_archive}" -Wl,--wrap=calloc -Wl,--wrap=getrandom \
  -o "${p1_tmp}/test-p1-failpoints-sanitized"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  "${p1_tmp}/test-p1-failpoints-sanitized" >/dev/null

for run in 1 2; do
  compile_public_object "${p1_public_source}" "${p1_tmp}/module-${run}.o" \
    "${p1_tmp}/module-${run}.d" "${p1_tmp}/module-${run}.log"
  compile_public_object "${p1_example_source}" "${p1_tmp}/example-${run}.o" \
    "${p1_tmp}/example-${run}.d" "${p1_tmp}/example-${run}.log"
  compile_public_object "${p1_public_stub_source}" \
    "${p1_tmp}/public-stubs-${run}.o" \
    "${p1_tmp}/public-stubs-${run}.d" \
    "${p1_tmp}/public-stubs-${run}.log"
  compile_public_object "${p1_public_reverse_source}" \
    "${p1_tmp}/public-reverse-${run}.o" \
    "${p1_tmp}/public-reverse-${run}.d" \
    "${p1_tmp}/public-reverse-${run}.log"
  compile_trusted_object "${p1_test_source}" "${p1_tmp}/test-${run}.o" \
    "${p1_tmp}/test-${run}.d" "${p1_tmp}/test-${run}.log"

  compile_public_aot "${p1_example_source}" "${p1_tmp}/example-${run}" \
    "${p1_tmp}/example-${run}.aot.log"
  timeout --foreground --signal=TERM --kill-after=2s 20s \
    "${p1_tmp}/example-${run}" >"${p1_tmp}/example-${run}.stdout" \
    2>"${p1_tmp}/example-${run}.stderr"

  compile_public_aot "${p1_public_stub_source}" \
    "${p1_tmp}/public-stubs-${run}" \
    "${p1_tmp}/public-stubs-${run}.aot.log"
  timeout --foreground --signal=TERM --kill-after=2s 20s \
    "${p1_tmp}/public-stubs-${run}" \
    >"${p1_tmp}/public-stubs-${run}.stdout" \
    2>"${p1_tmp}/public-stubs-${run}.stderr"

  compile_public_aot "${p1_public_reverse_source}" \
    "${p1_tmp}/public-reverse-${run}" \
    "${p1_tmp}/public-reverse-${run}.aot.log"
  timeout --foreground --signal=TERM --kill-after=2s 20s \
    "${p1_tmp}/public-reverse-${run}" \
    >"${p1_tmp}/public-reverse-${run}.stdout" \
    2>"${p1_tmp}/public-reverse-${run}.stderr"

  compile_trusted_aot "${p1_test_source}" "${p1_tmp}/test-${run}" \
    "${p1_tmp}/test-${run}.aot.log"
  timeout --foreground --signal=TERM --kill-after=2s 30s \
    "${p1_tmp}/test-${run}" >"${p1_tmp}/test-${run}.stdout" \
    2>"${p1_tmp}/test-${run}.stderr"

  compile_trusted_aot "${p1_registry_test_source}" \
    "${p1_tmp}/registry-${run}" "${p1_tmp}/registry-${run}.aot.log" \
    "${p1_test_trusted_link}"
  timeout --foreground --signal=TERM --kill-after=2s 30s \
    "${p1_tmp}/registry-${run}" >"${p1_tmp}/registry-${run}.stdout" \
    2>"${p1_tmp}/registry-${run}.stderr"
done

for run in 1 2; do
  for stem in module example public-stubs public-reverse; do
    depfile="${p1_tmp}/${stem}-${run}.d"
    grep -F "${p1_public_source}" "${depfile}" >/dev/null || \
      die "P1 public depfile omits transformer.module"
    if rg 'module_internal|templates/p1|tests/p1/providers|native/p1_identity|internal/p1|error_(internal|core)|e1b_error_consumer_private' \
        "${depfile}" >/dev/null; then
      die "P1 public dependency closure contains a trusted source"
    fi
    if nm -a "${p1_tmp}/${stem}-${run}.o" | \
        rg 'et_p1_private_|et_e1b_(private|consumer|box|ensure)|p1-native-|module_internal' >/dev/null; then
      die "P1 public Eshkol object contains a private symbol"
    fi
    if strings -a "${p1_tmp}/${stem}-${run}.o" | \
        rg -i 'module_internal|p1-native-|fixture|p1-core-dispatch' >/dev/null; then
      die "P1 public Eshkol object contains a private marker"
    fi
  done
  grep -F "${p1_trusted_source}" "${p1_tmp}/test-${run}.d" >/dev/null || \
    die "P1 trusted depfile omits its alternative transformer.module root"
  if grep -F "${p1_public_source}" "${p1_tmp}/test-${run}.d" >/dev/null; then
    die "P1 trusted dependency closure contains the public Eshkol root"
  fi
  grep -F "${p1_provider_source}" "${p1_tmp}/test-${run}.d" >/dev/null || \
    die "P1 trusted depfile omits the explicit test provider"
  nm -u "${p1_tmp}/test-${run}.o" | \
    rg 'et_p1_private_provider_seal_v1' >/dev/null || \
    die "P1 trusted object omits its explicit private bridge reference"
done

for stem in module example public-stubs public-reverse test; do
  cmp "${p1_tmp}/${stem}-1.o" "${p1_tmp}/${stem}-2.o"
  cmp "${p1_tmp}/${stem}-1.log" "${p1_tmp}/${stem}-2.log"
done
cmp "${p1_tmp}/example-1.aot.log" "${p1_tmp}/example-2.aot.log"
cmp "${p1_tmp}/public-stubs-1.aot.log" \
  "${p1_tmp}/public-stubs-2.aot.log"
cmp "${p1_tmp}/public-reverse-1.aot.log" \
  "${p1_tmp}/public-reverse-2.aot.log"
cmp "${p1_tmp}/test-1.aot.log" "${p1_tmp}/test-2.aot.log"
cmp "${p1_tmp}/registry-1.aot.log" "${p1_tmp}/registry-2.aot.log"
cmp "${p1_tmp}/example-1.stdout" "${p1_tmp}/example-2.stdout"
cmp "${p1_tmp}/public-stubs-1.stdout" \
  "${p1_tmp}/public-stubs-2.stdout"
cmp "${p1_tmp}/public-reverse-1.stdout" \
  "${p1_tmp}/public-reverse-2.stdout"
cmp "${p1_tmp}/test-1.stdout" "${p1_tmp}/test-2.stdout"
cmp "${p1_tmp}/registry-1.stdout" "${p1_tmp}/registry-2.stdout"
cmp "${PROJECT_ROOT}/tests/expected/p1-example.stdout" \
  "${p1_tmp}/example-1.stdout"
cmp "${PROJECT_ROOT}/tests/expected/p1-public-stubs.stdout" \
  "${p1_tmp}/public-stubs-1.stdout"
cmp "${PROJECT_ROOT}/tests/expected/p1-public-import-reverse.stdout" \
  "${p1_tmp}/public-reverse-1.stdout"
cmp "${PROJECT_ROOT}/tests/expected/p1.stdout" "${p1_tmp}/test-1.stdout"
cmp "${PROJECT_ROOT}/tests/expected/p1-registry-atomicity.stdout" \
  "${p1_tmp}/registry-1.stdout"
[[ ! -s "${p1_tmp}/example-1.stderr" && \
   ! -s "${p1_tmp}/example-2.stderr" && \
   ! -s "${p1_tmp}/public-stubs-1.stderr" && \
   ! -s "${p1_tmp}/public-stubs-2.stderr" && \
   ! -s "${p1_tmp}/public-reverse-1.stderr" && \
   ! -s "${p1_tmp}/public-reverse-2.stderr" && \
   ! -s "${p1_tmp}/test-1.stderr" && \
   ! -s "${p1_tmp}/test-2.stderr" && \
   ! -s "${p1_tmp}/registry-1.stderr" && \
   ! -s "${p1_tmp}/registry-2.stderr" ]] || \
  die "P1 runtime gate wrote stderr"

negative_sources=(
  negative_public_provider_internal
  negative_public_module_internal
  negative_public_handle_internal
  negative_public_state_entry_internal
  negative_public_state_dict_internal
)
for stem in "${negative_sources[@]}"; do
  source="${PROJECT_ROOT}/tests/p1/${stem}.esk"
  output="${p1_tmp}/${stem}.o"
  for run in 1 2; do
    log="${p1_tmp}/${stem}-source-${run}.log"
    if run_fresh_compiler "${p1_tmp}/cache-${stem}-source-${run}" \
        --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
        -L "${p1_package_link}" --lib eshkol_transformer_p1 -r \
        "${source}" >"${log}" 2>&1; then
      die "P1 public internal-name negative unexpectedly ran: ${stem}"
    fi

    log="${p1_tmp}/${stem}-object-${run}.log"
    if run_fresh_compiler "${p1_tmp}/cache-${stem}-object-${run}" \
        --strict-types --emit-object --no-stdlib -I "${PROJECT_ROOT}/lib" \
        "${source}" -o "${output}" >"${log}" 2>&1; then
      die "P1 public internal-name negative unexpectedly compiled: ${stem}"
    fi
    [[ ! -e "${output}" ]] || die "P1 internal-name negative left an object"

    log="${p1_tmp}/${stem}-aot-${run}.log"
    if run_fresh_compiler "${p1_tmp}/cache-${stem}-aot-${run}" \
        --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
        -L "${p1_package_link}" --lib eshkol_transformer_p1 \
        "${source}" -o "${p1_tmp}/${stem}" >"${log}" 2>&1; then
      die "P1 public internal-name negative unexpectedly linked: ${stem}"
    fi
    [[ ! -e "${p1_tmp}/${stem}" ]] || \
      die "P1 internal-name negative left an executable"
  done
  # E1B is deliberately AOT-only; JIT diagnostics include set-ordered missing
  # wrapper names. Determinism is asserted for object and AOT diagnostics.
  for phase in object aot; do
    sed -e "s#cache-${stem}-${phase}-1#cache-${stem}-${phase}-RUN#g" \
      "${p1_tmp}/${stem}-${phase}-1.log" \
      >"${p1_tmp}/${stem}-${phase}-1.normalized.log"
    sed -e "s#cache-${stem}-${phase}-2#cache-${stem}-${phase}-RUN#g" \
      "${p1_tmp}/${stem}-${phase}-2.log" \
      >"${p1_tmp}/${stem}-${phase}-2.normalized.log"
    cmp "${p1_tmp}/${stem}-${phase}-1.normalized.log" \
      "${p1_tmp}/${stem}-${phase}-2.normalized.log"
  done
done

while IFS=$'\t' read -r visibility name; do
  if [[ "${visibility}" == internal ]]; then
    for phase in source object aot; do
      rg -F "${name}" "${p1_tmp}"/negative_public_*-"${phase}"-1.log \
        >/dev/null || \
        die "P1 public ${phase} negative did not diagnose ${name}"
    done
  fi
done <"${PROJECT_ROOT}/tools/p1/module_surface.tsv"
for name in p1-native-context-create p1-native-provider-seal p1-core-dispatch; do
  for phase in source object aot; do
    rg -F "${name}" \
      "${p1_tmp}/negative_public_provider_internal-${phase}-1.log" \
      >/dev/null || \
      die "P1 public ${phase} negative did not diagnose ${name}"
  done
done

for run in 1 2; do
  guessed_object="${p1_tmp}/guessed-private.o"
  run_fresh_compiler "${p1_tmp}/cache-guessed-object-${run}" \
    --strict-types --emit-object --no-stdlib \
    "${PROJECT_ROOT}/tests/p1/negative_guessed_private_native.esk" \
    -o "${guessed_object}" >"${p1_tmp}/guessed-object-${run}.log" 2>&1
  for symbol in et_p1_private_context_create_v1 \
                et_p1_private_provider_abort_v1 \
                et_p1_private_callback_identity_revoke_v1; do
    nm -u "${guessed_object}" | rg -F "${symbol}" >/dev/null || \
      die "P1 guessed-private object lacks undefined symbol: ${symbol}"
  done
  if run_fresh_compiler "${p1_tmp}/cache-guessed-aot-${run}" \
      --strict-types --no-stdlib -L "${p1_package_link}" \
      --lib eshkol_transformer_p1 \
      "${PROJECT_ROOT}/tests/p1/negative_guessed_private_native.esk" \
      -o "${p1_tmp}/guessed-private" \
      >"${p1_tmp}/guessed-aot-${run}.log" 2>&1; then
    die "P1 guessed private native symbol linked through the public archive"
  fi
  [[ ! -e "${p1_tmp}/guessed-private" ]] || \
    die "P1 guessed-private link negative left an executable"
done
cmp "${p1_tmp}/guessed-object-1.log" "${p1_tmp}/guessed-object-2.log"
cmp "${p1_tmp}/guessed-aot-1.log" "${p1_tmp}/guessed-aot-2.log"
"${p1_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -c \
  "${PROJECT_ROOT}/tests/p1/p1_all_identity_symbols_probe.c" \
  -o "${p1_tmp}/all-private-guesses.o"
"${p1_cxx}" -r "${p1_tmp}/all-private-guesses.o" \
  "${p1_package_object}" -o "${p1_tmp}/all-private-package-link.o"
awk '/^et_p1_private_/ { print }' \
  "${PROJECT_ROOT}/native/p1_identity_trusted_symbols.txt" \
  >"${p1_tmp}/all-private.expected"
nm -u --format=posix "${p1_tmp}/all-private-package-link.o" | \
  awk '$1 ~ /^et_p1_private_/ { print $1 }' | LC_ALL=C sort \
  >"${p1_tmp}/all-private.actual"
cmp "${p1_tmp}/all-private.expected" "${p1_tmp}/all-private.actual"
"${p1_cxx}" -r "${p1_tmp}/all-private-guesses.o" \
  "${p1_trusted_archive}" -o "${p1_tmp}/all-private-trusted-link.o"
if nm -u --format=posix "${p1_tmp}/all-private-trusted-link.o" | \
    awk '{ print $1 }' | rg '^et_p1_private_' >/dev/null; then
  die "P1 noninstalled trusted positive control left a private guess unresolved"
fi
run_compiler --strict-types --no-stdlib -L "${p1_trusted_link}" \
  --lib eshkol_transformer_p1_identity \
  "${PROJECT_ROOT}/tests/p1/negative_guessed_private_native.esk" \
  -o "${p1_tmp}/trusted-private-positive" >/dev/null 2>&1
"${p1_tmp}/trusted-private-positive" >/dev/null

for run in 1 2; do
  for negative in negative_wrong_arity negative_fixture_not_public \
                  negative_provider_callback_arity; do
    output="${p1_tmp}/${negative}.o"
    log="${p1_tmp}/${negative}-${run}.log"
    includes=(-I "${PROJECT_ROOT}/lib")
    if [[ "${negative}" == negative_provider_callback_arity ]]; then
      includes=(-I "${p1_trusted_root}" -I "${PROJECT_ROOT}/lib"
                -I "${PROJECT_ROOT}/tests/p1/providers")
    fi
    if run_fresh_compiler "${p1_tmp}/cache-${negative}-${run}" \
        --strict-types --emit-object --no-stdlib "${includes[@]}" \
        "${PROJECT_ROOT}/tests/p1/${negative}.esk" -o "${output}" \
        >"${log}" 2>&1; then
      die "P1 compile negative unexpectedly succeeded: ${negative}"
    fi
    [[ ! -e "${output}" ]] || die "P1 compile negative left an object"
  done
done
cmp "${p1_tmp}/negative_wrong_arity-1.log" \
  "${p1_tmp}/negative_wrong_arity-2.log"
cmp "${p1_tmp}/negative_fixture_not_public-1.log" \
  "${p1_tmp}/negative_fixture_not_public-2.log"
cmp "${p1_tmp}/negative_provider_callback_arity-1.log" \
  "${p1_tmp}/negative_provider_callback_arity-2.log"

if rg -ni '(fixture|tests/p1/providers|python|pytorch|torch\.load|finite[- ]difference|scalar[- ]fallback|cpu[- ]fallback)' \
    "${PROJECT_ROOT}/lib/transformer/module.esk" \
    "${p1_trusted_source}"; then
  die "P1 production Eshkol roots contain a forbidden fallback reference"
fi
if sed -n '/^(define (module-load-state-dict!/,/^(define (set-mode-recursive!/p' \
    "${p1_trusted_source}" | \
    rg 'state-from-flat|state-provider-bind!|p1-native-state-create' >/dev/null; then
  die "P1 strict load creates or binds a temporary expected state"
fi

printf 'P1 PASS: E1B-integrated public/private packaging, 178 structural checks, 183 native checks, 28 registry-atomicity checks, sanitizers, negatives, atomicity, and determinism\n'
