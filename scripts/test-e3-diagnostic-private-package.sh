#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar awk cmp grep nm readelf; do require_command "${command}"; done

artifact="${1:-$(project_build_dir)/e3-diagnostic-private}"
object="${artifact}/e3_diagnostic_private_package.o"
archive="${artifact}/libeshkol_transformer_e3_diagnostic_private.a"
evidence="${object}.evidence"
[[ -s "${object}" && -s "${archive}" && -d "${evidence}" ]] || \
  die "E3 diagnostic private package artifact is incomplete"

for pair in \
    defined_symbols:global-defined \
    public_exports:package-exports \
    public_strings:public-strings \
    undefined_symbols:undefined \
    source_closure:source-closure \
    native_source_closure:native-source-closure \
    native_objects:native-objects; do
  expected=${pair%%:*}
  observed=${pair#*:}
  cmp "${PROJECT_ROOT}/native/e3_diagnostic_private_package_${expected}.txt" \
    "${evidence}/${observed}.txt"
done
cmp "${PROJECT_ROOT}/native/e3_diagnostic_private_package_undefined_symbols.txt" \
  "${evidence}/expected-undefined.txt"
ar t "${archive}" | \
  cmp "${PROJECT_ROOT}/native/e3_diagnostic_private_package_archive_members.txt" -

for symbol in \
    et_e3_private_frame_create_v1 et_e3_private_acquire_v1 \
    et_e3_private_publish_v1 et_e3_private_selected_metric_bits_ref_v1 \
    et_e3_diagnostic_destination_create_v1 \
    et_e3_diagnostic_destination_bits_v1 \
    et_e3_diagnostic_destination_destroy_v1 \
    et_e3_diagnostic_test_run_cabi_v1 \
    et_e3_diagnostic_test_failure_cabi_v1; do
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
    "${evidence}/readelf-symbols.txt" >/dev/null || \
    die "E3 diagnostic required private definition is not local: ${symbol}"
done
if nm -g --defined-only --format=posix "${object}" | awk '{print $1}' | \
    grep -E '^et_e3_(private_|diagnostic_destination_)'; then
  die "E3 diagnostic object exposes private destination or frame authority"
fi

temporary="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-e3-diagnostic-package.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT
ar p "${archive}" e3_diagnostic_private_package.o \
  >"${temporary}/archive-member.o"
cmp "${object}" "${temporary}/archive-member.o"

printf 'E3 diagnostic source-private package manifests and localization passed\n'
