#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp env grep ln nm python3 readelf timeout; do
  require_command "${command}"
done

artifact="${1:-$(project_build_dir)/e3-private}"
object="${artifact}/e3_private_package.o"
archive="${artifact}/libeshkol_transformer_e3_private.a"
evidence="${object}.evidence"
[[ -s "${object}" && -s "${archive}" && -d "${evidence}" ]] || \
  die "E3 private package artifact is incomplete"

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
  cmp "${PROJECT_ROOT}/native/e3_private_package_${expected}.txt" \
    "${evidence}/${observed}.txt"
done
cmp "${PROJECT_ROOT}/native/e3_private_package_undefined_symbols.txt" \
  "${evidence}/expected-undefined.txt"
ar t "${archive}" | \
  cmp "${PROJECT_ROOT}/native/e3_private_package_archive_members.txt" -

for symbol in \
    et_e3_private_frame_create_v1 et_e3_private_acquire_v1 \
    et_e3_private_stage_inputs_v1 et_e3_private_forward_role_v1 \
    et_e3_private_publish_v1 et_e3_private_selected_metric_bits_ref_v1 \
    et_e3_d2_dataset_idle_preflight_v1 \
    et_l3s_kernel_provider_v1 et_e3_metrics_kernel_provider_v1 \
    et_e3_private_test_run_cabi_v1 et_e3_test_destination_create_v1 \
    et_e3_test_destination_destroy_v1 et_e3_test_destination_bits_v1 \
    et_e3_test_frame_fail_alloc_after_v1; do
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
    "${evidence}/readelf-symbols.txt" >/dev/null || \
    die "E3 required private definition is not local: ${symbol}"
done
if nm -g --defined-only --format=posix "${object}" | awk '{print $1}' | \
    grep -E '^et_e3_private_|^et_e3_test_(destination|frame)_'; then
  die "E3 object exposes private or test-hook authority"
fi

temporary="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-e3-package.XXXXXX")"
cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then rm -rf -- "${temporary}"; else
    printf 'E3 package failure evidence retained at %s\n' "${temporary}" >&2
  fi
}
trap cleanup EXIT

ar p "${archive}" e3_private_package.o >"${temporary}/archive-member.o"
cmp "${object}" "${temporary}/archive-member.o"

mkdir -p "${temporary}/publish/staging" "${temporary}/publish/target"
printf 'new-generation\n' >"${temporary}/publish/staging/identity"
printf 'old-generation\n' >"${temporary}/publish/target/identity"
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${temporary}/publish/staging" "${temporary}/publish/target"
grep -Fx new-generation "${temporary}/publish/target/identity" >/dev/null
grep -Fx old-generation "${temporary}/publish/staging/identity" >/dev/null
mkdir "${temporary}/publish/staging-new"
printf 'first-generation\n' >"${temporary}/publish/staging-new/identity"
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${temporary}/publish/staging-new" "${temporary}/publish/target-new"
grep -Fx first-generation "${temporary}/publish/target-new/identity" >/dev/null
[[ ! -e "${temporary}/publish/staging-new" ]] || \
  die "E3 first-generation publication left a staging path"
mkdir "${temporary}/publish/staging-link"
printf 'rejected-generation\n' >"${temporary}/publish/staging-link/identity"
ln -s "${temporary}/publish/target" "${temporary}/publish/target-link"
if python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
    "${temporary}/publish/staging-link" "${temporary}/publish/target-link" \
    >"${temporary}/publish/link.stdout" \
    2>"${temporary}/publish/link.stderr"; then
  die "E3 artifact publication accepted a symlink target"
fi
grep -F "E3 artifact target must be a directory" \
  "${temporary}/publish/link.stderr" >/dev/null
grep -Fx rejected-generation \
  "${temporary}/publish/staging-link/identity" >/dev/null

python3 "${PROJECT_ROOT}/scripts/generate-e3-d2-source.py" \
  --output-dir "${temporary}/generated" >"${temporary}/d2-generation.json"
cmp "${temporary}/d2-generation.json" "${evidence}/d2-generation.json"
cmp "${temporary}/generated/source/e3_d2_dataset.esk" \
  "${evidence}/e3_d2_dataset.esk"
cmp "${temporary}/generated/source/e3_d2_source_provenance.json" \
  "${evidence}/e3_d2_source_provenance.json"

# The lower-level builder owns the generated source.  Its former seventh
# include-root shape must fail before compilation and publish no object.
if E1B_COMPILER_TIMEOUT_SECONDS=30 /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/e3_private_driver_root.esk" \
    "${PROJECT_ROOT}/native/e3_private_bridge.c" \
    "${PROJECT_ROOT}/native/e3_private_package_private_renames.txt" \
    "${PROJECT_ROOT}/native/e3_private_package_public_exports.txt" \
    "${temporary}/forged.o" \
    "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
    "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib" \
    "${temporary}/generated/source" \
    >"${temporary}/forged.stdout" 2>"${temporary}/forged.stderr"; then
  die "E3 lower-level builder accepted caller-owned generated source"
fi
grep -F "E3 policy requires exact ordered trusted include roots" \
  "${temporary}/forged.stderr" >/dev/null || \
  die "E3 caller-owned source fixture failed for another reason"
[[ ! -e "${temporary}/forged.o" && \
   ! -e "${temporary}/forged.o.evidence" ]] || \
  die "rejected E3 generated source published an artifact"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runtime="$(eshkol_build_dir)/libeshkol-runtime.a"
[[ -s "${runtime}" ]] || die "E3 hostile-link runtime archive is missing"
link_libraries=(-lpng -ljpeg -lwebp -lz -lopenblas -lcrypto -pthread -ldl -lm)
printf '%s\n' 'int main(void) { return 0; }' >"${temporary}/control.c"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -c "${temporary}/control.c" -o "${temporary}/control.o"
"${cxx}" "${temporary}/control.o" -Wl,--whole-archive \
  "${archive}" "${runtime}" -Wl,--no-whole-archive "${link_libraries[@]}" \
  -o "${temporary}/control" \
  >"${temporary}/control.stdout" 2>"${temporary}/control.stderr"
[[ -x "${temporary}/control" ]] || \
  die "E3 whole-archive control link did not publish an executable"
printf '%s\n' \
  'extern void et_e3_private_publish_v1(void);' \
  'int main(void) { et_e3_private_publish_v1(); return 0; }' \
  >"${temporary}/hostile.c"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -c "${temporary}/hostile.c" -o "${temporary}/hostile.o"
if "${cxx}" "${temporary}/hostile.o" -Wl,--whole-archive \
    "${archive}" "${runtime}" -Wl,--no-whole-archive "${link_libraries[@]}" \
    -o "${temporary}/hostile" \
    >"${temporary}/hostile.stdout" 2>"${temporary}/hostile.stderr"; then
  die "E3 localized private authority satisfied an external link"
fi
grep -F et_e3_private_publish_v1 "${temporary}/hostile.stderr" >/dev/null || \
  die "E3 hostile private link failed for another reason"
[[ ! -e "${temporary}/hostile" ]] || \
  die "E3 hostile private link published an executable"

printf 'E3 private package policy, exact surfaces, localization and hostile link passed\n'
