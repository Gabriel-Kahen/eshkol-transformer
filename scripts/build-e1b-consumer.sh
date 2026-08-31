#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ "$#" -ge 5 ]] || die \
  "usage: $0 TRUSTED_PRIVATE_ROOT.esk PACKAGE_BRIDGE.c PACKAGE_RENAMES.txt PUBLIC_EXPORTS.txt OUTPUT.o [INCLUDE_DIR ...]"

require_command realpath
private_root="$(realpath -- "$1")"
package_bridge="$(realpath -- "$2")"
package_renames="$(realpath -- "$3")"
public_exports="$(realpath -- "$4")"
output_object="$(realpath -m -- "$5")"
shift 5
include_dirs=("$@")

[[ -f "${private_root}" ]] || die "E1B private root not found: ${private_root}"
[[ -f "${package_bridge}" ]] || die "E1B package bridge not found: ${package_bridge}"
[[ -f "${package_renames}" ]] || die "E1B rename map not found: ${package_renames}"
[[ -f "${public_exports}" ]] || die "E1B public export list not found: ${public_exports}"
[[ "${output_object}" == *.o ]] || die "E1B output must end in .o"

canonical_include_dirs=()
for include_dir in "${include_dirs[@]}"; do
  [[ -d "${include_dir}" ]] || die "E1B include directory not found: ${include_dir}"
  canonical_include_dirs+=("$(realpath -- "${include_dir}")")
done

base_undefined_symbols="$(realpath -- "${PROJECT_ROOT}/native/e1b_undefined_symbols.txt")"
x1_private_root="$(realpath -- "${PROJECT_ROOT}/native/x1_config_consumer_root.esk")"
x1_package_bridge="$(realpath -- "${PROJECT_ROOT}/native/x1_config_consumer_bridge.c")"
x1_package_renames="$(realpath -- "${PROJECT_ROOT}/native/x1_config_private_renames.txt")"
x1_public_exports="$(realpath -- "${PROJECT_ROOT}/native/x1_config_public_exports.txt")"
x1_include_dir="$(realpath -- "${PROJECT_ROOT}/native")"
x1_undefined_symbols="$(realpath -- "${PROJECT_ROOT}/native/x1_undefined_symbols.txt")"
p1_private_root="$(realpath -- "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk")"
p1_package_bridge="$(realpath -- "${PROJECT_ROOT}/native/p1_package_bridge.c")"
p1_package_renames="$(realpath -- "${PROJECT_ROOT}/native/p1_package_renames.txt")"
p1_public_exports="$(realpath -- "${PROJECT_ROOT}/native/p1_package_public_exports.txt")"
p1_undefined_symbols="$(realpath -- "${PROJECT_ROOT}/native/p1_package_undefined_symbols.txt")"
d1_private_root="$(realpath -- "${PROJECT_ROOT}/native/d1_e1b_private.esk")"
d1_fault_private_root="$(realpath -- "${PROJECT_ROOT}/tests/d1/d1_e1b_fault_root.esk")"
d1_package_bridge="$(realpath -- "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c")"
d1_package_renames="$(realpath -- "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt")"
d1_public_exports="$(realpath -- "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt")"
d1_include_dir="$(realpath -- "${PROJECT_ROOT}/src")"
d1_undefined_symbols="$(realpath -- "${PROJECT_ROOT}/native/d1_e1b_undefined_symbols.txt")"
d1_fault_undefined_symbols="$(realpath -- "${PROJECT_ROOT}/native/d1_e1b_fault_undefined_symbols.txt")"
d1_native_source="$(realpath -- "${PROJECT_ROOT}/native/data_io.c")"

[[ -z "${E1B_PACKAGE_POLICY+x}" ]] || \
  die "E1B_PACKAGE_POLICY overrides are forbidden; package policy is derived from exact repository inputs"
package_policy=
package_native_source=
package_native_define=
if [[ "${private_root}" == "${d1_private_root}" || \
      "${private_root}" == "${d1_fault_private_root}" ]]; then
    [[ "${package_bridge}" == "${d1_package_bridge}" ]] || \
      die "D1 package policy requires the exact repository D1 bridge"
    [[ "${package_renames}" == "${d1_package_renames}" ]] || \
      die "D1 package policy requires the exact repository D1 rename map"
    [[ "${public_exports}" == "${d1_public_exports}" ]] || \
      die "D1 package policy requires the exact repository D1 export list"
    [[ "${#canonical_include_dirs[@]}" == 1 && \
       "${canonical_include_dirs[0]}" == "${d1_include_dir}" ]] || \
      die "D1 package policy requires exactly the repository src include directory"
    package_native_source="${d1_native_source}"
    if [[ "${private_root}" == "${d1_private_root}" ]]; then
      package_policy=d1
      undefined_symbols="${d1_undefined_symbols}"
    else
      package_policy=d1-test-faults
      undefined_symbols="${d1_fault_undefined_symbols}"
      package_native_define=-DET_D1_TEST_FAULTS
    fi
elif [[ "${private_root}" == "${x1_private_root}" ]]; then
  [[ "${package_bridge}" == "${x1_package_bridge}" ]] || \
    die "X1 package policy requires the exact repository bridge"
  [[ "${package_renames}" == "${x1_package_renames}" ]] || \
    die "X1 package policy requires the exact repository rename map"
  [[ "${public_exports}" == "${x1_public_exports}" ]] || \
    die "X1 package policy requires the exact repository export list"
  [[ "${#canonical_include_dirs[@]}" == 1 && \
     "${canonical_include_dirs[0]}" == "${x1_include_dir}" ]] || \
    die "X1 package policy requires exactly the repository native include directory"
  package_policy=x1
  undefined_symbols="${x1_undefined_symbols}"
elif [[ "${private_root}" == "${p1_private_root}" ]]; then
  [[ "${package_bridge}" == "${p1_package_bridge}" ]] || \
    die "P1 wider undefined-symbol policy requires the exact reviewed input tuple"
  [[ "${package_renames}" == "${p1_package_renames}" ]] || \
    die "P1 wider undefined-symbol policy requires the exact reviewed input tuple"
  [[ "${public_exports}" == "${p1_public_exports}" ]] || \
    die "P1 wider undefined-symbol policy requires the exact reviewed input tuple"
  [[ "${#canonical_include_dirs[@]}" == 0 ]] || \
    die "P1 wider undefined-symbol policy requires the exact reviewed input tuple"
  package_policy=p1
  undefined_symbols="${p1_undefined_symbols}"
else
  p1_tuple_matches=0
  [[ "${package_bridge}" == "${p1_package_bridge}" ]] && \
    p1_tuple_matches=$((p1_tuple_matches + 1))
  [[ "${package_renames}" == "${p1_package_renames}" ]] && \
    p1_tuple_matches=$((p1_tuple_matches + 1))
  [[ "${public_exports}" == "${p1_public_exports}" ]] && \
    p1_tuple_matches=$((p1_tuple_matches + 1))
  if [[ "${p1_tuple_matches}" != 0 ]]; then
    die "P1 wider undefined-symbol policy requires the exact reviewed input tuple"
  fi
  for reserved_input in \
      "${d1_package_bridge}" "${d1_package_renames}" "${d1_public_exports}" \
      "${x1_package_bridge}" "${x1_package_renames}" "${x1_public_exports}" \
      "${p1_package_bridge}" "${p1_package_renames}" "${p1_public_exports}"; do
    if [[ "${package_bridge}" == "${reserved_input}" || \
          "${package_renames}" == "${reserved_input}" || \
          "${public_exports}" == "${reserved_input}" ]]; then
      die "repository package components require their exact repository-owned private root"
    fi
  done
  package_policy=base-e1b
  undefined_symbols="${base_undefined_symbols}"
fi
if [[ "${package_policy}" == d1-test-faults ]]; then
  canonical_d1_artifact_dir="$(realpath -m -- "$(project_build_dir)/d1")"
  case "${output_object}" in
    "${canonical_d1_artifact_dir}"/*)
      die "D1 test-fault object cannot target the canonical production directory"
      ;;
  esac
fi

require_command awk
require_command cmp
require_command comm
require_command env
require_command grep
require_command nm
require_command objcopy
require_command readelf
require_command sort
require_command strings
verify_toolchain

e1b_runner="$(eshkol_build_dir)/eshkol-run"
e1b_source="$(eshkol_source_dir)"
e1b_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
e1b_cc="$(tsv_value "${e1b_provenance}" cc_path)"
e1b_cxx="$(tsv_value "${e1b_provenance}" cxx_path)"
[[ -f "${undefined_symbols}" ]] || \
  die "E1B undefined-symbol allowlist not found: ${undefined_symbols}"
[[ -s "${undefined_symbols}" ]] || \
  die "E1B undefined-symbol allowlist must not be empty"
e1b_timeout_seconds="${E1B_COMPILER_TIMEOUT_SECONDS:-120}"
[[ "${e1b_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || \
  die "E1B_COMPILER_TIMEOUT_SECONDS must be a positive integer"

e1b_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-e1b.XXXXXX")"
trap 'rm -rf -- "${e1b_tmp}"' EXIT
mkdir -p "${e1b_tmp}/cache" "$(dirname -- "${output_object}")"

awk '
  NF != 1 || $1 !~ /^[A-Za-z_][A-Za-z0-9_]*$/ { bad = 1 }
  END { if (bad) exit 1 }
' "${undefined_symbols}" || \
  die "E1B undefined-symbol allowlist must be one symbol per line"
LC_ALL=C sort -u "${undefined_symbols}" \
  >"${e1b_tmp}/expected-undefined.txt"
cmp -s "${undefined_symbols}" "${e1b_tmp}/expected-undefined.txt" || \
  die "E1B undefined-symbol allowlist must already be byte-exact C-sorted unique text"

export_pattern='^et_e1b_public_[a-z0-9_]+_v1$'
awk -v pattern="${export_pattern}" '
  NF != 1 || $1 !~ pattern { bad = 1; next }
  { print $1 }
  END { if (bad) exit 1 }
' "${public_exports}" | LC_ALL=C sort -u >"${e1b_tmp}/package-exports.txt" || \
  die "E1B package exports do not match the repository-owned policy"
[[ -s "${e1b_tmp}/package-exports.txt" ]] || \
  die "E1B package export list must not be empty"
cmp -s "${public_exports}" "${e1b_tmp}/package-exports.txt" || \
  die "E1B package export allowlist must already be byte-exact C-sorted unique text"

{
  printf '%s\n' \
    et_e1b_error_predicate_v1 \
    et_e1b_error_category_v1 \
    et_e1b_error_operation_v1 \
    et_e1b_error_message_v1 \
    et_e1b_error_details_v1 \
    et_e1b_error_cause_v1
  cat "${e1b_tmp}/package-exports.txt"
} | LC_ALL=C sort -u >"${e1b_tmp}/expected-global-defined.txt"

run_compiler() {
  env -u ESHKOL_PATH \
    XDG_CACHE_HOME="${e1b_tmp}/cache" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${e1b_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${e1b_timeout_seconds}s" "${e1b_runner}" "$@"
}

include_args=(-I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native")
for include_dir in "${canonical_include_dirs[@]}"; do
  include_args+=(-I "${include_dir}")
done

(
  cd "${e1b_tmp}"
  run_compiler --strict-types --no-stdlib "${include_args[@]}" \
    --shared-lib --dump-ir --emit-depfile "${e1b_tmp}/private.d" \
    "${private_root}" \
    -o private
)
[[ -f "${e1b_tmp}/private.ll" ]] || die "E1B private IR was not emitted"
e1b_raise_attribute="$(
  awk '
    /^define .*@et-e1b-private-raise__eshkol_internal_abi/ {
      if (match($0, /#[0-9]+/)) print substr($0, RSTART, RLENGTH)
    }
  ' "${e1b_tmp}/private.ll"
)"
[[ -n "${e1b_raise_attribute}" ]] || \
  die "E1B private root omitted the fixed raise-only seam"
grep -Eq "^attributes ${e1b_raise_attribute} = .*noreturn" \
  "${e1b_tmp}/private.ll" || \
  die "E1B fixed raise-only seam is not noreturn in generated IR"

"${e1b_cc}" -c -x ir "${e1b_tmp}/private.ll" \
  -o "${e1b_tmp}/private.o"

{
  cat "${PROJECT_ROOT}/native/e1b_private_renames.txt"
  cat "${package_renames}"
} >"${e1b_tmp}/renames.txt"
objcopy --redefine-syms="${e1b_tmp}/renames.txt" \
  "${e1b_tmp}/private.o"

"${e1b_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -fstack-protector-all \
  -I "${e1b_source}/inc" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/e1b_error_consumer_bridge.c" \
  -o "${e1b_tmp}/bridge.o"

"${e1b_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${e1b_source}/inc" -I "${PROJECT_ROOT}/native" \
  -c "${package_bridge}" -o "${e1b_tmp}/package-bridge.o"

package_native_objects=()
if [[ -n "${package_native_source}" ]]; then
  package_native_objects+=("${e1b_tmp}/package-native.o")
  package_native_cflags=(
    -std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
  )
  if [[ -n "${package_native_define}" ]]; then
    package_native_cflags+=("${package_native_define}")
  fi
  "${e1b_cc}" "${package_native_cflags[@]}" \
    -c "${package_native_source}" -o "${package_native_objects[0]}"
fi

"${e1b_cxx}" -r -Wl,-Map,"${e1b_tmp}/combined.map" \
  "${e1b_tmp}/private.o" "${e1b_tmp}/bridge.o" \
  "${e1b_tmp}/package-bridge.o" "${package_native_objects[@]}" \
  -o "${e1b_tmp}/combined.raw.o"

nm -g --defined-only --format=posix "${e1b_tmp}/combined.raw.o" | \
  awk 'NR == FNR { allowed[$1] = 1; next }
       !($1 in allowed) { print $1 }' \
  "${e1b_tmp}/expected-global-defined.txt" - \
  >"${e1b_tmp}/localize-symbols.txt"

cp "${e1b_tmp}/combined.raw.o" "${e1b_tmp}/combined.o"
objcopy --localize-symbols="${e1b_tmp}/localize-symbols.txt" \
  "${e1b_tmp}/combined.o"

nm -g --defined-only --format=posix "${e1b_tmp}/combined.o" | \
  awk '{ print $1 }' | LC_ALL=C sort >"${e1b_tmp}/global-defined.txt"
cmp -s "${e1b_tmp}/expected-global-defined.txt" \
  "${e1b_tmp}/global-defined.txt" || \
  die "E1B final object differs from the exact public export allowlist"

nm -u --format=posix "${e1b_tmp}/combined.o" | awk '{ print $1 }' | \
  LC_ALL=C sort -u >"${e1b_tmp}/undefined.txt"
if ! e1b_compare_exact_undefined_allowlist \
    "${e1b_tmp}/expected-undefined.txt" "${e1b_tmp}/undefined.txt"; then
  die "E1B final object differs from the exact undefined-symbol allowlist"
fi
if grep -E 'et_e1b|e1(-internal-dispatch|_2Dinternal_2Ddispatch)|transformer(-error-(make|raise|wrap-foreign)|_2Derror_2D(make|raise|wrap_2Dforeign))' \
    "${e1b_tmp}/undefined.txt" >/dev/null; then
  die "E1B final object retains an unresolved privileged reference"
fi

readelf --wide --syms "${e1b_tmp}/combined.o" \
  >"${e1b_tmp}/readelf-symbols.txt"
if awk '
  /et_e1b_(private|consumer|box|ensure)|e1(-internal-dispatch|_2Dinternal_2Ddispatch)|transformer(-error-(make|raise|wrap-foreign)|_2Derror_2D(make|raise|wrap_2Dforeign))/ {
    if ($5 != "LOCAL" && $7 != "UND") bad = 1
  }
  END { exit bad ? 0 : 1 }
' "${e1b_tmp}/readelf-symbols.txt"; then
  die "E1B privileged definition was not localized"
fi
for privileged in \
  transformer-error-make transformer-error-raise \
  transformer-error-wrap-foreign e1-internal-dispatch \
  et_e1b_consumer_raise_v1 et_e1b_private_raise_cabi_v1; do
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${privileged}$" \
    "${e1b_tmp}/readelf-symbols.txt" >/dev/null || \
    die "E1B required privileged definition is not local: ${privileged}"
done
if [[ "${package_policy}" == d1 || "${package_policy}" == d1-test-faults ]]; then
  for privileged in \
    et_d1_checked_write_new_v1 \
    et_e1b_private_d1_token_corpus_write_cabi_v1 \
    et_e1b_private_d1_token_corpus_validate_cabi_v1; do
    grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${privileged}$" \
      "${e1b_tmp}/readelf-symbols.txt" >/dev/null || \
      die "D1 required privileged definition is not local: ${privileged}"
  done
fi

evidence_dir="${output_object}.evidence"
temporary_output="${output_object}.tmp.$$"
rm -rf -- "${evidence_dir}.tmp.$$"
mkdir -p "${evidence_dir}.tmp.$$"
sed "s#${e1b_tmp}#<E1B_BUILD>#g" "${e1b_tmp}/private.d" \
  >"${evidence_dir}.tmp.$$/private.d"
sed "s#${e1b_tmp}#<E1B_BUILD>#g" "${e1b_tmp}/combined.map" \
  >"${evidence_dir}.tmp.$$/link.map"
nm -a "${e1b_tmp}/combined.o" >"${evidence_dir}.tmp.$$/nm.txt"
cp "${e1b_tmp}/readelf-symbols.txt" \
  "${evidence_dir}.tmp.$$/readelf-symbols.txt"
cp "${e1b_tmp}/global-defined.txt" \
  "${evidence_dir}.tmp.$$/global-defined.txt"
cp "${e1b_tmp}/package-exports.txt" \
  "${evidence_dir}.tmp.$$/package-exports.txt"
cp "${e1b_tmp}/undefined.txt" \
  "${evidence_dir}.tmp.$$/undefined.txt"
cp "${e1b_tmp}/expected-undefined.txt" \
  "${evidence_dir}.tmp.$$/expected-undefined.txt"
{
  printf 'package_policy\t%s\n' "${package_policy}"
  printf 'allowlist\t%s\n' "$(basename -- "${undefined_symbols}")"
  printf 'llvm_version\t%s\n' "$(tsv_value "${e1b_provenance}" llvm_version)"
  printf 'cc_version\t%s\n' "$(tsv_value "${e1b_provenance}" cc_version)"
  printf 'cxx_version\t%s\n' "$(tsv_value "${e1b_provenance}" cxx_version)"
} >"${evidence_dir}.tmp.$$/allowlist-provenance.tsv"
strings "${e1b_tmp}/combined.o" >"${evidence_dir}.tmp.$$/strings.txt"
cp "${e1b_tmp}/combined.o" "${temporary_output}"
rm -rf -- "${evidence_dir}"
mv "${evidence_dir}.tmp.$$" "${evidence_dir}"
mv "${temporary_output}" "${output_object}"

printf 'E1B combined consumer object: %s\n' "${output_object}"
