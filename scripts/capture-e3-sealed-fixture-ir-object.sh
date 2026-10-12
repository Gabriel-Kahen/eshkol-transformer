#!/usr/bin/env bash
# One use of the ROOT-accepted J IR. This is not the localized E1B package.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -eq 4 ]] || die "usage: $0 J_RUN J_ROOT_REVIEW CLAIM_ROOT FRESH_OUTPUT"
for input in "$1" "$2" "$3" "$4"; do
  [[ "${input}" == /* && "$(realpath -m -- "${input}")" == "${input}" ]] || \
    die "capture paths must be absolute, canonical and free of symlinks"
done
run=$1
review=$2
claim_root=$3
out=$4
[[ -d "${claim_root}" && ! -L "${claim_root}" ]] || \
  die "claim root must be an existing directory"
claim="${claim_root}/e3-section9-j-ir-object-fb31-fe9"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "IR-object output must be fresh"
for protected in "${run}" "${run}.claim" "${review}"; do
  case "${claim_root}" in
    "${protected}"|"${protected}/"*) die "claim root overlaps accepted J evidence" ;;
  esac
done
for protected in "${run}" "${run}.claim" "${review}" "${claim_root}"; do
  case "${out}" in
    "${protected}"|"${protected}/"*) die "IR-object output overlaps accepted evidence or claim root" ;;
  esac
done
mkdir -p -- "${out}"
finalize() {
  local status=$? post=0
  trap - EXIT
  set +e
  python3 -B "${PROJECT_ROOT}/scripts/verify-e3-sealed-fixture-j.py" \
    "${run}" "${review}" >"${out}/input-after.stdout" \
    2>"${out}/input-after.stderr"
  post=$?
  printf '%s\n' "${post}" >"${out}/input-after.exit"
  if [[ "${status}" == 0 && "${post}" != 0 ]]; then status=1; fi
  "${PROJECT_ROOT}/scripts/seal-e3-ir-object-capture.sh" "${out}" "${status}" || status=1
  exit "${status}"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec >"${out}/capture.stdout" 2>"${out}/capture.stderr"

set +e
python3 -B "${PROJECT_ROOT}/scripts/verify-e3-sealed-fixture-j.py" \
  "${run}" "${review}" >"${out}/input-before.stdout" \
  2>"${out}/input-before.stderr"
before_status=$?
set -e
printf '%s\n' "${before_status}" >"${out}/input-before.exit"
[[ "${before_status}" == 0 ]] || die "accepted J input preflight failed"
printf 'ir-derived-private-object\n' >"${out}/evidence-kind.txt"
printf 'final-E1B-OUTPUT.o.evidence-open\n' >"${out}/package-status.txt"
cp -- "${run}/source-once/source-pin.txt" "${out}/j-source-pin.txt"
cp -- "${run}/source-once/runtime-pin.txt" "${out}/j-runtime-pin.txt"
sha256sum "${run}/source-once/compiler/private.ll" \
  "${run}/source-once/compiler/libprivate.so" >"${out}/j-input-hashes.txt"
sha256sum "${review}/SHA256SUMS-ROOT" \
  "${run}.claim/attempt/RUN-SHA256SUMS" >"${out}/j-seal-hashes.txt"

verify_toolchain
require_command nm
require_command readelf
require_command timeout
(cd / && sha256sum --quiet -c "${run}/source-once/toolchain.sha256")
[[ "$(git -C "$(eshkol_source_dir)" rev-parse HEAD)" == \
   "$(cat "${run}/source-once/runtime-pin.txt")" ]] || \
  die "active runtime differs from J"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse fb31fac32530e0ff9b00ed3458e9080eaba9643a^{tree})" == \
   5ca77e78a6f798f677a0b23a14e763d957f830bd ]] || \
  die "J source tree is unavailable"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
[[ -x "${cc}" ]] || die "pinned C compiler is absent"
"${cc}" --version >"${out}/compiler-version.stdout" \
  2>"${out}/compiler-version.stderr"
"${PROJECT_ROOT}/scripts/claim-e3-sealed-fixture-ir-object.sh" \
  "${claim_root}" "${out}"
printf '%q ' env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
  -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES \
  -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
  -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${cc}" -c -x ir "${run}/source-once/compiler/private.ll" \
  -o "${out}/private.o" >"${out}/compiler.argv"
printf '\n' >>"${out}/compiler.argv"
set +e
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
  -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES \
  -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
  -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${cc}" -c -x ir "${run}/source-once/compiler/private.ll" \
  -o "${out}/private.o" >"${out}/compiler.stdout" 2>"${out}/compiler.stderr"
compile_status=$?
set -e
printf '%s\n' "${compile_status}" >"${out}/compiler.exit"
[[ "${compile_status}" == 0 && -s "${out}/private.o" ]] || \
  die "J IR object emission failed; rejected output retained"

nm -a "${out}/private.o" >"${out}/private.nm-all.txt" \
  2>"${out}/private.nm-all.stderr"
nm -g --defined-only --format=posix "${out}/private.o" \
  >"${out}/private.global-defined.posix.txt" \
  2>"${out}/private.global-defined.stderr"
nm -u --format=posix "${out}/private.o" \
  >"${out}/private.undefined.posix.txt" \
  2>"${out}/private.undefined.stderr"
readelf --wide --syms "${out}/private.o" \
  >"${out}/private.readelf-symbols.txt" \
  2>"${out}/private.readelf-symbols.stderr"
nm -D "${run}/source-once/compiler/libprivate.so" \
  >"${out}/j-shared.nm-dynamic.txt" 2>"${out}/j-shared.nm-dynamic.stderr"
readelf --wide --dyn-syms "${run}/source-once/compiler/libprivate.so" \
  >"${out}/j-shared.readelf-dynamic.txt" \
  2>"${out}/j-shared.readelf-dynamic.stderr"
