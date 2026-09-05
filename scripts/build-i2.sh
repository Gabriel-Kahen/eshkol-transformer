#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar awk cmp nm readelf; do
  require_command "${command}"
done

artifact_dir="${1:-$(project_build_dir)/i2}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.i2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
case "${mode}" in
  normal) ;;
  sanitize-test)
    cflags+=(
      -DET_F32_TENSOR_TESTING
      -fsanitize=address,undefined -fno-omit-frame-pointer
    )
    ;;
  *) die "unknown I2 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/f32_tensor.c" \
  -o "${temporary_dir}/f32_tensor.o"
nm -g --defined-only --format=posix "${temporary_dir}/f32_tensor.o" | \
  awk '{ print $1 }' | LC_ALL=C sort \
  >"${temporary_dir}/f32-defined.txt"
nm -u --format=posix "${temporary_dir}/f32_tensor.o" | \
  awk '{ print $1 }' | LC_ALL=C sort -u \
  >"${temporary_dir}/f32-undefined.txt"
readelf --wide --syms "${temporary_dir}/f32_tensor.o" \
  >"${temporary_dir}/f32-readelf-symbols.txt"
if [[ "${mode}" == normal ]]; then
  cmp "${PROJECT_ROOT}/native/f32_tensor_defined_symbols.txt" \
    "${temporary_dir}/f32-defined.txt"
  cmp "${PROJECT_ROOT}/native/f32_tensor_undefined_symbols.txt" \
    "${temporary_dir}/f32-undefined.txt"
fi
awk '
  NR == FNR { public[$1] = 1; next }
  $5 == "GLOBAL" && $7 != "UND" && $8 ~ /^et_f32_/ {
    seen[$8] = 1
    if (($8 in public) && $6 != "DEFAULT") bad = 1
    if (!($8 in public) && $6 != "HIDDEN") bad = 1
  }
  END {
    for (symbol in public) if (!(symbol in seen)) bad = 1
    exit bad ? 1 : 0
  }
' "${PROJECT_ROOT}/native/f32_tensor_public_symbols.txt" \
  "${temporary_dir}/f32-readelf-symbols.txt" || \
  die "I2 standalone object visibility differs from the fixed ABI policy"
ar rcsD "${temporary_dir}/libeshkol_transformer_f32.a" \
  "${temporary_dir}/f32_tensor.o"

if [[ "${mode}" == normal ]]; then
  E1B_COMPILER_TIMEOUT_SECONDS="${I2_COMPILER_TIMEOUT_SECONDS:-360}" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/i2_wave2_root.esk" \
    "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/i2_wave2_private_renames.txt" \
    "${PROJECT_ROOT}/native/i2_wave2_public_exports.txt" \
    "${temporary_dir}/i2_wave2.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/src"
  cmp "${PROJECT_ROOT}/native/i2_wave2_defined_symbols.txt" \
    "${temporary_dir}/i2_wave2.o.evidence/global-defined.txt"
  ar rcsD "${temporary_dir}/libeshkol_transformer_wave2.a" \
    "${temporary_dir}/i2_wave2.o"
fi

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/f32_tensor.o.evidence.tmp"
mkdir -p "${artifact_dir}/f32_tensor.o.evidence.tmp"
cp "${temporary_dir}/f32-defined.txt" \
  "${artifact_dir}/f32_tensor.o.evidence.tmp/global-defined.txt"
cp "${temporary_dir}/f32-undefined.txt" \
  "${artifact_dir}/f32_tensor.o.evidence.tmp/undefined.txt"
cp "${temporary_dir}/f32-readelf-symbols.txt" \
  "${artifact_dir}/f32_tensor.o.evidence.tmp/readelf-symbols.txt"
{
  printf 'llvm_version\t%s\n' "$(tsv_value "${provenance}" llvm_version)"
  printf 'cc_version\t%s\n' "$(tsv_value "${provenance}" cc_version)"
} >"${artifact_dir}/f32_tensor.o.evidence.tmp/provenance.tsv"
mv -f "${temporary_dir}/f32_tensor.o" "${artifact_dir}/f32_tensor.o"
mv -f "${temporary_dir}/libeshkol_transformer_f32.a" \
  "${artifact_dir}/libeshkol_transformer_f32.a"
if [[ "${mode}" == normal ]]; then
  rm -rf -- "${artifact_dir}/i2_wave2.o.evidence"
  mv "${temporary_dir}/i2_wave2.o.evidence" \
    "${artifact_dir}/i2_wave2.o.evidence"
  mv -f "${temporary_dir}/i2_wave2.o" "${artifact_dir}/i2_wave2.o"
  mv -f "${temporary_dir}/libeshkol_transformer_wave2.a" \
    "${artifact_dir}/libeshkol_transformer_wave2.a"
fi
rm -rf -- "${artifact_dir}/f32_tensor.o.evidence"
mv "${artifact_dir}/f32_tensor.o.evidence.tmp" \
  "${artifact_dir}/f32_tensor.o.evidence"
printf 'built I2 carrier-local dense CPU-f32 substrate: %s\n' \
  "${artifact_dir}/libeshkol_transformer_f32.a"
if [[ "${mode}" == normal ]]; then
  printf 'built source-composed Wave 2 aggregate: %s\n' \
    "${artifact_dir}/libeshkol_transformer_wave2.a"
fi
