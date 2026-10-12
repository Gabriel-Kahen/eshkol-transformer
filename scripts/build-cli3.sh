#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp cp grep mkdir nm python3; do
  require_command "${command}"
done

artifact_dir="${1:-$(project_build_dir)/cli3}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.cli3-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT
public_dir="${temporary_dir}/public"
mkdir -p "${public_dir}/transformer"
python3 "${PROJECT_ROOT}/scripts/stage-cli3-d1-readonly.py" \
  "${PROJECT_ROOT}/lib/transformer/data.esk" \
  "${public_dir}/transformer/data.esk"
cp "${PROJECT_ROOT}/lib/transformer/error_consumer.esk" \
  "${public_dir}/transformer/error_consumer.esk"

E1B_COMPILER_TIMEOUT_SECONDS="${CLI3_COMPILER_TIMEOUT_SECONDS:-900}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/cli3_generate_root.esk" \
    "${PROJECT_ROOT}/native/cli3_generate_package_bridge.c" \
    "${PROJECT_ROOT}/native/cli3_generate_private_renames.txt" \
    "${PROJECT_ROOT}/native/cli3_generate_public_exports.txt" \
    "${temporary_dir}/cli3.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t2/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/internal/d2/lib" \
    "${PROJECT_ROOT}/internal/e3/lib" \
    "${PROJECT_ROOT}/src"

ar rcsD "${temporary_dir}/libeshkol_transformer_cli3.a" \
  "${temporary_dir}/cli3.o"
cmp "${PROJECT_ROOT}/native/cli3_generate_defined_symbols.txt" \
  "${temporary_dir}/cli3.o.evidence/global-defined.txt"

runner="$(eshkol_build_dir)/eshkol-run"
cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
mkdir -p "${temporary_dir}/app-cache"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${temporary_dir}/app-cache" \
  ESHKOL_LIB_DIR="${public_dir}" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  "${runner}" --strict-types --no-stdlib -I "${public_dir}" \
    -L "${temporary_dir}" --lib eshkol_transformer_cli3 \
    --emit-depfile "${temporary_dir}/eshkol-transformer.d" \
    "${PROJECT_ROOT}/src/eshkol_transformer/cli.esk" \
    -o "${temporary_dir}/eshkol-transformer"
grep -F "${public_dir}/transformer/data.esk" \
  "${temporary_dir}/eshkol-transformer.d" >/dev/null || \
  die "CLI3 app did not import the staged read-only D1 facade"
grep -F "${public_dir}/transformer/error_consumer.esk" \
  "${temporary_dir}/eshkol-transformer.d" >/dev/null || \
  die "CLI3 app did not import the staged error facade"
if grep -F "${PROJECT_ROOT}/lib/transformer/data.esk" \
    "${temporary_dir}/eshkol-transformer.d" >/dev/null; then
  die "CLI3 app imported the full D1/D2 facade"
fi
nm -u "${temporary_dir}/eshkol-transformer" \
  >"${temporary_dir}/eshkol-transformer.undefined.txt"
if grep -E 'et_e1b_public_(d1_token_corpus_write|d2_)' \
    "${temporary_dir}/eshkol-transformer.undefined.txt" >/dev/null; then
  die "CLI3 app retained a D1 writer or D2 public dependency"
fi

mkdir -p "${artifact_dir}"
cp "${public_dir}/transformer/data.esk" "${artifact_dir}/d1-readonly-data.esk"
cp "${temporary_dir}/eshkol-transformer.d" "${artifact_dir}/eshkol-transformer.d"
rm -rf -- "${artifact_dir}/cli3.o.evidence"
mv "${temporary_dir}/cli3.o.evidence" "${artifact_dir}/cli3.o.evidence"
mv -f "${temporary_dir}/cli3.o" "${artifact_dir}/cli3.o"
mv -f "${temporary_dir}/libeshkol_transformer_cli3.a" \
  "${artifact_dir}/libeshkol_transformer_cli3.a"
mv -f "${temporary_dir}/eshkol-transformer" \
  "${artifact_dir}/eshkol-transformer"
printf 'built source-composed CLI3 C2 successor: %s\n' \
  "${artifact_dir}/libeshkol_transformer_cli3.a"
