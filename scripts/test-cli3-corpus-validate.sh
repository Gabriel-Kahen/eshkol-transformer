#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command ar
require_command python3
require_command sed
require_command timeout
verify_toolchain

cli3_archive="$(project_build_dir)/d1/libeshkol_transformer_d1.a"
[[ -r "${cli3_archive}" ]] || die "canonical D1 archive is missing; run scripts/build-d1.sh first"
[[ "$(ar t "${cli3_archive}")" == stdlib.o ]] || die "unexpected D1 archive members"
cli3_evidence="${cli3_archive}.evidence"
cmp -s "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
  "${cli3_evidence}/package-exports.txt" || die "D1 archive public exports differ"
cmp -s "${PROJECT_ROOT}/native/d1_e1b_undefined_symbols.txt" \
  "${cli3_evidence}/undefined.txt" || die "D1 archive undefined symbols differ"

cli3_tmp=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-cli3.XXXXXX")
trap 'rm -rf -- "${cli3_tmp}"' EXIT
cli3_public="${cli3_tmp}/public"
mkdir -p "${cli3_public}/transformer" "${cli3_tmp}/cache"
for module in error_consumer error_public; do
  cp "${PROJECT_ROOT}/lib/transformer/${module}.esk" \
    "${cli3_public}/transformer/${module}.esk"
done

# D1 test-d1.sh uses these exact markers because importing the installed D1+D2
# facade emits D2 references even for a D1-only consumer.
cli3_begin=';;; D1 facade component begin.'
cli3_end=';;; D1 facade component end.'
[[ "$(grep -Fxc "${cli3_begin}" "${PROJECT_ROOT}/lib/transformer/data.esk")" == 1 ]]
[[ "$(grep -Fxc "${cli3_end}" "${PROJECT_ROOT}/lib/transformer/data.esk")" == 1 ]]
{
  printf '%s\n\n' ';;; D1-only staged facade from production wrappers.'
  printf '%s\n\n' '(require transformer.error_consumer)'
  printf '%s\n' '(provide token-corpus-write!'
  printf '%s\n' \
    '         token-corpus-validate' \
    '         token-corpus-summary-shard-count' \
    '         token-corpus-summary-total-tokens' \
    '         token-corpus-summary-vocab-size' \
    '         token-corpus-summary-shard-token-limit' \
    '         token-corpus-summary-tokenizer-fingerprint' \
    '         token-corpus-summary-total-shard-bytes)'
  sed -n "/^${cli3_begin//./\\.}$/,/^${cli3_end//./\\.}$/p" \
    "${PROJECT_ROOT}/lib/transformer/data.esk"
} >"${cli3_public}/transformer/data.esk"
[[ "$(grep -Ec '^\(define \(token-corpus-' \
  "${cli3_public}/transformer/data.esk")" == 8 ]] || \
  die "D1-only facade did not extract eight public wrappers"
if grep -F 'et_e1b_public_d2_' "${cli3_public}/transformer/data.esk" >/dev/null; then
  die "D1-only facade retained D2 references"
fi

cli3_runner="$(eshkol_build_dir)/eshkol-run"
cli3_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
cli3_compile() {
  local source=$1 output=$2
  env XDG_CACHE_HOME="${cli3_tmp}/cache" ESHKOL_CXX_COMPILER="${cli3_cxx}" \
    ESHKOL_LIB_DIR="${cli3_public}" \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
      "${cli3_runner}" --strict-types --no-stdlib \
      -I "${cli3_public}" -L "$(dirname -- "${cli3_archive}")" \
      --lib eshkol_transformer_d1 "${source}" -o "${output}"
}

cli3_compile "${PROJECT_ROOT}/cli/eshkol-transformer.esk" \
  "${cli3_tmp}/eshkol-transformer"
cli3_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${cli3_tmp}/corpus-tool"
CLI3_CORPUS_VALIDATE_BINARY="${cli3_tmp}/eshkol-transformer" \
  D1_CORPUS_TOOL="${cli3_tmp}/corpus-tool" \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -v -s "${PROJECT_ROOT}/tests/cli3" \
    -p 'test_corpus_validate.py'
