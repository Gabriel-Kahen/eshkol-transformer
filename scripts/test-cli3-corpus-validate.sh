#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ "$#" != 1 || "$1" != /* ]]; then
  printf 'usage: %s /absolute/new/evidence-directory\n' "$0" >&2
  exit 64
fi
require_command realpath
cli3_evidence_dir=$1
cli3_project_real=$(realpath -m -- "${PROJECT_ROOT}")
cli3_evidence_real=$(realpath -m -- "${cli3_evidence_dir}")
if [[ "${cli3_evidence_real}" == "${cli3_project_real}" ||
      "${cli3_evidence_real}" == "${cli3_project_real}/"* ]]; then
  printf 'error: evidence directory must be outside the project checkout: %s\n' \
    "${cli3_evidence_dir}" >&2
  exit 64
fi
if [[ -e "${cli3_evidence_dir}" || -L "${cli3_evidence_dir}" ]]; then
  printf 'error: evidence directory already exists: %s\n' "${cli3_evidence_dir}" >&2
  exit 64
fi
mkdir -- "${cli3_evidence_dir}"
cli3_finish() {
  local status=$?
  trap - EXIT
  git -C "${PROJECT_ROOT}" rev-parse HEAD \
    >"${cli3_evidence_dir}/final-project-head.txt" 2>"${cli3_evidence_dir}/final-git.stderr" || status=1
  git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}' \
    >"${cli3_evidence_dir}/final-project-tree.txt" 2>>"${cli3_evidence_dir}/final-git.stderr" || status=1
  git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all \
    >"${cli3_evidence_dir}/final-git-status.txt" 2>>"${cli3_evidence_dir}/final-git.stderr" || status=1
  if [[ -f "${cli3_evidence_dir}/initial-project-head.txt" ]]; then
    cmp -s "${cli3_evidence_dir}/initial-project-head.txt" \
      "${cli3_evidence_dir}/final-project-head.txt" || status=1
    cmp -s "${cli3_evidence_dir}/initial-project-tree.txt" \
      "${cli3_evidence_dir}/final-project-tree.txt" || status=1
  fi
  [[ ! -s "${cli3_evidence_dir}/final-git-status.txt" ]] || status=1
  printf '%s\n' "${status}" >"${cli3_evidence_dir}/overall.status"
  exit "${status}"
}
trap cli3_finish EXIT
mkdir -p "${cli3_evidence_dir}/steps" "${cli3_evidence_dir}/artifacts/cache"
cli3_tmp="${cli3_evidence_dir}/artifacts"

cli3_step() {
  local name=$1 status
  shift
  if ( "$@" ) >"${cli3_evidence_dir}/steps/${name}.stdout" \
      2>"${cli3_evidence_dir}/steps/${name}.stderr"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "${status}" >"${cli3_evidence_dir}/steps/${name}.status"
  return "${status}"
}

cli3_optional_tsv_value() {
  local file=$1 key=$2
  if [[ -r "${file}" ]]; then
    awk -F '\t' -v key="${key}" \
      '$1 == key { value = $2; count++ } END { if (count == 1) print value; else print "missing" }' \
      "${file}"
  else
    printf '%s\n' missing
  fi
}

cli3_hash_or_missing() {
  if [[ -r "$1" ]]; then
    sha256sum "$1" | awk '{ print $1 }'
  else
    printf '%s\n' missing
  fi
}

cli3_capture_metadata() {
  local runner provenance clean
  runner="$(eshkol_build_dir)/eshkol-run"
  provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
  git -C "${PROJECT_ROOT}" rev-parse HEAD \
    >"${cli3_evidence_dir}/initial-project-head.txt"
  git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}' \
    >"${cli3_evidence_dir}/initial-project-tree.txt"
  git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all \
    >"${cli3_evidence_dir}/initial-git-status.txt"
  clean=false
  [[ ! -s "${cli3_evidence_dir}/initial-git-status.txt" ]] && clean=true
  {
    printf 'format_version\t1\n'
    printf 'project_head\t%s\n' "$(cat "${cli3_evidence_dir}/initial-project-head.txt")"
    printf 'project_tree\t%s\n' "$(cat "${cli3_evidence_dir}/initial-project-tree.txt")"
    printf 'project_clean\t%s\n' "${clean}"
    printf 'lock_sha256\t%s\n' "$(cli3_hash_or_missing "${LOCK_FILE}")"
    printf 'lock_eshkol_repository\t%s\n' "$(lock_value eshkol_repository)"
    printf 'lock_eshkol_commit\t%s\n' "$(lock_value eshkol_commit)"
    printf 'compiler_path\t%s\n' "${runner}"
    printf 'compiler_sha256\t%s\n' "$(cli3_hash_or_missing "${runner}")"
    printf 'compiler_provenance_path\t%s\n' "${provenance}"
    printf 'compiler_provenance_sha256\t%s\n' "$(cli3_hash_or_missing "${provenance}")"
    printf 'compiler_provenance_repository\t%s\n' \
      "$(cli3_optional_tsv_value "${provenance}" eshkol_repository)"
    printf 'compiler_provenance_commit\t%s\n' \
      "$(cli3_optional_tsv_value "${provenance}" eshkol_commit)"
    printf 'compiler_provenance_binary_sha256\t%s\n' \
      "$(cli3_optional_tsv_value "${provenance}" eshkol_binary_sha256)"
    printf 'compiler_provenance_llvm_version\t%s\n' \
      "$(cli3_optional_tsv_value "${provenance}" llvm_version)"
  } >"${cli3_evidence_dir}/metadata.tsv"
}

cli3_check_project_pin() {
  [[ "$(tsv_value "${cli3_evidence_dir}/metadata.tsv" project_clean)" == true ]] || \
    die "project checkout is dirty; see initial-git-status.txt"
  [[ "$(cat "${cli3_evidence_dir}/initial-project-head.txt")" == \
     "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" ]] || die "project HEAD changed"
  [[ "$(cat "${cli3_evidence_dir}/initial-project-tree.txt")" == \
     "$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')" ]] || \
    die "project tree changed"
  [[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]] || \
    die "project checkout changed after metadata capture"
  [[ "$(tsv_value "${cli3_evidence_dir}/metadata.tsv" lock_sha256)" == \
     "$(cli3_hash_or_missing "${LOCK_FILE}")" ]] || die "toolchain lock changed"
}

cli3_check_runtime_pin() {
  local metadata="${cli3_evidence_dir}/metadata.tsv" runner provenance
  runner="$(tsv_value "${metadata}" compiler_path)"
  provenance="$(tsv_value "${metadata}" compiler_provenance_path)"
  [[ "$(tsv_value "${metadata}" compiler_sha256)" != missing ]] || \
    die "compiler was unavailable at metadata capture"
  [[ "$(tsv_value "${metadata}" compiler_sha256)" == \
     "$(cli3_hash_or_missing "${runner}")" ]] || die "compiler changed"
  [[ "$(tsv_value "${metadata}" compiler_provenance_sha256)" == \
     "$(cli3_hash_or_missing "${provenance}")" ]] || die "compiler provenance changed"
  [[ "$(tsv_value "${metadata}" compiler_sha256)" == \
     "$(tsv_value "${metadata}" compiler_provenance_binary_sha256)" ]] || \
    die "compiler hash differs from provenance"
  [[ "$(tsv_value "${metadata}" lock_eshkol_repository)" == \
     "$(tsv_value "${metadata}" compiler_provenance_repository)" ]] || \
    die "compiler repository differs from lock"
  [[ "$(tsv_value "${metadata}" lock_eshkol_commit)" == \
     "$(tsv_value "${metadata}" compiler_provenance_commit)" ]] || \
    die "compiler revision differs from lock"
  cli3_check_project_pin
}

require_command ar
require_command awk
require_command cat
require_command cmp
require_command cp
require_command grep
require_command mkdir
require_command mktemp
require_command nm
require_command python3
require_command readelf
require_command sed
require_command sha256sum
require_command sort
require_command timeout
require_command wc
cli3_step capture-metadata cli3_capture_metadata
cli3_step check-project-pin cli3_check_project_pin
cli3_step verify-toolchain verify_toolchain
cli3_step check-runtime-pin cli3_check_runtime_pin

cli3_archive="$(project_build_dir)/d1/libeshkol_transformer_d1.a"
# Rebuild from this exact tree. An older archive can satisfy the sidecar manifests
# while carrying an incompatible Eshkol runtime ABI.
cli3_step build-d1 "${PROJECT_ROOT}/scripts/build-d1.sh" \
  "$(dirname -- "${cli3_archive}")"
[[ -r "${cli3_archive}" ]] || die "fresh D1 archive is missing"
cli3_evidence="${cli3_archive}.evidence"
mkdir -p "${cli3_tmp}/d1"
cli3_step retain-d1 cp -a "${cli3_archive}" "${cli3_evidence}" \
  "${cli3_tmp}/d1/"
[[ "$(ar t "${cli3_archive}")" == stdlib.o ]] || die "unexpected D1 archive members"
for evidence in package-exports.txt global-defined.txt undefined.txt \
    expected-undefined.txt readelf-symbols.txt private.d; do
  [[ -r "${cli3_evidence}/${evidence}" ]] || die "missing D1 evidence: ${evidence}"
done
cmp -s "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
  "${cli3_evidence}/package-exports.txt" || die "D1 archive public exports differ"
cmp -s "${PROJECT_ROOT}/native/d1_e1b_undefined_symbols.txt" \
  "${cli3_evidence}/undefined.txt" || die "D1 archive undefined symbols differ"
cmp -s "${cli3_evidence}/undefined.txt" \
  "${cli3_evidence}/expected-undefined.txt" || die "D1 undefined evidence differs"

{
  printf '%s\n' \
    et_e1b_error_category_v1 \
    et_e1b_error_cause_v1 \
    et_e1b_error_details_v1 \
    et_e1b_error_message_v1 \
    et_e1b_error_operation_v1 \
    et_e1b_error_predicate_v1
  cat "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt"
} | LC_ALL=C sort -u >"${cli3_tmp}/expected-global-defined.txt"
[[ "$(wc -l <"${cli3_tmp}/expected-global-defined.txt")" == 14 ]] || \
  die "D1 archive must define six E1B and eight D1 symbols"
cmp -s "${cli3_tmp}/expected-global-defined.txt" \
  "${cli3_evidence}/global-defined.txt" || die "D1 global evidence differs"
nm -g --defined-only --format=posix "${cli3_archive}" | \
  awk '$2 ~ /^[A-Za-z]$/ { print $1 }' | LC_ALL=C sort -u \
  >"${cli3_tmp}/actual-nm-defined.txt"
cmp -s "${cli3_tmp}/expected-global-defined.txt" \
  "${cli3_tmp}/actual-nm-defined.txt" || die "D1 actual nm symbols differ"
readelf --wide --symbols "${cli3_archive}" | \
  awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }' | LC_ALL=C sort -u \
  >"${cli3_tmp}/actual-readelf-defined.txt"
cmp -s "${cli3_tmp}/expected-global-defined.txt" \
  "${cli3_tmp}/actual-readelf-defined.txt" || die "D1 actual ELF symbols differ"
for source in src/eshkol_transformer/token_shard.esk \
    lib/transformer/error_core.esk lib/transformer/error_internal.esk; do
  grep -F "${source}" "${cli3_evidence}/private.d" >/dev/null || \
    die "fresh D1 artifact omitted ${source}"
done
cli3_public="${cli3_tmp}/public"
mkdir -p "${cli3_public}/transformer" "${cli3_tmp}/cache"
for module in error_consumer error_public cli3_corpus_validate; do
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
  (
    cd -- "${cli3_public}"
    env XDG_CACHE_HOME="${cli3_tmp}/cache" ESHKOL_CXX_COMPILER="${cli3_cxx}" \
      ESHKOL_LIB_DIR="${cli3_public}" \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
        "${cli3_runner}" --strict-types --no-stdlib \
        -I "${cli3_public}" -L "$(dirname -- "${cli3_archive}")" \
        --lib eshkol_transformer_d1 --emit-depfile "${output}.d" \
        "${source}" -o "${output}"
  )
  [[ -s "${output}.d" ]] || die "CLI3 depfile missing for ${source}"
  for public_module in data error_consumer; do
    grep -F "${cli3_public}/transformer/${public_module}.esk" \
      "${output}.d" >/dev/null || die "CLI3 did not import staged ${public_module}"
  done
  for forbidden in \
      "${PROJECT_ROOT}/lib/transformer/data.esk" \
      "${PROJECT_ROOT}/src/eshkol_transformer/token_shard.esk" \
      "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
      'et_e1b_public_d2_'; do
    if grep -F "${forbidden}" "${output}.d" >/dev/null; then
      die "CLI3 depfile crossed D1 public boundary: ${forbidden}"
    fi
  done
}

cli3_step compile-command cli3_compile \
  "${PROJECT_ROOT}/cli/eshkol-transformer.esk" \
  "${cli3_tmp}/eshkol-transformer"
grep -F "${cli3_public}/transformer/cli3_corpus_validate.esk" \
  "${cli3_tmp}/eshkol-transformer.d" >/dev/null || \
  die "standalone command did not import the shared public D1 handler"
cli3_step compile-corpus-writer cli3_compile \
  "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${cli3_tmp}/corpus-tool"
cli3_step compile-newline-writer cli3_compile \
  "${PROJECT_ROOT}/tests/cli3/newline_corpus_writer.esk" \
  "${cli3_tmp}/newline-corpus-writer"
cli3_step command-tests env \
  CLI3_CORPUS_VALIDATE_BINARY="${cli3_tmp}/eshkol-transformer" \
  D1_CORPUS_TOOL="${cli3_tmp}/corpus-tool" \
  CLI3_NEWLINE_WRITER="${cli3_tmp}/newline-corpus-writer" \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -v -s "${PROJECT_ROOT}/tests/cli3" \
    -p 'test_corpus_validate.py'
