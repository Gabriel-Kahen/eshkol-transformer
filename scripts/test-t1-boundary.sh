#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar awk cmp grep ldd nm readelf sed strings timeout tr xargs; do
  require_command "${command}"
done

t1_boundary_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-t1-boundary.XXXXXX")"
trap 'rm -rf -- "${t1_boundary_tmp}"' EXIT
t1_boundary_artifact="$(project_build_dir)/t1/wave1.o"
t1_boundary_archive="$(project_build_dir)/t1/libeshkol_transformer_wave1.a"
t1_boundary_evidence="${t1_boundary_artifact}.evidence"
t1_boundary_runner="$(eshkol_build_dir)/eshkol-run"
t1_boundary_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
t1_boundary_cc="$(tsv_value "${t1_boundary_provenance}" cc_path)"
t1_boundary_cxx="$(tsv_value "${t1_boundary_provenance}" cxx_path)"
t1_boundary_timeout="${T1_COMPILER_TIMEOUT_SECONDS:-300}"
[[ "${t1_boundary_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "T1_COMPILER_TIMEOUT_SECONDS must be a positive integer"
[[ -r "${t1_boundary_artifact}" && -r "${t1_boundary_archive}" ]] || \
  die "canonical Wave 1 aggregate must be built before its boundary gate"

provide_words() {
  awk '
    /^\(provide / { active = 1 }
    active {
      print
      line = $0
      opens += gsub(/\(/, "(", line)
      line = $0
      closes += gsub(/\)/, ")", line)
      if (opens == closes) exit
    }
  ' "$1" | \
    tr '()' '  ' | tr -s '[:space:]' '\n' | \
    grep -v -e '^provide$' -e '^$' | xargs
}

check_public_module() {
  local module=$1
  local expected=$2
  local source="${PROJECT_ROOT}/lib/${module}.esk"
  [[ "$(provide_words "${source}")" == "${expected}" ]] || \
    die "${module} public provide surface drifted"
  if [[ "${module}" != transformer/error_consumer ]]; then
    grep -F '(require transformer.error_consumer)' "${source}" >/dev/null || \
      die "${module} must consume only the public error facade"
  fi
  if grep -E '\(require transformer\.(error_internal|error_core)\)|et-e1b-private-raise|c1-persistence-policy-internal|t1-private-|t1-shell-|x1-private-|p1-native-|d1-token-(checked|sha|corpus-.*impl)' \
      "${source}" >/dev/null; then
    die "${module} public source contains a private capability"
  fi
}

check_public_module transformer/error_consumer \
  'transformer-error? transformer-error-category transformer-error-operation transformer-error-message transformer-error-details transformer-error-cause'
check_public_module transformer/error_public \
  'transformer-error? transformer-error-category transformer-error-operation transformer-error-message transformer-error-details transformer-error-cause'
check_public_module transformer/config \
  'config-parse config-resolve config-validate config-canonical config-fingerprint config-ref'
check_public_module transformer/module \
  'module-parameters module-buffers module-state-dict module-load-state-dict! module-train! module-eval! module-zero-grad! parameter-tree-paths parameter-tree-handle parameter-tree-tie-groups parameter-handle-path parameter-handle-shape parameter-handle-dtype parameter-handle-device state-dict-paths state-dict-tensor state-dict-alias-groups state-dict-release!'
check_public_module transformer/data \
  'token-corpus-write! token-corpus-validate token-corpus-summary-shard-count token-corpus-summary-total-tokens token-corpus-summary-vocab-size token-corpus-summary-shard-token-limit token-corpus-summary-tokenizer-fingerprint token-corpus-summary-total-shard-bytes'
check_public_module transformer/persistence 'persistence-policy'
check_public_module transformer/tokenizer \
  'tokenizer-byte tokenizer-load tokenizer-save! tokenizer-encode tokenizer-decode tokenizer-vocab-size tokenizer-fingerprint tokenizer-special-token-id'

for evidence_name in global-defined.txt package-exports.txt undefined.txt \
    expected-undefined.txt public-strings.txt readelf-symbols.txt nm.txt \
    strings.txt private.d \
    link.map allowlist-provenance.tsv; do
  [[ -s "${t1_boundary_evidence}/${evidence_name}" ]] || \
    die "T1 aggregate evidence omits ${evidence_name}"
done
cmp "${PROJECT_ROOT}/native/t1_wave1_defined_symbols.txt" \
  "${t1_boundary_evidence}/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_public_exports.txt" \
  "${t1_boundary_evidence}/package-exports.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_public_strings.txt" \
  "${t1_boundary_evidence}/public-strings.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_undefined_symbols.txt" \
  "${t1_boundary_evidence}/undefined.txt"
cmp "${t1_boundary_evidence}/undefined.txt" \
  "${t1_boundary_evidence}/expected-undefined.txt"
[[ "$(wc -l <"${t1_boundary_evidence}/global-defined.txt")" == 47 ]] || \
  die "Wave 1 aggregate must expose E1 6 + P1 18 + D1 8 + X1 6 + C1 1 + T1 8"
[[ "$(wc -l <"${t1_boundary_evidence}/package-exports.txt")" == 41 ]] || \
  die "Wave 1 package export manifest must contain 41 non-E1 wrappers"
grep -Fx $'package_policy\tt1-wave1-aggregate' \
  "${t1_boundary_evidence}/allowlist-provenance.tsv" >/dev/null
[[ "$(ar t "${t1_boundary_archive}")" == wave1.o ]] || \
  die "Wave 1 archive must contain exactly its once-localized aggregate"

sed -e 's/^[^:]*://' -e 's/\\//g' "${t1_boundary_evidence}/private.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${t1_boundary_tmp}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_source_closure.txt" \
  "${t1_boundary_tmp}/source-closure.txt" || \
  die "Wave 1 trusted Eshkol source closure drifted"

required_local_symbols=(
  e1-internal-dispatch et-e1b-private-raise__eshkol_internal_abi
  transformer-error-make transformer-error-raise transformer-error-wrap-foreign
  et_e1b_private_raise_cabi_v1 et_e1b_consumer_raise_v1
  et_e1b_box_value_v1 et_e1b_ensure_private_initialized_v1
  et_checkpoint_io_atomic_write_v1 et_checkpoint_io_read_exact_v1
  et_d1_checked_write_new_v1 et_i64_tensor_create_v1
  et_i64_tensor_borrow_begin_v1 et_i64_tensor_borrow_view_v1
  et_i64_tensor_borrow_end_v1 et_i64_tensor_destroy_v1
  et_p1_private_context_create_v1 et_p1_private_provider_create_v1
  et_p1_private_provider_seal_release_v1
  et_p1_private_provider_snapshot_matches_release_v1
  et_p1_private_state_tensor_create_v1
  et_p1_private_state_tensor_validate_v1
  et_p1_private_state_release_begin_v1
  et_t1_i64_shell_create_v1 et_t1_i64_shell_length_v1
  et_t1_i64_shell_write_v1 et_t1_i64_shell_read_v1
  et_t1_i64_shell_seal_v1 et_t1_i64_shell_abort_v1
  et_t1_i64_shell_last_status_v1
)
for rename_manifest in e1b_private_renames.txt x1_config_private_renames.txt \
    p1_package_renames.txt d1_e1b_private_renames.txt \
    t1_wave1_private_renames.txt; do
  while read -r _ private_symbol; do
    required_local_symbols+=("${private_symbol}")
  done <"${PROJECT_ROOT}/native/${rename_manifest}"
done
for private_symbol in "${required_local_symbols[@]}"; do
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${private_symbol}$" \
    "${t1_boundary_evidence}/readelf-symbols.txt" >/dev/null || \
    die "Wave 1 private definition is not local: ${private_symbol}"
done

nm -s "${t1_boundary_archive}" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${t1_boundary_tmp}/archive-index.txt"
if grep -E 'et_e1b_(private|consumer|box|ensure)|et_(t1|i64|checkpoint|p1_private|d1_checked)|transformer-error-(make|raise|wrap-foreign)|e1-internal-dispatch' \
    "${t1_boundary_tmp}/archive-index.txt" >/dev/null; then
  die "Wave 1 archive index exposes a localized capability"
fi

run_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${t1_boundary_tmp}/cache/${cache_name}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${t1_boundary_tmp}/cache/${cache_name}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${t1_boundary_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${t1_boundary_timeout}s" "${t1_boundary_runner}" "$@"
}

private_source_bindings=(
  transformer-error-raise et-e1b-private-raise
  c1-persistence-policy-internal c1-wave1-persistence-policy
  t1-private-tokenizer-byte t1-wave1-tokenizer-byte t1-shell-create
  x1-private-config-parse p1-native-context-create
  state-dict-tensor-borrow-begin-internal
  state-dict-tensor-borrow-end-internal
  tensor-provider-preflight-internal
  tensor-provider-release-owned-internal!
  state-dict-adopt-owned-internal!
  et-d1-private-token-corpus-write
)
for private_binding in "${private_source_bindings[@]}"; do
  if run_compiler "private-binding-${private_binding}" \
      --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "$(project_build_dir)/t1" --lib eshkol_transformer_wave1 \
      -e "(begin (require transformer.tokenizer) ${private_binding})" \
      >"${t1_boundary_tmp}/private-binding-${private_binding}.stdout" \
      2>"${t1_boundary_tmp}/private-binding-${private_binding}.stderr"; then
    die "installed tokenizer module exposed private binding ${private_binding}"
  fi
  grep -F "${private_binding}" \
    "${t1_boundary_tmp}/private-binding-${private_binding}.stderr" >/dev/null || \
    die "private binding negative failed for the wrong reason: ${private_binding}"
done

aggregate_authority_negatives=(
  negative_public_provider_internal
  negative_public_module_internal
  negative_public_handle_internal
  negative_public_state_entry_internal
  negative_public_state_dict_internal
)
for negative_stem in "${aggregate_authority_negatives[@]}"; do
  negative_source="${PROJECT_ROOT}/tests/p1/${negative_stem}.esk"
  for phase in source object aot; do
    negative_output="${t1_boundary_tmp}/${negative_stem}-${phase}"
    negative_args=(--strict-types --no-stdlib -I "${PROJECT_ROOT}/lib"
                   -L "$(project_build_dir)/t1"
                   --lib eshkol_transformer_wave1)
    case "${phase}" in
      source) negative_args+=(-r "${negative_source}") ;;
      object)
        negative_args+=(--emit-object "${negative_source}"
                        -o "${negative_output}.o")
        ;;
      aot) negative_args+=("${negative_source}" -o "${negative_output}") ;;
    esac
    if run_compiler "authority-${negative_stem}-${phase}" \
        "${negative_args[@]}" \
        >"${negative_output}.stdout" 2>"${negative_output}.stderr"; then
      die "fresh-cache aggregate ${phase} admitted ${negative_stem}"
    fi
    [[ ! -e "${negative_output}" && ! -e "${negative_output}.o" ]] || \
      die "rejected aggregate authority probe published ${negative_stem} ${phase} output"
  done
done

run_compiler reverse-import-object --strict-types --no-stdlib --compile-only \
  -I "${PROJECT_ROOT}/lib" \
  --emit-depfile "${t1_boundary_tmp}/import-reverse.d" \
  "${PROJECT_ROOT}/tests/t1/import_reverse.esk" \
  -o "${t1_boundary_tmp}/import-reverse.o"
run_compiler reverse-import-object-repeat --strict-types --no-stdlib \
  --compile-only -I "${PROJECT_ROOT}/lib" \
  --emit-depfile "${t1_boundary_tmp}/import-reverse-repeat.d" \
  "${PROJECT_ROOT}/tests/t1/import_reverse.esk" \
  -o "${t1_boundary_tmp}/import-reverse-repeat.o"
cmp "${t1_boundary_tmp}/import-reverse.o" \
  "${t1_boundary_tmp}/import-reverse-repeat.o" || \
  die "reverse import object is not deterministic"
canonicalize_public_depfile() {
  local depfile=$1 output=$2
  sed -e 's/^[^:]*://' -e 's/\\//g' "${depfile}" | \
    tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
    sed "s#^${PROJECT_ROOT}/##" >"${output}"
}
canonicalize_public_depfile "${t1_boundary_tmp}/import-reverse.d" \
  "${t1_boundary_tmp}/public-source-closure.txt"
canonicalize_public_depfile "${t1_boundary_tmp}/import-reverse-repeat.d" \
  "${t1_boundary_tmp}/public-source-closure-repeat.txt"
cmp "${t1_boundary_tmp}/public-source-closure.txt" \
  "${t1_boundary_tmp}/public-source-closure-repeat.txt" || \
  die "reverse import depfile closure is not byte-deterministic"
cmp "${PROJECT_ROOT}/native/t1_wave1_public_source_closure.txt" \
  "${t1_boundary_tmp}/public-source-closure.txt" || \
  die "reverse import public source closure drifted"
run_compiler reverse-import-link --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -L "$(project_build_dir)/t1" \
  --lib eshkol_transformer_wave1 \
  "${PROJECT_ROOT}/tests/t1/import_reverse.esk" \
  -o "${t1_boundary_tmp}/import-reverse"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${t1_boundary_tmp}/import-reverse" \
  >"${t1_boundary_tmp}/import-reverse.stdout"
grep -Fx 't1-import-reverse:v1' \
  "${t1_boundary_tmp}/import-reverse.stdout" >/dev/null
for public_source in config data error_consumer error_public module persistence tokenizer; do
  grep -F "lib/transformer/${public_source}.esk" \
    "${t1_boundary_tmp}/import-reverse.d" >/dev/null || \
    die "reverse import depfile omits transformer.${public_source}"
done
if grep -E 'internal/(c1|p1|t1)|native/(e1b|t1|x1|d1)|error_(internal|core)\.esk|src/eshkol_transformer/token_shard\.esk' \
    "${t1_boundary_tmp}/import-reverse.d" >/dev/null; then
  die "installed aggregate module closure contains a trusted private source"
fi
for private_marker in et-e1b-private-raise t1-private- t1-wave1- \
    c1-persistence-policy-internal x1-private- p1-native- \
    state-dict-tensor-borrow-begin-internal \
    state-dict-tensor-borrow-end-internal \
    tensor-provider-preflight-internal \
    tensor-provider-release-owned-internal! \
    state-dict-adopt-owned-internal! \
    d1-token-corpus-write-impl d1-token-corpus-validate-impl; do
  if strings -a "${t1_boundary_tmp}/import-reverse.o" | \
      grep -F "${private_marker}" >/dev/null; then
    die "public aggregate caller contains private binding ${private_marker}"
  fi
done
nm -u --format=posix "${t1_boundary_tmp}/import-reverse.o" | \
  awk '{ print $1 }' | grep '^et_e1b_' | LC_ALL=C sort -u \
  >"${t1_boundary_tmp}/public-wrapper-refs.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_defined_symbols.txt" \
  "${t1_boundary_tmp}/public-wrapper-refs.txt" || \
  die "aggregate public source closure does not reference exactly 47 wrappers"
if ldd "${t1_boundary_tmp}/import-reverse" | grep -Eiq 'python|torch'; then
  die "Wave 1 aggregate caller links a Python or Torch runtime"
fi
if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/lib/transformer/tokenizer.esk" \
    "${PROJECT_ROOT}/lib/transformer/persistence.esk" \
    "${PROJECT_ROOT}/internal/t1/lib/transformer/tokenizer_internal.esk" \
    "${PROJECT_ROOT}/internal/t1/lib/t1_tokenizer_core.esk" \
    "${PROJECT_ROOT}/native/t1_wave1_root.esk" \
    "${PROJECT_ROOT}/native/t1_i64_shell.c"; then
  die "T1 delivered runtime contains a Python or PyTorch dependency"
fi

mkdir -p "${t1_boundary_tmp}/shadow/transformer"
printf '(error "hostile private seam loaded")\n' \
  >"${t1_boundary_tmp}/shadow/e1b_error_consumer_private.esk"
printf '(error "hostile tokenizer implementation loaded")\n' \
  >"${t1_boundary_tmp}/shadow/transformer/tokenizer_internal.esk"
ESHKOL_PATH="${t1_boundary_tmp}/shadow" \
ESHKOL_LIB_DIR="${t1_boundary_tmp}/shadow" \
E1B_COMPILER_TIMEOUT_SECONDS="${t1_boundary_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-t1.sh" \
    "${t1_boundary_tmp}/hostile-build"
cmp "${t1_boundary_artifact}" "${t1_boundary_tmp}/hostile-build/wave1.o"
for deterministic_evidence in global-defined.txt package-exports.txt \
    undefined.txt expected-undefined.txt public-strings.txt private.d link.map; do
  cmp "${t1_boundary_evidence}/${deterministic_evidence}" \
    "${t1_boundary_tmp}/hostile-build/wave1.o.evidence/${deterministic_evidence}"
done
if grep -F "${t1_boundary_tmp}/shadow" \
    "${t1_boundary_tmp}/hostile-build/wave1.o.evidence/private.d" >/dev/null; then
  die "Wave 1 build admitted a hostile module path"
fi

reject_builder_input() {
  local label=$1 expected=$2
  shift 2
  if "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" "$@" \
      >"${t1_boundary_tmp}/${label}.stdout" \
      2>"${t1_boundary_tmp}/${label}.stderr"; then
    die "Wave 1 builder admitted ${label}"
  fi
  grep -F "${expected}" "${t1_boundary_tmp}/${label}.stderr" >/dev/null || \
    die "Wave 1 ${label} rejection reported the wrong reason"
  [[ ! -e "${t1_boundary_tmp}/${label}.o" && \
     ! -e "${t1_boundary_tmp}/${label}.o.evidence" ]] || \
    die "rejected Wave 1 input published ${label} output"
}

cp "${PROJECT_ROOT}/native/t1_wave1_root.esk" \
  "${t1_boundary_tmp}/copied-root.esk"
reject_builder_input copied-root \
  'repository package components require their exact repository-owned private root' \
  "${t1_boundary_tmp}/copied-root.esk" \
  "${PROJECT_ROOT}/native/t1_wave1_package_bridge.c" \
  "${PROJECT_ROOT}/native/t1_wave1_private_renames.txt" \
  "${PROJECT_ROOT}/native/t1_wave1_public_exports.txt" \
  "${t1_boundary_tmp}/copied-root.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
reject_builder_input prelocalized-root \
  'repository package components require their exact repository-owned private root' \
  "${t1_boundary_artifact}" \
  "${PROJECT_ROOT}/native/t1_wave1_package_bridge.c" \
  "${PROJECT_ROOT}/native/t1_wave1_private_renames.txt" \
  "${PROJECT_ROOT}/native/t1_wave1_public_exports.txt" \
  "${t1_boundary_tmp}/prelocalized-root.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
reject_builder_input standalone-t1 \
  'repository package components require their exact repository-owned private root' \
  "${PROJECT_ROOT}/internal/t1/lib/transformer/tokenizer_internal.esk" \
  "${PROJECT_ROOT}/native/t1_wave1_package_bridge.c" \
  "${PROJECT_ROOT}/native/t1_wave1_private_renames.txt" \
  "${PROJECT_ROOT}/native/t1_wave1_public_exports.txt" \
  "${t1_boundary_tmp}/standalone-t1.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"

guessed_symbols=(
  transformer-error-raise e1-internal-dispatch et_e1b_private_raise_cabi_v1
  et_e1b_private_t1_tokenizer_encode_cabi_v1
  et_e1b_private_c1_persistence_policy_cabi_v1
  et_t1_i64_shell_create_v1 et_i64_tensor_create_v1
  et_checkpoint_io_atomic_write_v1 et_p1_private_context_create_v1
  et_d1_checked_write_new_v1 et_e1b_private_x1_config_parse_cabi_v1
)
for guessed_symbol in "${guessed_symbols[@]}"; do
  guessed_object="${t1_boundary_tmp}/negative-${guessed_symbol}.o"
  guessed_combined="${t1_boundary_tmp}/negative-${guessed_symbol}-combined.o"
  "${t1_boundary_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -DT1_PRIVATE_SYMBOL="${guessed_symbol}" -c \
    "${PROJECT_ROOT}/tests/t1/negative_native_symbol_link.c" \
    -o "${guessed_object}"
  "${t1_boundary_cc}" -r "${guessed_object}" -Wl,--whole-archive \
    "${t1_boundary_archive}" -Wl,--no-whole-archive \
    -o "${guessed_combined}"
  nm -u "${guessed_combined}" | awk '{ print $NF }' | \
    grep -Fx "${guessed_symbol}" >/dev/null || \
    die "guessed private symbol resolved from Wave 1: ${guessed_symbol}"
done

E1B_COMPILER_TIMEOUT_SECONDS="${t1_boundary_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-x1.sh" \
    "${t1_boundary_tmp}/standalone-x1"
if "${t1_boundary_cc}" -r -Wl,--whole-archive \
    "${t1_boundary_archive}" \
    "${t1_boundary_tmp}/standalone-x1/libeshkol_transformer_x1.a" \
    -Wl,--no-whole-archive -o "${t1_boundary_tmp}/cross-artifact.o" \
    >"${t1_boundary_tmp}/cross-artifact.stdout" \
    2>"${t1_boundary_tmp}/cross-artifact.stderr"; then
  die "linker combined independent aggregate and standalone E1 registries"
fi
grep -E 'multiple definition.*et_e1b_error_(predicate|category)_v1' \
  "${t1_boundary_tmp}/cross-artifact.stderr" >/dev/null || \
  die "cross-artifact rejection did not identify duplicate E1 ownership"

printf 'T1 BOUNDARY PASS: exact 47-global source/localization/module manifests, reverse imports, hostile-path and copied/prelocalized/standalone rejection, crafted links, cross-artifact collision, and Python isolation\n'
