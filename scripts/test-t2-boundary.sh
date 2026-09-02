#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar awk cmp cp diff grep ldd nm readelf sed strings timeout tr xargs; do
  require_command "${command}"
done

t2_boundary_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-t2-boundary.XXXXXX")"
trap 'rm -rf -- "${t2_boundary_tmp}"' EXIT
t2_boundary_runner="$(eshkol_build_dir)/eshkol-run"
t2_boundary_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
t2_boundary_cc="$(tsv_value "${t2_boundary_provenance}" cc_path)"
t2_boundary_cxx="$(tsv_value "${t2_boundary_provenance}" cxx_path)"
t2_boundary_timeout="${T2_COMPILER_TIMEOUT_SECONDS:-300}"
[[ "${t2_boundary_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "T2_COMPILER_TIMEOUT_SECONDS must be a positive integer"

t2_root="${PROJECT_ROOT}/native/t2_wave2_root.esk"
t2_bridge="${PROJECT_ROOT}/native/t2_wave2_package_bridge.c"
t2_renames="${PROJECT_ROOT}/native/t2_wave2_private_renames.txt"
t2_exports="${PROJECT_ROOT}/native/t2_wave2_public_exports.txt"
t2_d1_root="${PROJECT_ROOT}/native/t2_wave2_d1_test_root.esk"
t2_d1_bridge="${PROJECT_ROOT}/native/t2_wave2_d1_test_package_bridge.c"
t2_d1_renames="${PROJECT_ROOT}/native/t2_wave2_d1_test_private_renames.txt"
t2_d1_exports="${PROJECT_ROOT}/native/t2_wave2_d1_test_public_exports.txt"
t2_includes=(
  "${PROJECT_ROOT}/internal/p1/lib"
  "${PROJECT_ROOT}/internal/c1/lib"
  "${PROJECT_ROOT}/internal/t2/lib"
  "${PROJECT_ROOT}/internal/t1/lib"
  "${PROJECT_ROOT}/src"
)

mkdir -p "${t2_boundary_tmp}/shadow/transformer"
for hostile_source in t2_bpe_core.esk t2_d1_bridge.esk; do
  printf '(error "hostile T2 source loaded")\n' \
    >"${t2_boundary_tmp}/shadow/${hostile_source}"
done
printf '(error "hostile T2 tokenizer implementation loaded")\n' \
  >"${t2_boundary_tmp}/shadow/transformer/tokenizer_internal.esk"
printf '#error "hostile T2 native include loaded"\n' \
  >"${t2_boundary_tmp}/shadow/e1b_error_consumer_bridge.h"

build_d1_test() {
  local output=$1
  E1B_COMPILER_TIMEOUT_SECONDS="${t2_boundary_timeout}" \
    "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${t2_d1_root}" "${t2_d1_bridge}" "${t2_d1_renames}" \
    "${t2_d1_exports}" "${output}" "${t2_includes[@]}"
  ar rcsD "$(dirname -- "${output}")/libeshkol_transformer_wave2_d1_test.a" \
    "${output}"
}

# Build every Wave 2 package twice at the identical invocation/output path
# while hostile ambient lookup variables are present. Retain each completed
# build before the next fresh compiler cache replaces the publication path.
# This avoids asserting that ELF bytes are invariant under output-root spelling.
working_dir="${t2_boundary_tmp}/working"
mkdir -p "${working_dir}/production" "${working_dir}/d1"
for repetition in 1 2; do
  repetition_dir="${t2_boundary_tmp}/repeat-${repetition}"
  mkdir -p "${repetition_dir}/production" "${repetition_dir}/d1"
  ESHKOL_PATH="${t2_boundary_tmp}/shadow" \
  ESHKOL_LIB_DIR="${t2_boundary_tmp}/shadow" \
    E1B_COMPILER_TIMEOUT_SECONDS="${t2_boundary_timeout}" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-t2.sh" \
      "${working_dir}/production"
  ESHKOL_PATH="${t2_boundary_tmp}/shadow" \
  ESHKOL_LIB_DIR="${t2_boundary_tmp}/shadow" \
    build_d1_test "${working_dir}/d1/wave2-d1-test.o"
  cp "${working_dir}/production/wave2.o" \
    "${repetition_dir}/production/wave2.o"
  cp "${working_dir}/production/libeshkol_transformer_wave2.a" \
    "${repetition_dir}/production/libeshkol_transformer_wave2.a"
  cp -a "${working_dir}/production/wave2.o.evidence" \
    "${repetition_dir}/production/wave2.o.evidence"
  cp "${working_dir}/d1/wave2-d1-test.o" \
    "${repetition_dir}/d1/wave2-d1-test.o"
  cp "${working_dir}/d1/libeshkol_transformer_wave2_d1_test.a" \
    "${repetition_dir}/d1/libeshkol_transformer_wave2_d1_test.a"
  cp -a "${working_dir}/d1/wave2-d1-test.o.evidence" \
    "${repetition_dir}/d1/wave2-d1-test.o.evidence"
done

production_one="${t2_boundary_tmp}/repeat-1/production"
production_two="${t2_boundary_tmp}/repeat-2/production"
d1_one="${t2_boundary_tmp}/repeat-1/d1"
d1_two="${t2_boundary_tmp}/repeat-2/d1"
cmp "${production_one}/wave2.o" "${production_two}/wave2.o"
cmp "${production_one}/libeshkol_transformer_wave2.a" \
  "${production_two}/libeshkol_transformer_wave2.a"
diff -ru "${production_one}/wave2.o.evidence" \
  "${production_two}/wave2.o.evidence"
cmp "${d1_one}/wave2-d1-test.o" "${d1_two}/wave2-d1-test.o"
cmp "${d1_one}/libeshkol_transformer_wave2_d1_test.a" \
  "${d1_two}/libeshkol_transformer_wave2_d1_test.a"
diff -ru "${d1_one}/wave2-d1-test.o.evidence" \
  "${d1_two}/wave2-d1-test.o.evidence"

production_object="${production_one}/wave2.o"
production_archive="${production_one}/libeshkol_transformer_wave2.a"
production_evidence="${production_object}.evidence"
d1_object="${d1_one}/wave2-d1-test.o"
d1_archive="${d1_one}/libeshkol_transformer_wave2_d1_test.a"
d1_evidence="${d1_object}.evidence"

[[ "$(wc -l <"${production_evidence}/global-defined.txt")" == 46 ]] || \
  die "Wave 2 production aggregate must expose exactly 46 globals"
[[ "$(wc -l <"${production_evidence}/package-exports.txt")" == 40 ]] || \
  die "Wave 2 production package must expose exactly 40 wrappers"
[[ "$(wc -l <"${d1_evidence}/global-defined.txt")" == 47 ]] || \
  die "Wave 2 D1 test aggregate must expose exactly 47 globals"
[[ "$(wc -l <"${d1_evidence}/package-exports.txt")" == 41 ]] || \
  die "Wave 2 D1 test package must expose exactly 41 wrappers"
cmp "${PROJECT_ROOT}/native/t2_wave2_defined_symbols.txt" \
  "${production_evidence}/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_public_exports.txt" \
  "${production_evidence}/package-exports.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_d1_test_defined_symbols.txt" \
  "${d1_evidence}/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_d1_test_public_exports.txt" \
  "${d1_evidence}/package-exports.txt"
[[ "$(ar t "${production_archive}")" == wave2.o ]] || \
  die "Wave 2 archive must contain exactly its once-localized aggregate"
[[ "$(ar t "${d1_archive}")" == wave2-d1-test.o ]] || \
  die "Wave 2 D1 test archive must contain exactly its localized aggregate"

check_source_closure() {
  local depfile=$1 expected=$2 actual=$3
  sed -e 's/^[^:]*://' -e 's/\\//g' "${depfile}" | \
    tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
    sed "s#^${PROJECT_ROOT}/##" >"${actual}"
  cmp "${expected}" "${actual}"
  if grep -F "${t2_boundary_tmp}/shadow" "${depfile}" >/dev/null; then
    die "Wave 2 source closure admitted a hostile include root"
  fi
}
check_source_closure "${production_evidence}/private.d" \
  "${PROJECT_ROOT}/native/t2_wave2_source_closure.txt" \
  "${t2_boundary_tmp}/production-source-closure.txt"
check_source_closure "${d1_evidence}/private.d" \
  "${PROJECT_ROOT}/native/t2_wave2_d1_test_source_closure.txt" \
  "${t2_boundary_tmp}/d1-source-closure.txt"

check_localized_archive() {
  local object=$1 archive=$2 evidence=$3
  shift 3
  nm -s "${archive}" | \
    awk '/^Archive index:$/ { active = 1; next }
         active && /^$/ { active = 0; next }
         active { print }' >"${archive}.index"
  for rename_manifest in "$@"; do
    while read -r _ private_symbol; do
      grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${private_symbol}$" \
        "${evidence}/readelf-symbols.txt" >/dev/null || \
        die "Wave 2 private definition is not local: ${private_symbol}"
      if grep -F "${private_symbol}" "${archive}.index" >/dev/null; then
        die "Wave 2 archive index leaks localized symbol ${private_symbol}"
      fi
    done <"${rename_manifest}"
  done
  for native_symbol in et_t1_i64_shell_create_v1 et_i64_tensor_create_v1 \
      et_checkpoint_io_atomic_write_v1 et_d1_checked_write_new_v1 \
      et_p1_private_context_create_v1; do
    grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${native_symbol}$" \
      "${evidence}/readelf-symbols.txt" >/dev/null || \
      die "Wave 2 native definition is not local: ${native_symbol}"
    if grep -F "${native_symbol}" "${archive}.index" >/dev/null; then
      die "Wave 2 archive index leaks native symbol ${native_symbol}"
    fi
  done
  if nm -g --defined-only --format=posix "${object}" | \
      awk '{ print $1 }' | grep -Ev '^et_e1b_(error|public)_' >/dev/null; then
    die "Wave 2 aggregate exposes a nonpublic global definition"
  fi
}
common_rename_manifests=(
  "${PROJECT_ROOT}/native/e1b_private_renames.txt"
  "${PROJECT_ROOT}/native/x1_config_private_renames.txt"
  "${PROJECT_ROOT}/native/p1_package_renames.txt"
  "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt"
)
check_localized_archive "${production_object}" "${production_archive}" \
  "${production_evidence}" "${common_rename_manifests[@]}" "${t2_renames}"
check_localized_archive "${d1_object}" "${d1_archive}" "${d1_evidence}" \
  "${common_rename_manifests[@]}" "${t2_d1_renames}"

reject_builder_input() {
  local label=$1 expected=$2
  shift 2
  if "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" "$@" \
      >"${t2_boundary_tmp}/${label}.stdout" \
      2>"${t2_boundary_tmp}/${label}.stderr"; then
    die "Wave 2 builder admitted ${label}"
  fi
  grep -F "${expected}" "${t2_boundary_tmp}/${label}.stderr" >/dev/null || \
    die "Wave 2 ${label} rejection reported the wrong reason"
  [[ ! -e "${t2_boundary_tmp}/${label}.o" && \
     ! -e "${t2_boundary_tmp}/${label}.o.evidence" ]] || \
    die "rejected Wave 2 input published ${label} output"
}

cp "${t2_root}" "${t2_boundary_tmp}/copied-root.esk"
cp "${t2_bridge}" "${t2_boundary_tmp}/copied-bridge.c"
cp "${t2_d1_root}" "${t2_boundary_tmp}/copied-d1-root.esk"
repository_root_error='repository package components require their exact repository-owned private root'
reject_builder_input copied-root "${repository_root_error}" \
  "${t2_boundary_tmp}/copied-root.esk" "${t2_bridge}" "${t2_renames}" \
  "${t2_exports}" "${t2_boundary_tmp}/copied-root.o" "${t2_includes[@]}"
reject_builder_input prelocalized-root "${repository_root_error}" \
  "${production_object}" "${t2_bridge}" "${t2_renames}" "${t2_exports}" \
  "${t2_boundary_tmp}/prelocalized-root.o" "${t2_includes[@]}"
for standalone_root in \
    "${PROJECT_ROOT}/internal/t2/lib/transformer/tokenizer_internal.esk" \
    "${PROJECT_ROOT}/internal/t2/lib/t2_bpe_core.esk" \
    "${PROJECT_ROOT}/internal/t2/lib/t2_d1_bridge.esk"; do
  label="standalone-$(basename -- "${standalone_root}" .esk)"
  reject_builder_input "${label}" "${repository_root_error}" \
    "${standalone_root}" "${t2_bridge}" "${t2_renames}" "${t2_exports}" \
    "${t2_boundary_tmp}/${label}.o" "${t2_includes[@]}"
done
reject_builder_input copied-d1-root "${repository_root_error}" \
  "${t2_boundary_tmp}/copied-d1-root.esk" "${t2_d1_bridge}" \
  "${t2_d1_renames}" "${t2_d1_exports}" \
  "${t2_boundary_tmp}/copied-d1-root.o" "${t2_includes[@]}"
reject_builder_input copied-bridge \
  'T2 aggregate policy requires the exact repository bridge' \
  "${t2_root}" "${t2_boundary_tmp}/copied-bridge.c" "${t2_renames}" \
  "${t2_exports}" "${t2_boundary_tmp}/copied-bridge.o" "${t2_includes[@]}"
reject_builder_input mismatched-bridge \
  'T2 aggregate policy requires the exact repository bridge' \
  "${t2_root}" "${t2_d1_bridge}" "${t2_renames}" "${t2_exports}" \
  "${t2_boundary_tmp}/mismatched-bridge.o" "${t2_includes[@]}"
reject_builder_input mismatched-renames \
  'T2 aggregate policy requires the exact repository rename map' \
  "${t2_root}" "${t2_bridge}" "${t2_d1_renames}" "${t2_exports}" \
  "${t2_boundary_tmp}/mismatched-renames.o" "${t2_includes[@]}"
reject_builder_input mismatched-exports \
  'T2 aggregate policy requires the exact repository export list' \
  "${t2_root}" "${t2_bridge}" "${t2_renames}" "${t2_d1_exports}" \
  "${t2_boundary_tmp}/mismatched-exports.o" "${t2_includes[@]}"
reject_builder_input mismatched-d1-bridge \
  'T2 D1 test aggregate policy requires the exact repository bridge' \
  "${t2_d1_root}" "${t2_bridge}" "${t2_d1_renames}" "${t2_d1_exports}" \
  "${t2_boundary_tmp}/mismatched-d1-bridge.o" "${t2_includes[@]}"
reject_builder_input wrong-include-order \
  'T2 aggregate policy requires exact ordered trusted include roots' \
  "${t2_root}" "${t2_bridge}" "${t2_renames}" "${t2_exports}" \
  "${t2_boundary_tmp}/wrong-include-order.o" \
  "${t2_includes[1]}" "${t2_includes[0]}" "${t2_includes[@]:2}"
reject_builder_input hostile-include \
  'T2 aggregate policy requires exact ordered trusted include roots' \
  "${t2_root}" "${t2_bridge}" "${t2_renames}" "${t2_exports}" \
  "${t2_boundary_tmp}/hostile-include.o" "${t2_boundary_tmp}/shadow" \
  "${t2_includes[@]}"

run_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${t2_boundary_tmp}/cache/${cache_name}"
  env -u ESHKOL_PATH \
    XDG_CACHE_HOME="${t2_boundary_tmp}/cache/${cache_name}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${t2_boundary_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${t2_boundary_timeout}s" "${t2_boundary_runner}" "$@"
}

private_source_bindings=(
  t2-private-tokenizer-parse t2-private-tokenizer-serialize
  t2-bpe-train-core t2-stream-encoder-open t2-stream-decoder-push!
  t2-token-corpus-read-i64-le t2-test-token-corpus-read
  et-e1b-private-t2-test-token-corpus-read-cabi-v1
)
for private_binding in "${private_source_bindings[@]}"; do
  if run_compiler private-binding-negatives \
      --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "${production_one}" --lib eshkol_transformer_wave2 \
      -e "(begin (require transformer.tokenizer) ${private_binding})" \
      >"${t2_boundary_tmp}/binding-${private_binding}.stdout" \
      2>"${t2_boundary_tmp}/binding-${private_binding}.stderr"; then
    die "installed Wave 2 module exposed private binding ${private_binding}"
  fi
  grep -F "${private_binding}" \
    "${t2_boundary_tmp}/binding-${private_binding}.stderr" >/dev/null || \
    die "private binding negative failed for the wrong reason: ${private_binding}"
done

guessed_symbols=(
  et_e1b_private_t1_tokenizer_encode_cabi_v1
  et_e1b_private_t2_test_token_corpus_read_cabi_v1
  et_t1_i64_shell_create_v1 et_i64_tensor_create_v1
  et_checkpoint_io_atomic_write_v1 et_d1_checked_write_new_v1
  et_p1_private_context_create_v1
)
for guessed_symbol in "${guessed_symbols[@]}"; do
  guessed_object="${t2_boundary_tmp}/native-${guessed_symbol}.o"
  guessed_combined="${t2_boundary_tmp}/native-${guessed_symbol}-combined.o"
  "${t2_boundary_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -DT1_PRIVATE_SYMBOL="${guessed_symbol}" -c \
    "${PROJECT_ROOT}/tests/t1/negative_native_symbol_link.c" \
    -o "${guessed_object}"
  "${t2_boundary_cc}" -r "${guessed_object}" -Wl,--whole-archive \
    "${production_archive}" -Wl,--no-whole-archive -o "${guessed_combined}"
  nm -u "${guessed_combined}" | awk '{ print $NF }' | \
    grep -Fx "${guessed_symbol}" >/dev/null || \
    die "crafted native reference resolved from Wave 2: ${guessed_symbol}"
done

# A public caller is compiled twice through fresh caches at the same output
# path; retained objects and linked binaries must be byte-identical.
caller_source="${PROJECT_ROOT}/tests/t1/import_reverse.esk"
for repetition in 1 2; do
  run_compiler "caller-object-${repetition}" --strict-types --no-stdlib \
    --compile-only -I "${PROJECT_ROOT}/lib" \
    --emit-depfile "${t2_boundary_tmp}/caller.d" "${caller_source}" \
    -o "${t2_boundary_tmp}/caller.o"
  run_compiler "caller-link-${repetition}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" -L "${production_one}" \
    --lib eshkol_transformer_wave2 "${caller_source}" \
    -o "${t2_boundary_tmp}/caller"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${t2_boundary_tmp}/caller" \
    >"${t2_boundary_tmp}/caller-${repetition}.stdout"
  cp "${t2_boundary_tmp}/caller.o" \
    "${t2_boundary_tmp}/caller-${repetition}.o"
  cp "${t2_boundary_tmp}/caller" \
    "${t2_boundary_tmp}/caller-${repetition}"
  cp "${t2_boundary_tmp}/caller.d" \
    "${t2_boundary_tmp}/caller-${repetition}.d"
done
cmp "${t2_boundary_tmp}/caller-1.o" "${t2_boundary_tmp}/caller-2.o"
cmp "${t2_boundary_tmp}/caller-1" "${t2_boundary_tmp}/caller-2"
cmp "${t2_boundary_tmp}/caller-1.d" "${t2_boundary_tmp}/caller-2.d"
cmp "${t2_boundary_tmp}/caller-1.stdout" \
  "${t2_boundary_tmp}/caller-2.stdout"
grep -Fx 't1-import-reverse:v1' \
  "${t2_boundary_tmp}/caller-1.stdout" >/dev/null
for public_source in config data error_consumer error_public module \
    persistence tokenizer; do
  grep -F "lib/transformer/${public_source}.esk" \
    "${t2_boundary_tmp}/caller-1.d" >/dev/null || \
    die "Wave 2 public caller depfile omits transformer.${public_source}"
done
if grep -E 'internal/(c1|p1|t1|t2)|native/(e1b|t1|t2|x1|d1)|error_(internal|core)\.esk|src/eshkol_transformer/token_shard\.esk' \
    "${t2_boundary_tmp}/caller-1.d" >/dev/null; then
  die "Wave 2 public caller depfile contains a trusted private source"
fi
if strings -a "${t2_boundary_tmp}/caller-1.o" | \
    grep -E 't2-private-|t2-wave2-|t2-bpe-|t2-stream-|t2-token-corpus-read|et-e1b-private|et_e1b_private' >/dev/null; then
  die "Wave 2 public caller contains a private source binding"
fi
nm -u --format=posix "${t2_boundary_tmp}/caller-1.o" | \
  awk '{ print $1 }' | grep '^et_e1b_' | LC_ALL=C sort -u \
  >"${t2_boundary_tmp}/caller-wrapper-refs.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_defined_symbols.txt" \
  "${t2_boundary_tmp}/caller-wrapper-refs.txt"
if ldd "${t2_boundary_tmp}/caller-1" | grep -Eiq 'python|torch'; then
  die "Wave 2 public caller links a Python or Torch runtime"
fi

# Independent Wave 1 and Wave 2 packages must not be combined: each owns the
# same E1 registry and public wrappers.
E1B_COMPILER_TIMEOUT_SECONDS="${t2_boundary_timeout}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-t1.sh" \
    "${t2_boundary_tmp}/wave1"
if "${t2_boundary_cc}" -r -Wl,--whole-archive \
    "${t2_boundary_tmp}/wave1/libeshkol_transformer_wave1.a" \
    "${production_archive}" -Wl,--no-whole-archive \
    -o "${t2_boundary_tmp}/wave1-wave2.o" \
    >"${t2_boundary_tmp}/wave1-wave2.stdout" \
    2>"${t2_boundary_tmp}/wave1-wave2.stderr"; then
  die "linker combined independent Wave 1 and Wave 2 E1 registries"
fi
grep -E 'multiple definition.*et_e1b_error_(predicate|category)_v1' \
  "${t2_boundary_tmp}/wave1-wave2.stderr" >/dev/null || \
  die "Wave 1 plus Wave 2 rejection did not identify duplicate E1 ownership"

printf 'T2 BOUNDARY PASS: exact 46/40 production and 47/41 D1-test surfaces, deterministic localized objects/archives/evidence/AOT, hostile-path and tuple rejection, localization, crafted-link isolation, public closure, and Wave1+Wave2 E1 collision\n'
