#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
require_command python3
require_command ar
require_command nm
require_command readelf
require_command strings
verify_toolchain

D1_TIMEOUT=$(command -v timeout)
D1_PYTHON=$(command -v python3)
D1_RUNNER="$(eshkol_build_dir)/eshkol-run"
D1_CC="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
D1_CXX="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
D1_TIMEOUT_SECONDS=${D1_TIMEOUT_SECONDS:-60}
D1_ARTIFACT_DIR="$(project_build_dir)/d1"
D1_LIBRARY="${D1_ARTIFACT_DIR}/libeshkol_transformer_d1.a"
D1_MODULE="${D1_ARTIFACT_DIR}/stdlib.o"
[[ "${D1_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
  die "D1_TIMEOUT_SECONDS must be a positive integer"
[[ -r "${D1_LIBRARY}" ]] || die "canonical D1 native archive is missing"
[[ -r "${D1_ARTIFACT_DIR}/data_io.o" ]] || \
  die "canonical D1 native object is missing"
[[ -r "${D1_MODULE}" ]] || die "canonical D1 precompiled module is missing"
[[ -r "${D1_ARTIFACT_DIR}/stdlib.d" ]] || \
  die "canonical D1 module depfile is missing"
[[ "$(ar t "${D1_LIBRARY}")" == "data_io.o" ]] || \
  die "canonical D1 native archive has unexpected members"

D1_PUBLIC_NAMES=(
  token-corpus-write!
  token-corpus-validate
  token-corpus-summary-shard-count
  token-corpus-summary-total-tokens
  token-corpus-summary-vocab-size
  token-corpus-summary-shard-token-limit
  token-corpus-summary-tokenizer-fingerprint
  token-corpus-summary-total-shard-bytes
)
mapfile -t D1_DECLARED_NAMES < <(
  sed -n '/^(provide /,/)/p' "${PROJECT_ROOT}/lib/transformer/data.esk" |
    tr '()' '  ' | tr -s '[:space:]' '\n' |
    grep -v -e '^provide$' -e '^$'
)
[[ "${D1_DECLARED_NAMES[*]}" == "${D1_PUBLIC_NAMES[*]}" ]] || \
  die "transformer.data must provide exactly the eight accepted D1 names"

grep -F 'lib/stdlib.esk' "${D1_ARTIFACT_DIR}/stdlib.d" >/dev/null
grep -F 'lib/transformer/data.esk' "${D1_ARTIFACT_DIR}/stdlib.d" >/dev/null
grep -F 'src/eshkol_transformer/token_shard.esk' \
  "${D1_ARTIFACT_DIR}/stdlib.d" >/dev/null
grep -F 'lib/transformer/error_public.esk' \
  "${D1_ARTIFACT_DIR}/stdlib.d" >/dev/null
grep -F 'lib/transformer/error_core.esk' \
  "${D1_ARTIFACT_DIR}/stdlib.d" >/dev/null
if grep -F 'lib/transformer/error_internal.esk' \
    "${D1_ARTIFACT_DIR}/stdlib.d" >/dev/null; then
  die "D1 precompiled module unexpectedly includes transformer.error_internal"
fi

for public_name in "${D1_PUBLIC_NAMES[@]}"; do
  [[ "$(nm -g --defined-only --format=posix "${D1_MODULE}" |
      awk -v wanted="${public_name}" '$1 == wanted { count++ } END { print count + 0 }')" == 1 ]] || \
    die "D1 precompiled module does not define public procedure ${public_name} exactly once"
  [[ "$(nm -g --defined-only --format=posix "${D1_MODULE}" |
      awk -v wanted="${public_name}_sexpr" '$1 == wanted { count++ } END { print count + 0 }')" == 1 ]] || \
    die "D1 precompiled module does not define first-class metadata for ${public_name}"
done
if nm -g --defined-only --format=posix "${D1_MODULE}" |
    awk '{ print $1 }' |
    grep -E '^(\.Lprivate_slot_|d1-|eshkol_g_d1_2D|et_d1_)' >/dev/null; then
  die "D1 precompiled module exports a private D1 helper or FFI alias"
fi
if readelf --wide --symbols "${D1_MODULE}" |
    awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }' |
    grep -E '^(\.Lprivate_slot_|d1-|eshkol_g_d1_2D|et_d1_)' >/dev/null; then
  die "readelf found a global private D1 helper or FFI alias"
fi
if strings -a "${D1_MODULE}" |
    grep -E '(d1-token-(write-temporary-new!|publish-temporary!|atomic-write-new!|corpus-(write|validate)-impl|checked-write-new))(_sexpr|__eshkol_internal_abi)|eshkol_g_d1_2D' \
      >/dev/null; then
  die "D1 precompiled module retains a former helper/FFI symbol name"
fi

D1_NM_DEFINED="$({
  nm -g --defined-only --format=posix "${D1_LIBRARY}" |
    awk '$2 ~ /^[A-Za-z]$/ { print $1 }'
})"
[[ "${D1_NM_DEFINED}" == et_d1_checked_write_new_v1 ]] || \
  die "nm found an unexpected globally defined symbol in the public D1 archive"
D1_READELF_DEFINED="$({
  readelf --wide --symbols "${D1_ARTIFACT_DIR}/data_io.o" |
    awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }'
})"
[[ "${D1_READELF_DEFINED}" == et_d1_checked_write_new_v1 ]] || \
  die "readelf found an unexpected globally defined symbol in the public D1 object"
if strings -a "${D1_LIBRARY}" |
    grep -E '(d1-token|d1_token_(write|publish|atomic|corpus|validate|list|helper)|et_d1_(token_)?(write|publish|atomic|corpus|validate|list|helper))' \
      >/dev/null; then
  die "canonical D1 native archive contains a private Eshkol/helper-family name"
fi

D1_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-d1.XXXXXX")
trap 'rm -rf -- "${D1_TMP}"' EXIT

d1_compile() {
  local source=$1 output=$2
  local cache_home=${D1_CACHE_HOME:-${D1_TMP}/cache}
  local library_dir=${D1_LIBRARY_DIR:-${D1_ARTIFACT_DIR}}
  local library_name=${D1_LIBRARY_NAME:-eshkol_transformer_d1}
  local module_object=${D1_MODULE_OBJECT:-${library_dir}/stdlib.o}
  shift 2
  mkdir -p "${cache_home}"
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
    env XDG_CACHE_HOME="${cache_home}" ESHKOL_CXX_COMPILER="${D1_CXX}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    "${D1_RUNNER}" \
    --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/src" \
    -L "${library_dir}" --lib "${library_name}" \
    "${module_object}" "$@" "${source}" -o "${output}"
}

d1_assert_private_binding() {
  local hidden_name=$1 hidden_source=$2 mode run output log depfile
  local source="${D1_TMP}/negative-private-${hidden_source}.esk"
  printf '(require transformer.data)\n\n(define leaked-private-binding %s)\n' \
    "${hidden_name}" >"${source}"
  for mode in object aot; do
    for run in 1; do
      output="${D1_TMP}/negative-private-${hidden_source}-${mode}-${run}"
      log="${output}.log"
      depfile="${output}.d"
      if [[ "${mode}" == object ]]; then
        output="${output}.o"
        if D1_CACHE_HOME="${D1_TMP}/negative-private-${hidden_source}-${mode}-cache-${run}" \
            d1_compile "${source}" "${output}" --emit-depfile "${depfile}" \
              --compile-only >"${log}" 2>&1; then
          die "D1 private ${hidden_name} unexpectedly compiled to an object"
        fi
      else
        output="${output}.bin"
        if D1_CACHE_HOME="${D1_TMP}/negative-private-${hidden_source}-${mode}-cache-${run}" \
            d1_compile "${source}" "${output}" >"${log}" 2>&1; then
          die "D1 private ${hidden_name} unexpectedly AOT-linked"
        fi
      fi
      grep -F "Unbound variable: ${hidden_name}" "${log}" >/dev/null
      test ! -e "${output}"
    done
  done
}

d1_assert_unavailable_call() {
  local hidden_name=$1 hidden_source=$2 mode output log source
  source="${D1_TMP}/negative-direct-${hidden_source}.esk"
  printf '(require transformer.data)\n\n(%s)\n' "${hidden_name}" >"${source}"
  for mode in object aot; do
    output="${D1_TMP}/negative-direct-${hidden_source}-${mode}"
    log="${output}.log"
    if [[ "${mode}" == object ]]; then
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.o" --compile-only >"${log}" 2>&1; then
        die "D1 private ${hidden_name} unexpectedly compiled as a direct call"
      fi
      test ! -e "${output}.o"
    else
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.bin" >"${log}" 2>&1; then
        die "D1 private ${hidden_name} unexpectedly linked as a direct call"
      fi
      test ! -e "${output}.bin"
    fi
    grep -F "Unknown function: ${hidden_name}" "${log}" >/dev/null
  done
}

d1_assert_public_wrong_arity() {
  local public_name=$1 source_tag=$2 mode output log source
  source="${D1_TMP}/negative-arity-${source_tag}.esk"
  printf '(require transformer.data)\n\n(%s)\n' "${public_name}" >"${source}"
  for mode in object aot; do
    output="${D1_TMP}/negative-arity-${source_tag}-${mode}"
    log="${output}.log"
    if [[ "${mode}" == object ]]; then
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.o" --compile-only >"${log}" 2>&1; then
        die "D1 public ${public_name} accepted a wrong-arity object call"
      fi
      test ! -e "${output}.o"
    else
      if D1_CACHE_HOME="${output}-cache" \
          d1_compile "${source}" "${output}.bin" >"${log}" 2>&1; then
        die "D1 public ${public_name} accepted a wrong-arity AOT call"
      fi
      test ! -e "${output}.bin"
    fi
    grep -F "Arity mismatch: ${public_name}" "${log}" >/dev/null
  done
}

D1_FAULT_ARTIFACT_DIR="${D1_TMP}/fault-artifacts"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-d1.sh" \
  "${D1_FAULT_ARTIFACT_DIR}" test-faults \
  >"${D1_TMP}/fault-build.stdout" 2>"${D1_TMP}/fault-build.stderr"
D1_FAULT_LIBRARY="${D1_FAULT_ARTIFACT_DIR}/libeshkol_transformer_d1_faults.a"
[[ -r "${D1_FAULT_LIBRARY}" ]] || die "D1 fault-test archive is missing"
[[ "$(ar t "${D1_FAULT_LIBRARY}")" == "data_io.o" ]] || \
  die "D1 fault-test archive has unexpected members"
for forbidden_fault_text in ET_D1_TEST_FAULT ET_D1_TEST_FAIL_CALL \
    short-write write-enospc write-eio close-eio; do
  if strings "${D1_LIBRARY}" | grep -F "${forbidden_fault_text}" >/dev/null; then
    die "canonical D1 native archive contains test-fault control: ${forbidden_fault_text}"
  fi
done
if nm -u "${D1_LIBRARY}" | grep -E ' (getenv|strtoull|strcmp)$' >/dev/null; then
  die "canonical D1 native archive references test-fault runtime helpers"
fi

D1_GUESSED_NATIVE_SYMBOLS=(
  d1_token_write_temporary_new
  et_d1_token_write_temporary_new_v1
  d1_token_publish_temporary
  et_d1_token_publish_temporary_v1
  d1_token_atomic_write_new
  et_d1_token_atomic_write_new_v1
  d1_token_corpus_write_impl
  et_d1_token_corpus_write_impl_v1
  d1_token_corpus_validate_impl
  et_d1_token_corpus_validate_impl_v1
  d1_token_list_ref
  et_d1_token_list_ref_v1
)
for guessed_symbol in "${D1_GUESSED_NATIVE_SYMBOLS[@]}"; do
  guessed_object="${D1_TMP}/negative-native-${guessed_symbol}.o"
  guessed_binary="${D1_TMP}/negative-native-${guessed_symbol}"
  guessed_log="${D1_TMP}/negative-native-${guessed_symbol}.log"
  "${D1_CC}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -DD1_PRIVATE_SYMBOL="${guessed_symbol}" -c \
    "${PROJECT_ROOT}/tests/d1/negative_native_symbol_link.c" \
    -o "${guessed_object}"
  if "${D1_CC}" "${guessed_object}" "${D1_LIBRARY}" \
      -Wl,--no-undefined -o "${guessed_binary}" >"${guessed_log}" 2>&1; then
    die "guessed D1 private native symbol unexpectedly linked: ${guessed_symbol}"
  fi
  grep -F "${guessed_symbol}" "${guessed_log}" >/dev/null
  test ! -e "${guessed_binary}"
done

for run in 1 2; do
  d1_compile "${PROJECT_ROOT}/tests/d1/compile_data_api.esk" \
    "${D1_TMP}/data-api-${run}.o" \
    --emit-depfile "${D1_TMP}/data-api-${run}.d" --compile-only \
    >"${D1_TMP}/data-api-${run}.stdout" 2>"${D1_TMP}/data-api-${run}.stderr"
  test -s "${D1_TMP}/data-api-${run}.o"
  grep -F 'lib/transformer/data.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'src/eshkol_transformer/token_shard.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  if grep -F 'src/eshkol_transformer/sha256.esk' \
      "${D1_TMP}/data-api-${run}.d" >/dev/null; then
    die "D1 public data graph unexpectedly imports the test-only SHA source"
  fi
  grep -F 'lib/transformer/error_public.esk' \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'lib/transformer/error_core.esk' \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
  if grep -F 'lib/transformer/error_internal.esk' \
      "${D1_TMP}/data-api-${run}.d" >/dev/null; then
    die "D1 public data graph unexpectedly includes transformer.error_internal"
  fi
done
cmp --silent "${D1_TMP}/data-api-1.o" "${D1_TMP}/data-api-2.o"
cmp --silent "${D1_TMP}/data-api-1.stdout" "${D1_TMP}/data-api-2.stdout"
for compile_log in "${D1_TMP}"/data-api-*.stdout "${D1_TMP}"/data-api-*.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compile-only gate emitted ERROR in ${compile_log}"
  fi
done
if nm -g --defined-only --format=posix "${D1_TMP}/data-api-1.o" |
    awk '{ print $1 }' | grep -E '^(d1-(token|sha)|eshkol_g_d1_2D)' \
      >/dev/null; then
  die "nm found a globally defined private D1 binding in the public consumer object"
fi
if readelf --wide --symbols "${D1_TMP}/data-api-1.o" |
    awk '$5 == "GLOBAL" && $7 != "UND" { print $8 }' |
    grep -E '^(d1-(token|sha)|eshkol_g_d1_2D)' >/dev/null; then
  die "readelf found a globally defined private D1 binding in the public consumer object"
fi
if strings -a "${D1_TMP}/data-api-1.o" |
    grep -E '(d1-token-(write-temporary-new!|publish-temporary!|atomic-write-new!|corpus-(write|validate)-impl|list-(length|ref))(_sexpr|__eshkol)|eshkol_g_d1_2D)' \
      >/dev/null; then
  die "public consumer object contains a compiled private D1 symbol name"
fi

D1_GUESSED_ESHKOL_SYMBOLS=(
  d1-token-write-temporary-new!
  d1-token-publish-temporary!
  d1-token-atomic-write-new!
  d1-token-corpus-write-impl
  d1-token-corpus-validate-impl
  d1-token-list-ref
  d1-sha256
  d1-token-checked-write-new
  d1-token-write-temporary-new!_sexpr
  eshkol_g_d1_2Dtoken_2Dsummary_2Dpublic_2Doperations
  d1-compiled-public-operations
  eshkol_g_d1_2Dcompiled_2Dpublic_2Doperations
)
guessed_index=0
for guessed_symbol in "${D1_GUESSED_ESHKOL_SYMBOLS[@]}"; do
  guessed_index=$((guessed_index + 1))
  guessed_object="${D1_TMP}/negative-eshkol-link-${guessed_index}.o"
  guessed_combined="${D1_TMP}/negative-eshkol-link-${guessed_index}-combined.o"
  "${D1_CC}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -DD1_RELOCATABLE_PROBE \
    -DD1_PRIVATE_LINK_NAME="\"${guessed_symbol}\"" -c \
    "${PROJECT_ROOT}/tests/d1/negative_native_symbol_link.c" \
    -o "${guessed_object}"
  "${D1_CC}" -r "${guessed_object}" "${D1_TMP}/data-api-1.o" \
    "${D1_MODULE}" \
    -o "${guessed_combined}"
  nm -u "${guessed_combined}" | awk '{ print $NF }' |
    grep -Fx "${guessed_symbol}" >/dev/null || \
    die "guessed private Eshkol symbol resolved from public object: ${guessed_symbol}"
done

D1_PRIVATE_BINDING_NAMES=(
  d1-token-write-temporary-new!
  d1-token-publish-temporary!
  d1-token-atomic-write-new!
  d1-token-corpus-write-impl
  d1-token-corpus-validate-impl
  d1-token-list-ref
  d1-token-derived-sizes
  d1-token-reraise-or-wrap
  d1-token-summary-public-operations
  d1-compiled-public-operations
  d1-sha256
  d1-token-checked-write-new
)
D1_PRIVATE_BINDING_SOURCES=(
  write-temporary
  publish-temporary
  atomic-write
  corpus-write-impl
  corpus-validate-impl
  list-ref
  derived-sizes
  error-mapping
  summary-operations
  compiled-operations
  sha256
  checked-write-ffi
)
for hidden_index in "${!D1_PRIVATE_BINDING_NAMES[@]}"; do
  d1_assert_private_binding \
    "${D1_PRIVATE_BINDING_NAMES[${hidden_index}]}" \
    "${D1_PRIVATE_BINDING_SOURCES[${hidden_index}]}"
done

D1_DIRECT_PRIVATE_NAMES=(
  d1-token-checked-write-new
  d1-token-write-temporary-new!
  d1-token-publish-temporary!
  d1-token-atomic-write-new!
  d1-token-corpus-write-impl
  d1-token-corpus-validate-impl
  d1-token-list-ref
  d1-sha256
)
D1_DIRECT_PRIVATE_SOURCES=(
  checked-write-ffi
  write-temporary
  publish-temporary
  atomic-write
  corpus-write-impl
  corpus-validate-impl
  list-ref
  sha256
)
for hidden_index in "${!D1_DIRECT_PRIVATE_NAMES[@]}"; do
  d1_assert_unavailable_call \
    "${D1_DIRECT_PRIVATE_NAMES[${hidden_index}]}" \
    "${D1_DIRECT_PRIVATE_SOURCES[${hidden_index}]}"
done

d1_assert_public_wrong_arity token-corpus-write! writer
d1_assert_public_wrong_arity token-corpus-validate validator
d1_assert_public_wrong_arity token-corpus-summary-shard-count accessor

hidden_error_names=(
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
)
hidden_error_sources=(make raise wrap)
for hidden_index in "${!hidden_error_names[@]}"; do
  hidden_name="${hidden_error_names[${hidden_index}]}"
  hidden_source="${hidden_error_sources[${hidden_index}]}"
  for run in 1 2; do
    if D1_CACHE_HOME="${D1_TMP}/negative-error-${hidden_source}-cache-${run}" \
        d1_compile \
        "${PROJECT_ROOT}/tests/d1/negative_error_${hidden_source}_public.esk" \
        "${D1_TMP}/negative-error-${hidden_source}-${run}.o" \
        --compile-only \
        >"${D1_TMP}/negative-error-${hidden_source}-${run}.log" 2>&1; then
      die "D1 hidden E1 ${hidden_source} helper unexpectedly compiled publicly"
    fi
    grep -F "Unbound variable: ${hidden_name}" \
      "${D1_TMP}/negative-error-${hidden_source}-${run}.log" >/dev/null
    test ! -e "${D1_TMP}/negative-error-${hidden_source}-${run}.o"
  done
  cmp --silent "${D1_TMP}/negative-error-${hidden_source}-1.log" \
    "${D1_TMP}/negative-error-${hidden_source}-2.log"
done

hidden_summary_names=(d1-token-summary-create d1-token-make-summary)
hidden_summary_sources=(constructor maker)
for hidden_index in "${!hidden_summary_names[@]}"; do
  hidden_name="${hidden_summary_names[${hidden_index}]}"
  hidden_source="${hidden_summary_sources[${hidden_index}]}"
  for run in 1 2; do
    if d1_compile \
        "${PROJECT_ROOT}/tests/d1/negative_summary_${hidden_source}_public.esk" \
        "${D1_TMP}/negative-summary-${hidden_source}-${run}.o" \
        --compile-only \
        >"${D1_TMP}/negative-summary-${hidden_source}-${run}.log" 2>&1; then
      die "D1 hidden summary ${hidden_source} unexpectedly compiled publicly"
    fi
    grep -F "Unbound variable: ${hidden_name}" \
      "${D1_TMP}/negative-summary-${hidden_source}-${run}.log" >/dev/null
    test ! -e "${D1_TMP}/negative-summary-${hidden_source}-${run}.o"
  done
  cmp --silent "${D1_TMP}/negative-summary-${hidden_source}-1.log" \
    "${D1_TMP}/negative-summary-${hidden_source}-2.log"
done

d1_compile "${PROJECT_ROOT}/tests/d1/sha256_probe.esk" \
  "${D1_TMP}/sha256-probe" \
  >"${D1_TMP}/sha256.compile.stdout" 2>"${D1_TMP}/sha256.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/primitive_probe.esk" \
  "${D1_TMP}/primitive-probe" \
  >"${D1_TMP}/primitive.compile.stdout" 2>"${D1_TMP}/primitive.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${D1_TMP}/corpus-tool" \
  --emit-depfile "${D1_TMP}/corpus-tool.d" \
  >"${D1_TMP}/corpus.compile.stdout" 2>"${D1_TMP}/corpus.compile.stderr"
D1_LIBRARY_DIR="${D1_FAULT_ARTIFACT_DIR}" \
D1_LIBRARY_NAME=eshkol_transformer_d1_faults \
D1_CACHE_HOME="${D1_TMP}/fault-tool-cache" \
  d1_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${D1_TMP}/corpus-tool-faults" \
  >"${D1_TMP}/corpus-faults.compile.stdout" \
  2>"${D1_TMP}/corpus-faults.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/arithmetic_probe.esk" \
  "${D1_TMP}/arithmetic-probe" \
  >"${D1_TMP}/arithmetic.compile.stdout" 2>"${D1_TMP}/arithmetic.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/e1_mapping_probe.esk" \
  "${D1_TMP}/e1-mapping-probe" \
  >"${D1_TMP}/e1-mapping.compile.stdout" 2>"${D1_TMP}/e1-mapping.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/summary_opacity_probe.esk" \
  "${D1_TMP}/summary-opacity-probe" \
  >"${D1_TMP}/summary-opacity.compile.stdout" \
  2>"${D1_TMP}/summary-opacity.compile.stderr"

for compile_log in "${D1_TMP}"/*.compile.stdout "${D1_TMP}"/*.compile.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compiler emitted ERROR in ${compile_log}"
  fi
done

mkdir "${D1_TMP}/primitive-root"
PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/sha256-probe" >"${D1_TMP}/sha256.stdout" 2>"${D1_TMP}/sha256.stderr"
grep -Fx 'D1_SHA256_PASS' "${D1_TMP}/sha256.stdout" >/dev/null
test ! -s "${D1_TMP}/sha256.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/primitive-probe" "${D1_TMP}/primitive-root" \
  >"${D1_TMP}/primitive.stdout" 2>"${D1_TMP}/primitive.stderr"
grep -Fx 'D1_PRIMITIVES_PASS' "${D1_TMP}/primitive.stdout" >/dev/null
test ! -s "${D1_TMP}/primitive.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/arithmetic-probe" >"${D1_TMP}/arithmetic.stdout" \
  2>"${D1_TMP}/arithmetic.stderr"
grep -Fx 'D1_ARITHMETIC_PASS' "${D1_TMP}/arithmetic.stdout" >/dev/null
test ! -s "${D1_TMP}/arithmetic.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/e1-mapping-probe" "${D1_TMP}/missing-corpus" \
  >"${D1_TMP}/e1-mapping.stdout" \
  2>"${D1_TMP}/e1-mapping.stderr"
grep -Fx 'D1_E1_MAPPING_PASS' "${D1_TMP}/e1-mapping.stdout" >/dev/null
test ! -s "${D1_TMP}/e1-mapping.stderr"

mkdir "${D1_TMP}/summary-writer" "${D1_TMP}/summary-validator"
PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/summary-opacity-probe" \
  "${D1_TMP}/summary-writer" "${D1_TMP}/summary-validator" \
  >"${D1_TMP}/summary-opacity.stdout" \
  2>"${D1_TMP}/summary-opacity.stderr"
grep -Fx 'D1_SUMMARY_OPACITY_PASS' "${D1_TMP}/summary-opacity.stdout" >/dev/null
test ! -s "${D1_TMP}/summary-opacity.stderr"

PYTHONDONTWRITEBYTECODE=1 D1_CORPUS_TOOL="${D1_TMP}/corpus-tool" \
  D1_FAULT_CORPUS_TOOL="${D1_TMP}/corpus-tool-faults" \
  "${D1_PYTHON}" -m unittest discover -s "${PROJECT_ROOT}/tests/d1" \
  -p 'test_*.py' -v

if find "${PROJECT_ROOT}/src" -type f \( -name '*.py' -o -name '*.pyc' \) -print -quit | \
    grep -q .; then
  die "D1 production source contains Python"
fi

printf 'D1 PASS: deterministic native writer, strict validator, primitive probes, and format tests\n'
