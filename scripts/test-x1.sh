#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar awk cmp grep ldd nm python3 readelf sed sha256sum strings timeout; do
  require_command "${command}"
done
verify_toolchain

x1_runner="$(eshkol_build_dir)/eshkol-run"
x1_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
x1_timeout=${X1_COMPILER_TIMEOUT_SECONDS:-180}
[[ "${x1_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "X1_COMPILER_TIMEOUT_SECONDS must be a positive integer"
x1_tmp=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-x1.XXXXXX")
trap 'rm -rf -- "${x1_tmp}"' EXIT
mkdir -p "${x1_tmp}/cache"

run_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${x1_tmp}/cache/${cache_name}"
  XDG_CACHE_HOME="${x1_tmp}/cache/${cache_name}" \
    ESHKOL_CXX_COMPILER="${x1_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${x1_timeout}s" \
      "${x1_runner}" "$@"
}

provide_words() {
  sed -n '/^(provide /,/)/p' "$1" | \
    sed -e '1s/^(provide //' -e '$s/)$//' | tr '\n' ' ' | xargs
}

x1_public_config="${PROJECT_ROOT}/lib/transformer/config.esk"
x1_expected_api="config-parse config-resolve config-validate config-canonical config-fingerprint config-ref"
[[ "$(provide_words "${x1_public_config}")" == "${x1_expected_api}" ]] || \
  die "X1 public API names or order drifted"
grep -F '(require transformer.error_consumer)' "${x1_public_config}" >/dev/null
if grep -E '\(require transformer\.(error_internal|error_core)\)' \
    "${x1_public_config}" >/dev/null; then
  die "X1 public source imports privileged E1"
fi

# Inventory every private x1-* definition and prove none enters installed source.
sed -nE \
  's/^[[:space:]]*\(define(-syntax)?[[:space:]]+\(?([^[:space:]()]+).*/\2/p' \
  "${PROJECT_ROOT}/native/x1_config_private.esk" | \
  grep '^x1-' | LC_ALL=C sort -u >"${x1_tmp}/private-x1-bindings.txt"
[[ -s "${x1_tmp}/private-x1-bindings.txt" ]] || \
  die "X1 private-binding inventory is empty"
while IFS= read -r private_binding; do
  if grep -F "${private_binding}" "${x1_public_config}" >/dev/null; then
    die "X1 private binding leaked into public source: ${private_binding}"
  fi
done <"${x1_tmp}/private-x1-bindings.txt"

build_x1_package() {
  local output=$1 log=$2
  if ! E1B_COMPILER_TIMEOUT_SECONDS="${x1_timeout}" \
      "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
        "${PROJECT_ROOT}/native/x1_config_consumer_root.esk" \
        "${PROJECT_ROOT}/native/x1_config_consumer_bridge.c" \
        "${PROJECT_ROOT}/native/x1_config_private_renames.txt" \
        "${PROJECT_ROOT}/native/x1_config_public_exports.txt" \
        "${output}" "${PROJECT_ROOT}/native" >"${log}" 2>&1; then
    sed -n '1,260p' "${log}" >&2
    die "X1 E1B package build failed"
  fi
}

build_x1_package "${x1_tmp}/x1-package-1.o" "${x1_tmp}/package-1.log"
build_x1_package "${x1_tmp}/x1-package-2.o" "${x1_tmp}/package-2.log"
cmp "${x1_tmp}/x1-package-1.o" "${x1_tmp}/x1-package-2.o"
for evidence in global-defined.txt undefined.txt expected-undefined.txt \
  package-exports.txt readelf-symbols.txt link.map allowlist-provenance.tsv; do
  cmp "${x1_tmp}/x1-package-1.o.evidence/${evidence}" \
    "${x1_tmp}/x1-package-2.o.evidence/${evidence}"
done
cmp "${PROJECT_ROOT}/native/x1_config_public_exports.txt" \
  "${x1_tmp}/x1-package-1.o.evidence/package-exports.txt"
cmp "${PROJECT_ROOT}/native/x1_undefined_symbols.txt" \
  "${x1_tmp}/x1-package-1.o.evidence/expected-undefined.txt"
cmp "${x1_tmp}/x1-package-1.o.evidence/expected-undefined.txt" \
  "${x1_tmp}/x1-package-1.o.evidence/undefined.txt"
for private_source in \
  lib/transformer/error_internal.esk \
  lib/transformer/error_core.esk \
  native/e1b_error_consumer_private.esk \
  native/x1_config_private.esk; do
  grep -F "${private_source}" \
    "${x1_tmp}/x1-package-1.o.evidence/private.d" >/dev/null || \
    die "X1 package evidence omits trusted source: ${private_source}"
done

ar rcsD "${x1_tmp}/libeshkol_transformer_x1.a" \
  "${x1_tmp}/x1-package-1.o"
[[ "$(ar t "${x1_tmp}/libeshkol_transformer_x1.a")" == \
    "x1-package-1.o" ]] || die "X1 archive has unexpected members"
nm -s "${x1_tmp}/libeshkol_transformer_x1.a" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${x1_tmp}/archive-index.txt"
if grep -E 'transformer-error-(make|raise|wrap-foreign)|e1-internal-dispatch|et_e1b_(consumer|private|box|ensure)|x1-(make|resolved|unresolved|parse|validate|canonical|fingerprint)' \
    "${x1_tmp}/archive-index.txt" >/dev/null; then
  die "X1 archive index exposes a privileged symbol"
fi

compile_object() {
  local cache=$1 source=$2 output=$3 depfile=$4 log=$5
  run_compiler "${cache}" --strict-types --no-stdlib --compile-only \
    -I "${PROJECT_ROOT}/lib" --emit-depfile "${depfile}" \
    "${source}" -o "${output}" >"${log}" 2>&1
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "Eshkol reported ERROR while compiling ${source}"
  [[ -s "${output}" && -s "${depfile}" ]] || \
    die "Eshkol omitted object or depfile for ${source}"
}

compile_binary() {
  local cache=$1 source=$2 output=$3 log=$4
  run_compiler "${cache}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -L "${x1_tmp}" --lib eshkol_transformer_x1 \
    "${source}" -o "${output}" >"${log}" 2>&1
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "Eshkol reported ERROR while linking ${source}"
  [[ -x "${output}" ]] || die "Eshkol omitted executable for ${source}"
}

check_public_depfile() {
  local depfile=$1
  grep -F 'lib/transformer/config.esk' "${depfile}" >/dev/null
  grep -F 'lib/transformer/error_consumer.esk' "${depfile}" >/dev/null
  if grep -E 'error_(internal|core)\.esk|e1b_error_consumer_private\.esk|x1_config_(private|consumer_root)\.esk' \
      "${depfile}" >/dev/null; then
    die "X1 public depfile contains privileged source: ${depfile}"
  fi
}

for run in 1 2; do
  compile_object "behavior-object-${run}" \
    "${PROJECT_ROOT}/tests/x1/test_config.esk" \
    "${x1_tmp}/test-config-${run}.o" \
    "${x1_tmp}/test-config-${run}.d" \
    "${x1_tmp}/test-config-${run}.compile.log"
  check_public_depfile "${x1_tmp}/test-config-${run}.d"
  compile_binary "behavior-aot-${run}" \
    "${PROJECT_ROOT}/tests/x1/test_config.esk" \
    "${x1_tmp}/test-config-${run}" \
    "${x1_tmp}/test-config-${run}.link.log"
done
cmp "${x1_tmp}/test-config-1.o" "${x1_tmp}/test-config-2.o"
cmp "${x1_tmp}/test-config-1" "${x1_tmp}/test-config-2"

for order in config_first error_first; do
  for run in 1 2; do
    compile_object "import-${order}-object-${run}" \
      "${PROJECT_ROOT}/tests/x1/import_${order}.esk" \
      "${x1_tmp}/import-${order}-${run}.o" \
      "${x1_tmp}/import-${order}-${run}.d" \
      "${x1_tmp}/import-${order}-${run}.compile.log"
    check_public_depfile "${x1_tmp}/import-${order}-${run}.d"
    grep -F 'lib/transformer/error_public.esk' \
      "${x1_tmp}/import-${order}-${run}.d" >/dev/null
    compile_binary "import-${order}-aot-${run}" \
      "${PROJECT_ROOT}/tests/x1/import_${order}.esk" \
      "${x1_tmp}/import-${order}-${run}" \
      "${x1_tmp}/import-${order}-${run}.link.log"
    if ! timeout --foreground --signal=TERM --kill-after=5s 30s \
        "${x1_tmp}/import-${order}-${run}" \
        >"${x1_tmp}/import-${order}-${run}.stdout" \
        2>"${x1_tmp}/import-${order}-${run}.stderr"; then
      sed -n '1,120p' "${x1_tmp}/import-${order}-${run}.stdout" >&2
      sed -n '1,120p' "${x1_tmp}/import-${order}-${run}.stderr" >&2
      die "X1 import-order runtime failed: ${order} run ${run}"
    fi
    [[ ! -s "${x1_tmp}/import-${order}-${run}.stderr" ]] || \
      die "X1 import-order runtime emitted stderr: ${order} run ${run}"
  done
  cmp "${x1_tmp}/import-${order}-1.o" "${x1_tmp}/import-${order}-2.o"
  cmp "${x1_tmp}/import-${order}-1" "${x1_tmp}/import-${order}-2"
  cmp "${x1_tmp}/import-${order}-1.stdout" \
    "${x1_tmp}/import-${order}-2.stdout"
  grep -Fx "x1-import-${order//_/-}:v1" \
    "${x1_tmp}/import-${order}-1.stdout" >/dev/null
done

strings "${x1_tmp}/import-config_first-1.o" \
  >"${x1_tmp}/public-object-strings.txt"
if grep -E 'transformer-error-(make|raise|wrap-foreign)|e1-internal-dispatch|et-e1b-private-raise|et_e1b_(consumer|private|box|ensure)|x1-(make-resolved|resolved-values|parse-object)' \
    "${x1_tmp}/public-object-strings.txt" >/dev/null; then
  die "X1 public caller object contains a privileged binding"
fi
nm -g --defined-only "${x1_tmp}/import-config_first-1.o" \
  >"${x1_tmp}/public-object-defined.txt"
if grep -E 'transformer-error-(make|raise|wrap-foreign)|e1-internal-dispatch|et_e1b_(consumer|private|box|ensure)|x1-(make-resolved|resolved-values|parse-object)' \
    "${x1_tmp}/public-object-defined.txt" >/dev/null; then
  die "X1 public caller object defines a privileged global"
fi
while IFS= read -r public_export; do
  nm -u "${x1_tmp}/import-config_first-1.o" | \
    grep -F "${public_export}" >/dev/null || \
    die "X1 public object omits admitted wrapper reference: ${public_export}"
done <"${PROJECT_ROOT}/native/x1_config_public_exports.txt"
nm -g --defined-only "${x1_tmp}/import-config_first-1" \
  >"${x1_tmp}/final-global-symbols.txt"
nm -D "${x1_tmp}/import-config_first-1" \
  >"${x1_tmp}/final-dynamic-symbols.txt"
for symbol_table in final-global-symbols.txt final-dynamic-symbols.txt; do
  if grep -E 'transformer-error-(make|raise|wrap-foreign)|e1-internal-dispatch|et_e1b_(consumer|private|box|ensure)|x1-(make-resolved|resolved-values|parse-object)' \
      "${x1_tmp}/${symbol_table}" >/dev/null; then
    die "X1 final executable exposes a privileged global"
  fi
done

compile_binding_negative() {
  local order=$1 symbol=$2
  local source="${x1_tmp}/negative-binding-${order}-${symbol}.esk"
  local output="${x1_tmp}/negative-binding-${order}-${symbol}.o"
  local log="${x1_tmp}/negative-binding-${order}-${symbol}.log"
  sed "s/@X1_SYMBOL@/${symbol}/g" \
    "${PROJECT_ROOT}/tests/x1/negative_binding_${order}.esk.in" >"${source}"
  if run_compiler "binding-${order}-${symbol}" --strict-types --no-stdlib \
      --compile-only -I "${PROJECT_ROOT}/lib" "${source}" -o "${output}" \
      >"${log}" 2>&1; then
    die "X1 public binding unexpectedly resolved: ${order} ${symbol}"
  fi
  grep -F "Unbound variable: ${symbol}" "${log}" >/dev/null || \
    die "X1 binding negative failed for the wrong reason: ${symbol}"
  [[ ! -e "${output}" ]] || die "X1 binding negative emitted ${output}"
}

binding_negatives=(
  transformer-error-make transformer-error-raise
  transformer-error-wrap-foreign e1-internal-dispatch
  et-e1b-private-raise transformer-error-consumer-raise
  x1-make-resolved x1-resolved-values x1-parse-object
)
for order in config_first error_first; do
  for symbol in "${binding_negatives[@]}"; do
    compile_binding_negative "${order}" "${symbol}"
  done
done

arity_names=(config_parse config_resolve config_validate config_canonical config_fingerprint config_ref)
arity_symbols=(config-parse config-resolve config-validate config-canonical config-fingerprint config-ref)
arity_expected=(1 2 1 1 1 2)
for index in "${!arity_names[@]}"; do
  name="${arity_names[${index}]}"
  symbol="${arity_symbols[${index}]}"
  output="${x1_tmp}/negative-arity-${name}.o"
  log="${x1_tmp}/negative-arity-${name}.log"
  if run_compiler "arity-${name}" --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/x1/negative_arity_${name}.esk" \
      -o "${output}" >"${log}" 2>&1; then
    die "X1 wrong-arity fixture unexpectedly compiled: ${symbol}"
  fi
  grep -F "Arity mismatch: ${symbol} expects ${arity_expected[${index}]} arguments" \
    "${log}" >/dev/null || die "X1 wrong-arity diagnostic drifted: ${symbol}"
  [[ ! -e "${output}" ]] || die "X1 wrong-arity fixture emitted ${output}"
done

compile_link_negative() {
  local symbol=$1
  local source="${x1_tmp}/negative-extern-${symbol}.esk"
  local output="${x1_tmp}/negative-extern-${symbol}"
  local log="${x1_tmp}/negative-extern-${symbol}.log"
  sed "s/@X1_SYMBOL@/${symbol}/g" \
    "${PROJECT_ROOT}/tests/x1/negative_extern.esk.in" >"${source}"
  if run_compiler "extern-${symbol}" --strict-types --no-stdlib \
      -I "${PROJECT_ROOT}/lib" -L "${x1_tmp}" --lib eshkol_transformer_x1 \
      "${source}" -o "${output}" >"${log}" 2>&1; then
    die "X1 guessed private symbol unexpectedly linked: ${symbol}"
  fi
  grep -F "${symbol}" "${log}" >/dev/null || \
    die "X1 guessed-symbol negative failed for the wrong reason: ${symbol}"
  [[ ! -e "${output}" ]] || die "X1 guessed-symbol negative emitted ${output}"
}

link_negatives=(
  transformer-error-make transformer-error-raise transformer-error-wrap-foreign
  e1-internal-dispatch et_e1b_consumer_raise_v1
  et_e1b_private_raise_cabi_v1 et_e1b_box_value_v1
  et_e1b_private_x1_config_parse_cabi_v1 x1-make-resolved
)
for symbol in "${link_negatives[@]}"; do
  compile_link_negative "${symbol}"
done

compile_binary "emit-manifest" "${PROJECT_ROOT}/tests/x1/emit_minimal.esk" \
  "${x1_tmp}/emit-minimal" "${x1_tmp}/emit-minimal.compile.log"
compile_binary "emit-fingerprint" \
  "${PROJECT_ROOT}/tests/x1/emit_minimal_fingerprint.esk" \
  "${x1_tmp}/emit-fingerprint" "${x1_tmp}/emit-fingerprint.compile.log"

[[ ! -e "${PROJECT_ROOT}/should-not-run" && ! -e "${x1_tmp}/should-not-run" ]] || \
  die "hostile-input sentinel already exists"
if ! env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV='$(touch should-not-run)' \
  PYTHON=/definitely/unavailable PYTHONPATH=/definitely/unavailable \
  timeout --foreground --signal=TERM --kill-after=2s 30s \
  "${x1_tmp}/test-config-1" >"${x1_tmp}/test-1.stdout"; then
  sed -n '1,260p' "${x1_tmp}/test-1.stdout" >&2
  die "X1 native behavior test failed"
fi
(
  cd -- "${x1_tmp}"
  env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV='${HOME}:include:eval' \
    PYTHON=/another/unavailable PYTHONPATH=/another/unavailable \
    timeout --foreground --signal=TERM --kill-after=2s 30s \
    "${x1_tmp}/test-config-1" >"${x1_tmp}/test-2.stdout"
)
cmp "${x1_tmp}/test-1.stdout" "${x1_tmp}/test-2.stdout"
[[ ! -e "${PROJECT_ROOT}/should-not-run" && ! -e "${x1_tmp}/should-not-run" ]] || \
  die "executable-looking configuration input created a sentinel"
grep -Fx 'X1 SUMMARY: 111 passed, 0 failed' "${x1_tmp}/test-1.stdout" >/dev/null || \
  die "X1 native behavior summary did not report all tests passing"

env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=first \
  timeout --foreground --signal=TERM --kill-after=2s 30s \
  "${x1_tmp}/emit-minimal" >"${x1_tmp}/resolved-1.json"
env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=first \
  timeout --foreground --signal=TERM --kill-after=2s 30s \
  "${x1_tmp}/emit-fingerprint" >"${x1_tmp}/resolved-1.sha256"
(
  cd -- "${x1_tmp}"
  env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=second \
    "${x1_tmp}/emit-minimal" >"${x1_tmp}/resolved-2.json"
  env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=second \
    "${x1_tmp}/emit-fingerprint" >"${x1_tmp}/resolved-2.sha256"
)
cmp "${x1_tmp}/resolved-1.json" "${x1_tmp}/resolved-2.json"
cmp "${x1_tmp}/resolved-1.sha256" "${x1_tmp}/resolved-2.sha256"
cmp "${PROJECT_ROOT}/tests/x1/fixtures/resolved_minimal_v1.json" \
  "${x1_tmp}/resolved-1.json"
cmp "${PROJECT_ROOT}/tests/x1/fixtures/resolved_minimal_v1.sha256" \
  "${x1_tmp}/resolved-1.sha256"
x1_digest=$(sha256sum "${x1_tmp}/resolved-1.json" | awk '{print $1}')
grep -Fx "sha256:eshkol-config-json-v1:${x1_digest}" \
  "${x1_tmp}/resolved-1.sha256" >/dev/null || \
  die "X1 fingerprint does not cover exact canonical bytes"

if ldd "${x1_tmp}/test-config-1" | grep -Eiq 'python|torch'; then
  die "X1 native binary links a Python or Torch runtime"
fi
if grep -Ein 'python|pytorch|torch|fallback' \
    "${PROJECT_ROOT}/lib/transformer/config.esk" \
    "${PROJECT_ROOT}/native/x1_config_private.esk" \
    "${PROJECT_ROOT}/native/x1_config_consumer_root.esk" \
    "${PROJECT_ROOT}/native/x1_config_consumer_bridge.c"; then
  die "X1 production path contains Python, PyTorch, or a fallback"
fi

export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "${PROJECT_ROOT}/tests/x1" -p 'test_*.py' -v

printf 'X1 PASS: 111 native semantics; fresh source/object/AOT, import-order, E1B admission, canonical/reference, and leakage evidence verified\n'
