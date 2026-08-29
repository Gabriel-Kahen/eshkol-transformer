#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cmp grep nm readelf strings timeout; do
  require_command "${command}"
done
verify_toolchain

e1b_runner="$(eshkol_build_dir)/eshkol-run"
e1b_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
e1b_compile_timeout="${E1B_COMPILER_TIMEOUT_SECONDS:-180}"
e1b_runtime_timeout="${E1B_RUNTIME_TIMEOUT_SECONDS:-10}"
[[ "${e1b_compile_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E1B_COMPILER_TIMEOUT_SECONDS must be a positive integer"
[[ "${e1b_runtime_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E1B_RUNTIME_TIMEOUT_SECONDS must be a positive integer"

e1b_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-e1b-test.XXXXXX")"
trap 'rm -rf -- "${e1b_tmp}"' EXIT
trap 'die "E1B evidence command failed at line ${LINENO}"' ERR
mkdir -p "${e1b_tmp}/cache"

run_compiler() {
  local cache_name="$1"
  shift
  mkdir -p "${e1b_tmp}/cache/${cache_name}"
  XDG_CACHE_HOME="${e1b_tmp}/cache/${cache_name}" \
    ESHKOL_CXX_COMPILER="${e1b_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${e1b_compile_timeout}s" "${e1b_runner}" "$@"
}

run_program() {
  timeout --foreground --signal=TERM --kill-after=5s \
    "${e1b_runtime_timeout}s" "$@"
}

provide_words() {
  sed -n '/^(provide /,/)/p' "$1" | \
    sed -e '1s/^(provide //' -e '$s/)$//' | tr '\n' ' ' | xargs
}

public_exports="transformer-error? transformer-error-category transformer-error-operation transformer-error-message transformer-error-details transformer-error-cause"
[[ "$(provide_words "${PROJECT_ROOT}/lib/transformer/error_consumer.esk")" == \
    "${public_exports}" ]] || die "E1B public accessor surface drifted"

for public_stub in \
  "${PROJECT_ROOT}/lib/transformer/error_consumer.esk" \
  "${PROJECT_ROOT}/tests/fixtures/e1b/public/transformer/consumer_a.esk" \
  "${PROJECT_ROOT}/tests/fixtures/e1b/public/transformer/consumer_b.esk" \
  "${PROJECT_ROOT}/tests/fixtures/e1b/public/transformer/consumer_probe.esk"; do
  if grep -E '\(require transformer\.(error_internal|error_core)\)' \
      "${public_stub}" >/dev/null; then
    die "E1B installed public source imports privileged E1: ${public_stub}"
  fi
done

build_package() {
  local output="$1" log="$2"
  E1B_COMPILER_TIMEOUT_SECONDS="${e1b_compile_timeout}" \
    "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
      "${PROJECT_ROOT}/tests/fixtures/e1b/trusted_fixture_package.esk" \
      "${PROJECT_ROOT}/tests/fixtures/e1b/fixture_package_bridge.c" \
      "${PROJECT_ROOT}/tests/fixtures/e1b/fixture_package_renames.txt" \
      "${PROJECT_ROOT}/tests/fixtures/e1b/fixture_public_exports.txt" \
      "${output}" "${PROJECT_ROOT}/tests/fixtures/e1b/public" \
      >"${log}" 2>&1
}

build_package "${e1b_tmp}/package-1.o" "${e1b_tmp}/build-1.log"
build_package "${e1b_tmp}/package-2.o" "${e1b_tmp}/build-2.log"
cmp "${e1b_tmp}/package-1.o" "${e1b_tmp}/package-2.o"
cmp "${e1b_tmp}/package-1.o.evidence/global-defined.txt" \
  "${e1b_tmp}/package-2.o.evidence/global-defined.txt"
cmp "${e1b_tmp}/package-1.o.evidence/undefined.txt" \
  "${e1b_tmp}/package-2.o.evidence/undefined.txt"
cmp "${e1b_tmp}/package-1.o.evidence/readelf-symbols.txt" \
  "${e1b_tmp}/package-2.o.evidence/readelf-symbols.txt"
cmp "${e1b_tmp}/package-1.o.evidence/link.map" \
  "${e1b_tmp}/package-2.o.evidence/link.map"

grep -F 'lib/transformer/error_internal.esk' \
  "${e1b_tmp}/package-1.o.evidence/private.d" >/dev/null
grep -F 'lib/transformer/error_core.esk' \
  "${e1b_tmp}/package-1.o.evidence/private.d" >/dev/null
grep -F 'native/e1b_error_consumer_private.esk' \
  "${e1b_tmp}/package-1.o.evidence/private.d" >/dev/null
grep -F 'et_e1b_consumer_raise_v1' \
  "${e1b_tmp}/package-1.o.evidence/strings.txt" >/dev/null
grep -F 'e1-internal-dispatch' \
  "${e1b_tmp}/package-1.o.evidence/strings.txt" >/dev/null

if find "${e1b_tmp}" -maxdepth 2 -type f \
    \( -name '*.ll' -o -name '*.bc' -o -name '*.raw.o' \
       -o -name 'private.o' -o -name 'bridge.o' \) | grep . >/dev/null; then
  die "E1B builder published a private intermediate"
fi

ar rcsD "${e1b_tmp}/libe1b_fixture.a" "${e1b_tmp}/package-1.o"
nm -s "${e1b_tmp}/libe1b_fixture.a" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { exit }
       in_index { print }' >"${e1b_tmp}/archive-index.txt"
if grep -E 'et_e1b_(consumer|private|box|ensure)|e1-internal-dispatch|transformer-error-(make|raise|wrap-foreign)' \
    "${e1b_tmp}/archive-index.txt" >/dev/null; then
  die "E1B archive index exposed privileged symbols"
fi

compile_public_app() {
  local run="$1" source="$2" output="$3"
  run_compiler "app-${run}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/tests/fixtures/e1b/public" \
    -L "${e1b_tmp}" --lib e1b_fixture \
    "${source}" -o "${output}"
}

compile_public_object() {
  local run="$1" source="$2" output="$3" depfile="$4"
  run_compiler "app-object-${run}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/tests/fixtures/e1b/public" \
    --emit-depfile "${depfile}" --compile-only "${source}" -o "${output}"
}

for run in 1 2; do
  compile_public_object "${run}" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/applications/public_consumer.esk" \
    "${e1b_tmp}/public-consumer-${run}.o" \
    "${e1b_tmp}/public-consumer-${run}.d" \
    >"${e1b_tmp}/object-${run}.stdout" \
    2>"${e1b_tmp}/object-${run}.stderr"
  compile_public_app "${run}" \
    "${PROJECT_ROOT}/tests/fixtures/e1b/applications/public_consumer.esk" \
    "${e1b_tmp}/public-consumer-${run}" \
    >"${e1b_tmp}/app-${run}.stdout" 2>"${e1b_tmp}/app-${run}.stderr"
  run_program "${e1b_tmp}/public-consumer-${run}" \
    >"${e1b_tmp}/run-${run}.stdout" 2>"${e1b_tmp}/run-${run}.stderr"
done
cmp "${e1b_tmp}/public-consumer-1.o" "${e1b_tmp}/public-consumer-2.o"
cmp "${e1b_tmp}/public-consumer-1" "${e1b_tmp}/public-consumer-2"
cmp "${e1b_tmp}/run-1.stdout" "${e1b_tmp}/run-2.stdout"
cmp "${PROJECT_ROOT}/tests/expected/e1b.stdout" \
  "${e1b_tmp}/run-1.stdout"
for run in 1 2; do
  if [[ -s "${e1b_tmp}/run-${run}.stderr" ]]; then
    sed -n '1,120p' "${e1b_tmp}/run-${run}.stderr" >&2
    die "E1B AOT run ${run} produced unexpected stderr"
  fi
done

for required_public_source in \
  lib/transformer/error_consumer.esk \
  tests/fixtures/e1b/public/transformer/consumer_a.esk \
  tests/fixtures/e1b/public/transformer/consumer_b.esk \
  tests/fixtures/e1b/public/transformer/consumer_probe.esk; do
  grep -F "${required_public_source}" "${e1b_tmp}/public-consumer-1.d" >/dev/null
done
if grep -E 'error_(internal|core)\.esk|e1b_error_consumer_private\.esk|trusted_fixture_package\.esk' \
    "${e1b_tmp}/public-consumer-1.d" >/dev/null; then
  die "E1B public application depfile contains privileged source"
fi
strings "${e1b_tmp}/public-consumer-1.o" >"${e1b_tmp}/public-strings.txt"
if grep -E 'e1-internal-dispatch|et_e1b_(consumer|private|box|ensure)' \
    "${e1b_tmp}/public-strings.txt" >/dev/null; then
  die "E1B public application object contains a privileged binding"
fi

compile_public_app reverse \
  "${PROJECT_ROOT}/tests/fixtures/e1b/applications/public_consumer_reverse.esk" \
  "${e1b_tmp}/public-consumer-reverse" \
  >"${e1b_tmp}/reverse-compile.stdout" \
  2>"${e1b_tmp}/reverse-compile.stderr"
run_program "${e1b_tmp}/public-consumer-reverse" \
  >"${e1b_tmp}/reverse.stdout" 2>"${e1b_tmp}/reverse.stderr"
grep -Fx 'e1b-import-both-reverse:v1' "${e1b_tmp}/reverse.stdout" >/dev/null
test ! -s "${e1b_tmp}/reverse.stderr"

if nm -g --defined-only "${e1b_tmp}/public-consumer-1" | \
    grep -E 'et_e1b_(consumer|private|box|ensure)|e1-internal-dispatch|transformer-error-(make|raise|wrap-foreign)' >/dev/null; then
  die "E1B final executable exposed privileged global definitions"
fi
if nm -D "${e1b_tmp}/public-consumer-1" | \
    grep -E 'et_e1b_(consumer|private|box|ensure)|e1-internal-dispatch|transformer-error-(make|raise|wrap-foreign)' >/dev/null; then
  die "E1B final executable exposed a privileged dynamic symbol"
fi

binding_negatives=(
  negative_error_make_binding
  negative_error_raise_binding
  negative_error_wrap_binding
  negative_dispatch_binding
  negative_lexical_helper_binding
  negative_accessor_wrong_arity
  negative_consumer_wrong_arity
)
binding_needles=(
  'Unbound variable: transformer-error-make'
  'Unbound variable: transformer-error-raise'
  'Unbound variable: transformer-error-wrap-foreign'
  'Unbound variable: e1-internal-dispatch'
  'Unbound variable: et-e1b-private-raise'
  'Arity mismatch: transformer-error-category expects 1 arguments but got 0'
  'Arity mismatch: consumer-a-raise expects 1 arguments but got 0'
)
for index in "${!binding_negatives[@]}"; do
  name="${binding_negatives[${index}]}"
  if run_compiler "negative-${name}" --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/tests/fixtures/e1b/public" \
      "${PROJECT_ROOT}/tests/fixtures/e1b/applications/${name}.esk" \
      -o "${e1b_tmp}/${name}.o" >"${e1b_tmp}/${name}.log" 2>&1; then
    die "E1B ${name} unexpectedly compiled"
  fi
  grep -F "${binding_needles[${index}]}" "${e1b_tmp}/${name}.log" >/dev/null
  test ! -e "${e1b_tmp}/${name}.o"
done

for arity in too_few too_many; do
  if run_compiler "boundary-${arity}" --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      "${PROJECT_ROOT}/tests/fixtures/e1b/negative_boundary_${arity}.esk" \
      -o "${e1b_tmp}/boundary-${arity}.o" \
      >"${e1b_tmp}/boundary-${arity}.log" 2>&1; then
    die "E1B five-value seam ${arity} unexpectedly compiled"
  fi
  grep -F 'expects 5 arguments' \
    "${e1b_tmp}/boundary-${arity}.log" >/dev/null
  test ! -e "${e1b_tmp}/boundary-${arity}.o"
done

link_negatives=(
  negative_generic_raise_extern
  negative_private_cabi_extern
  negative_bridge_helper_extern
  negative_private_variable_extern
  negative_constructor_extern
  negative_dispatch_extern
)
link_needles=(
  et_e1b_consumer_raise_v1
  et_e1b_private_fixture_consumer_a_raise_cabi_v1
  et_e1b_box_value_v1
  et_e1b_private_raise_cabi_v1
  transformer-error-raise
  e1-internal-dispatch
)
for index in "${!link_negatives[@]}"; do
  name="${link_negatives[${index}]}"
  if run_compiler "link-${name}" --strict-types --no-stdlib \
      -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/tests/fixtures/e1b/public" \
      -L "${e1b_tmp}" --lib e1b_fixture \
      "${PROJECT_ROOT}/tests/fixtures/e1b/applications/${name}.esk" \
      -o "${e1b_tmp}/${name}" >"${e1b_tmp}/${name}.log" 2>&1; then
    die "E1B guessed symbol ${name} unexpectedly linked"
  fi
  grep -F "${link_needles[${index}]}" "${e1b_tmp}/${name}.log" >/dev/null
  test ! -e "${e1b_tmp}/${name}"
done

if find "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/native" -type f \
    \( -name '*.esk' -o -name '*.c' -o -name '*.h' \) -print0 | \
    xargs -0 grep -Ei '(^|[^[:alnum:]_])(python|pytorch|torch)([^[:alnum:]_]|$)' \
      >/dev/null; then
  die "E1B production Eshkol/native path references Python or PyTorch"
fi
if grep -Ei '(^|[^[:alnum:]_])fallback([^[:alnum:]_]|$)' \
    "${PROJECT_ROOT}/lib/transformer/error_consumer.esk" \
    "${PROJECT_ROOT}/native/e1b_error_consumer_bridge.c" \
    "${PROJECT_ROOT}/native/e1b_error_consumer_private.esk" >/dev/null; then
  die "E1B production boundary contains a fallback path"
fi

printf 'E1B PASS: 35 public AOT checks; reusable prelocalized boundary and adversarial artifact evidence verified\n'
