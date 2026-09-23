#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar awk cmp diff env grep ldd nm python3 rg sort strings \
    timeout tr wc; do
  require_command "${command}"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${C2_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "C2_COMPILER_TIMEOUT_SECONDS must be a positive integer"

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-c2-public.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT
artifact_dir="$(project_build_dir)/c2"
object="${artifact_dir}/c2_wave2.o"
library="${artifact_dir}/libeshkol_transformer_wave2.a"
evidence="${object}.evidence"

[[ -r "${object}" && -r "${library}" ]] || \
  die "canonical C2 aggregate is missing; run scripts/build-c2.sh"
[[ "$(ar t "${library}")" == "c2_wave2.o" ]] || \
  die "canonical C2 archive must contain exactly c2_wave2.o"
ar p "${library}" c2_wave2.o >"${temporary_dir}/canonical-archive-member.o"
cmp "${object}" "${temporary_dir}/canonical-archive-member.o"
if nm -a "${object}" | \
    rg 'et_k2_test_(require_count|require_shape_count|fail_require_at)_v1' \
      >/dev/null; then
  die "canonical C2 aggregate contains a private K2 test hook"
fi

evidence_names=(
  global-defined.txt package-exports.txt undefined.txt expected-undefined.txt
  public-strings.txt source-closure.txt native-source-closure.txt
  readelf-symbols.txt nm.txt strings.txt private.d link.map
  allowlist-provenance.tsv
)
for evidence_name in "${evidence_names[@]}"; do
  [[ -s "${evidence}/${evidence_name}" ]] || \
    die "C2 aggregate evidence omits ${evidence_name}"
done

manifest_pairs=(
  c2_wave2_defined_symbols.txt:global-defined.txt
  c2_wave2_public_exports.txt:package-exports.txt
  c2_wave2_public_strings.txt:public-strings.txt
  c2_wave2_undefined_symbols.txt:undefined.txt
  c2_wave2_source_closure.txt:source-closure.txt
  c2_wave2_native_source_closure.txt:native-source-closure.txt
)
for pair in "${manifest_pairs[@]}"; do
  manifest="${pair%%:*}"
  generated="${pair#*:}"
  cmp "${PROJECT_ROOT}/native/${manifest}" "${evidence}/${generated}"
done
cmp "${evidence}/undefined.txt" "${evidence}/expected-undefined.txt"
grep -Fx $'package_policy\tc2-wave2-aggregate' \
  "${evidence}/allowlist-provenance.tsv" >/dev/null

for manifest in c2_wave2_defined_symbols.txt c2_wave2_public_exports.txt \
    c2_wave2_public_strings.txt c2_wave2_undefined_symbols.txt; do
  LC_ALL=C sort -c "${PROJECT_ROOT}/native/${manifest}"
done
awk 'NF != 2 { bad = 1 } END { exit bad ? 1 : 0 }' \
  "${PROJECT_ROOT}/native/c2_wave2_private_renames.txt" || \
  die "C2 private rename manifest must contain exact symbol pairs"
[[ "$(wc -l <"${evidence}/global-defined.txt")" == 81 ]] || \
  die "C2 aggregate must expose exactly 81 global definitions"
[[ "$(wc -l <"${evidence}/package-exports.txt")" == 75 ]] || \
  die "C2 aggregate must expose exactly 75 package exports"
[[ "$(wc -l <"${evidence}/public-strings.txt")" == 81 ]] || \
  die "C2 aggregate must contain exactly 81 public-name strings"
[[ "$(wc -l <"${evidence}/source-closure.txt")" == 32 ]] || \
  die "C2 aggregate must have exactly 32 trusted Eshkol sources"
[[ "$(wc -l <"${evidence}/native-source-closure.txt")" == 53 ]] || \
  die "C2 aggregate must have exactly 53 trusted native sources"
[[ "$(wc -l <"${PROJECT_ROOT}/native/c2_wave2_private_renames.txt")" == 6 ]] || \
  die "C2 aggregate must have exactly six private facade renames"
if grep -Ev '^et_e1b_(error|public)_[a-z0-9_]+_v1$' \
    "${evidence}/global-defined.txt" >/dev/null; then
  die "C2 aggregate exposes a global outside the E1B public namespace"
fi
if nm -g --defined-only --format=posix "${library}" | \
    grep -E '^eshkol_transformer_kernel_provider_v1[[:space:]]' >/dev/null; then
  die "C2 archive defines the forbidden canonical K1 provider symbol"
fi
nm -s "${library}" | \
  awk '/^Archive index:$/ { in_index = 1; next }
       in_index && /^$/ { in_index = 0; next }
       in_index { print }' >"${temporary_dir}/archive-index.txt"
if grep -E 'et_(c2|k2|o2|d2|i2|f32|p1|kernel)_|et_e1b_private_|c2-' \
    "${temporary_dir}/archive-index.txt" >/dev/null; then
  die "C2 archive index exposes a localized authority"
fi

run_compiler() {
  local cache_name=$1
  shift
  mkdir -p "${temporary_dir}/compiler-cache/${cache_name}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${temporary_dir}/compiler-cache/${cache_name}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" "$@"
}

for fixture_set in a b; do
  PYTHONDONTWRITEBYTECODE=1 python3 -m tests.c2.prepare_checkpoint_load_fixtures \
    "${temporary_dir}/fixtures-${fixture_set}"
done
diff -ru "${temporary_dir}/fixtures-a" "${temporary_dir}/fixtures-b"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_operational_fixtures.py" \
  "${temporary_dir}/model-fixtures"

for repetition in 1 2; do
  contract_dir="${temporary_dir}/public-contract-${repetition}"
  runtime_dir="${temporary_dir}/public-runtime-${repetition}"
  mkdir -p "${contract_dir}" "${runtime_dir}"
  run_compiler "public-contract-${repetition}" \
    --strict-types --no-stdlib --compile-only -I "${PROJECT_ROOT}/lib" \
    "${PROJECT_ROOT}/tests/c2/compile_public_api.esk" \
    -o "${contract_dir}/api.o" \
    >"${contract_dir}/compile.log" 2>&1
  run_compiler "public-aot-${repetition}" \
    --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
    "${PROJECT_ROOT}/tests/c2/public_runtime.esk" \
    -o "${runtime_dir}/c2-public-runtime" \
    >"${runtime_dir}/compile.log" 2>&1
  ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${runtime_dir}/c2-public-runtime" \
      "${temporary_dir}/fixtures-a/valid.c2" \
      "${runtime_dir}/roundtrip.c2" \
      "${temporary_dir}/fixtures-a/nonf32-buffer.c2" \
      >"${runtime_dir}/run.stdout" 2>"${runtime_dir}/run.stderr"
  [[ ! -s "${runtime_dir}/run.stderr" ]] || \
    die "C2 public runtime wrote stderr on repetition ${repetition}"
  grep -Fx 'C2 PUBLIC RUNTIME PASS: 10 checks' \
    "${runtime_dir}/run.stdout" >/dev/null
  cmp "${temporary_dir}/fixtures-a/valid.c2" \
    "${runtime_dir}/roundtrip.c2"
done

models_dir="${temporary_dir}/public-models"
mkdir -p "${models_dir}"
run_compiler public-models-aot \
  --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
  -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
  "${PROJECT_ROOT}/tests/c2/public_models_runtime.esk" \
  -o "${models_dir}/c2-public-models" \
  >"${models_dir}/compile.log" 2>&1
ESHKOL_ARENA_POISON=1 \
  timeout --foreground --signal=TERM --kill-after=5s 180s \
  "${models_dir}/c2-public-models" \
    "${temporary_dir}/model-fixtures/m3-model.c2" \
    "${models_dir}/m3-roundtrip.c2" \
    "${temporary_dir}/model-fixtures/c4-schema.c2" \
    "${models_dir}/c4-roundtrip.c2" \
    >"${models_dir}/run.stdout" 2>"${models_dir}/run.stderr"
[[ ! -s "${models_dir}/run.stderr" ]] || \
  die "C2 installed-facade model witness wrote stderr"
grep -Fx 'C2 PUBLIC MODELS PASS: 8 checks' \
  "${models_dir}/run.stdout" >/dev/null
cmp "${temporary_dir}/model-fixtures/m3-model.c2" \
  "${models_dir}/m3-roundtrip.c2"
cmp "${temporary_dir}/model-fixtures/c4-schema.c2" \
  "${models_dir}/c4-roundtrip.c2"

cmp "${temporary_dir}/public-contract-1/api.o" \
  "${temporary_dir}/public-contract-2/api.o"
cmp "${temporary_dir}/public-runtime-1/c2-public-runtime" \
  "${temporary_dir}/public-runtime-2/c2-public-runtime"
cmp "${temporary_dir}/public-runtime-1/run.stdout" \
  "${temporary_dir}/public-runtime-2/run.stdout"

grep -E '^et_e1b_public_(c1_persistence_policy|c2_)' \
  "${PROJECT_ROOT}/native/c2_wave2_public_exports.txt" \
  >"${temporary_dir}/expected-c2-public-undefined.txt"
nm -u --format=posix "${temporary_dir}/public-contract-1/api.o" | \
  awk '$1 ~ /^et_e1b_public_(c1_persistence_policy|c2_)/ { print $1 }' | \
  LC_ALL=C sort -u >"${temporary_dir}/actual-c2-public-undefined.txt"
cmp "${temporary_dir}/expected-c2-public-undefined.txt" \
  "${temporary_dir}/actual-c2-public-undefined.txt"
[[ "$(wc -l <"${temporary_dir}/actual-c2-public-undefined.txt")" == 6 ]] || \
  die "C2 public contract object must use exactly six C1/C2 bridge symbols"
if nm --format=posix "${temporary_dir}/public-contract-1/api.o" | \
    awk '{ print $1 }' | grep -E '^et_(c2_private|e1b_private_c2)_' \
      >/dev/null; then
  die "C2 public contract object references localized authority"
fi

ldd "${temporary_dir}/public-runtime-1/c2-public-runtime" \
  >"${temporary_dir}/public-runtime.ldd"
if grep -Ei 'python|torch' "${temporary_dir}/public-runtime.ldd" >/dev/null; then
  die "C2 production executable links a Python/PyTorch runtime"
fi
if nm -u --format=posix "${object}" | \
    grep -Ei '(^|[[:space:]])(_?Py|python|torch)' >/dev/null; then
  die "C2 production object depends on a Python/PyTorch runtime"
fi
if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/lib/transformer/persistence.esk" \
    "${PROJECT_ROOT}/lib/transformer/trainer.esk" \
    "${PROJECT_ROOT}/native/c2_wave2_root.esk" \
    "${PROJECT_ROOT}/native/c2_public_extension.esk" \
    "${PROJECT_ROOT}/native/c2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/c2_wave2_public_wrappers.inc"; then
  die "C2 production facade/compositor source contains Python/PyTorch"
fi

retention_dir="${temporary_dir}/arena-retention"
mkdir -p "${retention_dir}"
run_compiler arena-retention-aot \
  --strict-types --no-stdlib -O 2 -I "${PROJECT_ROOT}/lib" \
  -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
  "${PROJECT_ROOT}/tests/c2/arena_retention.esk" \
  -o "${retention_dir}/c2-arena-retention" \
  >"${retention_dir}/compile.log" 2>&1
for horizon in 1024 8192; do
  ESHKOL_ARENA_REPORT=1 \
    timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${retention_dir}/c2-arena-retention" "${horizon}" \
      "${temporary_dir}/fixtures-a/valid.c2" \
      >"${retention_dir}/${horizon}.stdout" \
      2>"${retention_dir}/${horizon}.stderr"
  grep -Fx "C2 ARENA RETENTION PASS: ${horizon} policy/metadata iterations" \
    "${retention_dir}/${horizon}.stdout" >/dev/null
  retained_bytes="$(awk -F= '/global_total_allocated_bytes=/{print $2}' \
    "${retention_dir}/${horizon}.stderr" | tail -1)"
  [[ "${retained_bytes}" =~ ^[0-9]+$ ]] || \
    die "C2 ${horizon} run omitted the Eshkol retained root-arena counter"
  if [[ "${horizon}" == 1024 ]]; then
    arena_1024="${retained_bytes}"
  else
    arena_8192="${retained_bytes}"
  fi
done
[[ "${arena_1024}" == "${arena_8192}" ]] || \
  die "C2 retained root arena grew from ${arena_1024} to ${arena_8192} bytes"
printf 'C2 ARENA RETENTION PASS: 1024=%s bytes 8192=%s bytes slope=0\n' \
  "${arena_1024}" "${arena_8192}"

arity_stems=(
  persistence_policy checkpoint_inspect checkpoint_metadata_ref
  checkpoint_load checkpoint_save trainer_state_release
)
arity_symbols=(
  persistence-policy checkpoint-inspect checkpoint-metadata-ref
  checkpoint-load checkpoint-save! trainer-state-release!
)
arity_expected=(5 2 2 3 4 1)
for index in "${!arity_stems[@]}"; do
  stem="${arity_stems[${index}]}"
  output="${temporary_dir}/negative-${stem}.o"
  log="${temporary_dir}/negative-${stem}.log"
  if run_compiler "negative-${stem}" \
      --strict-types --no-stdlib --compile-only -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/c2/negative_${stem}_arity.esk" \
      -o "${output}" >"${log}" 2>&1; then
    die "C2 wrong-arity fixture unexpectedly compiled: ${stem}"
  fi
  if [[ "${stem}" == trainer_state_release ]]; then
    grep -F "function 'trainer-state-release!' expects 1 arguments, got 2" \
      "${log}" >/dev/null || \
      die "C2 wrong-arity diagnostic drifted: trainer-state-release!"
  else
    grep -F "Arity mismatch: ${arity_symbols[${index}]} expects ${arity_expected[${index}]} arguments" \
      "${log}" >/dev/null || \
      die "C2 wrong-arity diagnostic drifted: ${arity_symbols[${index}]}"
  fi
  [[ ! -e "${output}" ]] || \
    die "C2 wrong-arity fixture emitted ${output}"
done

private_bindings=(
  c2-public-checkpoint-load
  c2-public-capability-admit-one
  c2-public-capability-admit-all
  c2-checkpoint-load-stage-internal
  c2-training-state-compose-tag
  c2-training-state-borrow-tag
)
for private_binding in "${private_bindings[@]}"; do
  private_output="${temporary_dir}/private-${private_binding}"
  if run_compiler "private-${private_binding}" \
      --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
      -L "${artifact_dir}" --lib eshkol_transformer_wave2 \
      -e "(begin (require transformer.persistence) (require transformer.trainer) ${private_binding})" \
      >"${private_output}.stdout" 2>"${private_output}.stderr"; then
    die "C2 public facade exposed trusted binding ${private_binding}"
  fi
  grep -F "${private_binding}" "${private_output}.stderr" >/dev/null || \
    die "C2 private-binding rejection failed for the wrong reason"
done

guessed_private_source="${PROJECT_ROOT}/tests/c2/guessed_private_native.esk"
run_compiler guessed-private-object \
  --strict-types --no-stdlib --emit-object "${guessed_private_source}" \
  -o "${temporary_dir}/guessed-private-native.o" \
  >"${temporary_dir}/guessed-private-object.stdout" \
  2>"${temporary_dir}/guessed-private-object.stderr"
nm -u --format=posix "${temporary_dir}/guessed-private-native.o" | \
  awk '{ print $1 }' | \
  grep -Fx 'et_c2_private_checkpoint_load_stage_v1' >/dev/null
if run_compiler guessed-private-aot \
    --strict-types --no-stdlib -L "${artifact_dir}" \
    --lib eshkol_transformer_wave2 "${guessed_private_source}" \
    -o "${temporary_dir}/guessed-private-native" \
    >"${temporary_dir}/guessed-private-aot.stdout" \
    2>"${temporary_dir}/guessed-private-aot.stderr"; then
  die "C2 public archive linked a guessed localized native seam"
fi
grep -F 'et_c2_private_checkpoint_load_stage_v1' \
  "${temporary_dir}/guessed-private-aot.stderr" >/dev/null || \
  die "C2 guessed-native rejection failed for the wrong reason"
[[ ! -e "${temporary_dir}/guessed-private-native" ]] || \
  die "rejected C2 guessed-native AOT left an executable"

hostile_root="${PROJECT_ROOT}/native/../native/c2_wave2_root.esk"
if E1B_COMPILER_TIMEOUT_SECONDS=30 \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
      "${hostile_root}" \
      "${PROJECT_ROOT}/native/../native/c2_wave2_package_bridge.c" \
      "${PROJECT_ROOT}/native/../native/c2_wave2_private_renames.txt" \
      "${PROJECT_ROOT}/native/../native/c2_wave2_public_exports.txt" \
      "${temporary_dir}/hostile-lexical.o" \
      "${PROJECT_ROOT}/internal/p1/../p1/lib" \
      "${PROJECT_ROOT}/internal/c1/../c1/lib" \
      "${PROJECT_ROOT}/internal/t2/../t2/lib" \
      "${PROJECT_ROOT}/internal/t1/../t1/lib" \
      "${PROJECT_ROOT}/internal/d2/../d2/lib" \
      "${PROJECT_ROOT}/src/../src" \
      >"${temporary_dir}/hostile-lexical.stdout" \
      2>"${temporary_dir}/hostile-lexical.stderr"; then
  die "C2 aggregate accepted canonical-equivalent nonexact tuple paths"
fi
grep -F 'C2 aggregate policy requires exact lexical repository inputs' \
  "${temporary_dir}/hostile-lexical.stderr" >/dev/null || \
  die "C2 hostile lexical tuple was rejected for the wrong reason"
test ! -e "${temporary_dir}/hostile-lexical.o"
test ! -e "${temporary_dir}/hostile-lexical.o.evidence"

hostile_include="${temporary_dir}/hostile-include"
mkdir -p "${hostile_include}"
printf '%s\n' \
  '#ifndef ET_C2_HOSTILE_STDINT_H' \
  '#define ET_C2_HOSTILE_STDINT_H' \
  '#include_next <stdint.h>' \
  'static const char et_c2_hostile_cpath[] __attribute__((used, section(".c2_hostile_cpath"))) = "ET_C2_HOSTILE_CPATH";' \
  '#endif' >"${hostile_include}/stdint.h"

for rebuild in a b; do
  rebuild_env=(
    env
    ESHKOL_PATH="${temporary_dir}/missing-${rebuild}"
    ESHKOL_LIB_DIR="${temporary_dir}/missing-${rebuild}"
    ESHKOL_JIT_CACHE_DIR="${temporary_dir}/cache-${rebuild}"
  )
  if [[ "${rebuild}" == b ]]; then
    rebuild_env+=(
      CPATH="${hostile_include}"
      C_INCLUDE_PATH="${hostile_include}"
      CPLUS_INCLUDE_PATH="${hostile_include}"
      OBJC_INCLUDE_PATH="${hostile_include}"
      DEPENDENCIES_OUTPUT="${temporary_dir}/hostile-dependencies.out"
      SUNPRO_DEPENDENCIES="${temporary_dir}/hostile-sunpro.out hostile"
      GCC_EXEC_PREFIX="${temporary_dir}/missing-gcc-prefix/"
      COMPILER_PATH="${temporary_dir}/missing-compiler-path"
      LIBRARY_PATH="${temporary_dir}/missing-library-path"
      CLANG_CONFIG_FILE="${temporary_dir}/missing-clang.cfg"
      CCC_OVERRIDE_OPTIONS="#^-DHOSTILE_ENV"
      CCC_CC="${temporary_dir}/missing-cc"
      CCC_CXX="${temporary_dir}/missing-cxx"
    )
  fi
  "${rebuild_env[@]}" /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-c2.sh" \
    "${temporary_dir}/rebuild-${rebuild}"
done
cmp "${temporary_dir}/rebuild-a/c2_wave2.o" \
  "${temporary_dir}/rebuild-b/c2_wave2.o"
cmp "${temporary_dir}/rebuild-a/libeshkol_transformer_wave2.a" \
  "${temporary_dir}/rebuild-b/libeshkol_transformer_wave2.a"
if strings "${temporary_dir}/rebuild-b/c2_wave2.o" | \
    grep -F 'ET_C2_HOSTILE_CPATH' >/dev/null; then
  die "C2 deterministic rebuild inherited a hostile compiler include path"
fi
deterministic_evidence=(
  global-defined.txt package-exports.txt undefined.txt expected-undefined.txt
  public-strings.txt source-closure.txt native-source-closure.txt private.d
  link.map allowlist-provenance.tsv
)
for evidence_name in "${deterministic_evidence[@]}"; do
  cmp "${temporary_dir}/rebuild-a/c2_wave2.o.evidence/${evidence_name}" \
    "${temporary_dir}/rebuild-b/c2_wave2.o.evidence/${evidence_name}"
done
cmp "${object}" "${temporary_dir}/rebuild-a/c2_wave2.o"
cmp "${library}" \
  "${temporary_dir}/rebuild-a/libeshkol_transformer_wave2.a"
for rebuild in a b; do
  if nm -a "${temporary_dir}/rebuild-${rebuild}/c2_wave2.o" | \
      rg 'et_k2_test_(require_count|require_shape_count|fail_require_at)_v1' \
        >/dev/null; then
    die "rebuilt C2 aggregate contains a private K2 test hook"
  fi
done

printf '%s\n' \
  'C2 PUBLIC PACKAGING PASS: exact 81/75/81 surface, one-object archive, public load/save/release, arity and private isolation, deterministic rebuilds, native-only linkage'
