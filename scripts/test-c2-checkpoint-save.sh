#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp diff find git nm python3 rg sed sha256sum sort tee \
               timeout tr xargs; do
  require_command "${command}"
done

cc="${CC:-/usr/bin/clang}"
cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
development="${C2_SAVE_DEVELOPMENT_RUNTIME_ONLY:-0}"
if [[ "${development}" != 0 && "${development}" != 1 ]]; then
  die "C2_SAVE_DEVELOPMENT_RUNTIME_ONLY must be 0 or 1"
fi
if [[ "${development}" == 1 ]]; then
  cc=/usr/bin/clang
  cxx=/usr/bin/clang++
  tmp="${C2_DEV_ARTIFACT_DIR:-}"
  [[ -n "${tmp}" && "${tmp}" = /* && -d "${tmp}" ]] || \
    die "developer mode requires an explicit absolute C2_DEV_ARTIFACT_DIR"
  [[ -z "$(find "${tmp}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    die "developer artifact directory must be empty: ${tmp}"
  exec > >(tee "${tmp}/driver.stdout") \
    2> >(tee "${tmp}/driver.stderr" >&2)
  printf '%s\n' \
    'C2 SAVE DEVELOPER MODE: NON-ACCEPTANCE; one Clang AOT and poison run only'
else
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-checkpoint-save.XXXXXX")"
fi
active_phase=
phase_started=0
phase_begin() {
  if [[ "${development}" == 1 ]]; then
    active_phase=$1
    phase_started=${SECONDS}
  fi
}
phase_end() {
  local phase_status=${1:-0}
  if [[ "${development}" == 1 ]]; then
    printf '%s_seconds=%s status=%s\n' "${active_phase}" \
      "$((SECONDS - phase_started))" "${phase_status}" >>"${tmp}/phase-times.txt"
    active_phase=
  fi
}
cleanup() {
  local status=$?
  if [[ "${development}" == 1 ]]; then
    if [[ -n "${active_phase}" ]]; then
      printf '%s_seconds=%s status=%s\n' "${active_phase}" \
        "$((SECONDS - phase_started))" "${status}" >>"${tmp}/phase-times.txt"
    fi
    printf 'C2 SAVE DEVELOPER ARTIFACTS: %s\n' "${tmp}" >&2
    return "${status}"
  fi
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,260p' {} \; >&2 || true
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fPIC -fvisibility=hidden
  -fno-common -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native"
)

if [[ "${development}" == 0 ]]; then
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_inspect_fixtures.py" \
  "${tmp}/fixtures-a"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_inspect_fixtures.py" \
  "${tmp}/fixtures-b"
diff -ru "${tmp}/fixtures-a" "${tmp}/fixtures-b"

bridge_sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_codec.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c"
)
build_bridge_driver() {
  local compiler=$1 output=$2
  "${compiler}" "${cflags[@]}" -DET_C2_CHECKPOINT_SAVE_TESTING \
    "${bridge_sources[@]}" \
    "${PROJECT_ROOT}/tests/c2/test_checkpoint_save_bridge.c" -o "${output}"
}
build_bridge_driver "${cc}" "${tmp}/bridge-clang"
"${tmp}/bridge-clang" "${tmp}/fixtures-a/valid.c2" \
  >"${tmp}/bridge-clang.stdout"
grep -Fx 'C2 checkpoint save bridge PASS: 43 checks' \
  "${tmp}/bridge-clang.stdout" >/dev/null

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/c2/checkpoint_save_bridge_header_cpp.cpp" \
  -o "${tmp}/header-clang"
"${tmp}/header-clang"
"${cxx}" -x c++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" \
  -o "${tmp}/bridge-cxx.o"

if [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]]; then
  build_bridge_driver /usr/bin/gcc "${tmp}/bridge-gcc"
  "${tmp}/bridge-gcc" "${tmp}/fixtures-a/valid.c2" \
    >"${tmp}/bridge-gcc.stdout"
  cmp "${tmp}/bridge-clang.stdout" "${tmp}/bridge-gcc.stdout"
  /usr/bin/g++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
    -I "${PROJECT_ROOT}/native" \
    "${PROJECT_ROOT}/tests/c2/checkpoint_save_bridge_header_cpp.cpp" \
    -o "${tmp}/header-gcc"
  "${tmp}/header-gcc"
  /usr/bin/g++ -x c++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
    -Wconversion -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native" \
    -c "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" \
    -o "${tmp}/bridge-gxx.o"
fi

"${cc}" "${cflags[@]}" -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" \
  -o "${tmp}/bridge-production.o"
nm -g --defined-only --format=posix "${tmp}/bridge-production.o" | \
  awk '$1 ~ /^et_/ { print $1 }' | LC_ALL=C sort >"${tmp}/symbols"
cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_save_bridge_symbols.txt" \
  "${tmp}/symbols"

while IFS= read -r source; do
  test -f "${PROJECT_ROOT}/${source}" || die "missing native closure file: ${source}"
done <"${PROJECT_ROOT}/native/c2_checkpoint_save_native_source_closure.txt"
"${cc}" -std=c11 -I "${PROJECT_ROOT}/native" -MM \
  "${bridge_sources[@]}" "${PROJECT_ROOT}/native/checkpoint_io.c" | \
  sed -e 's/^[^:]*://' -e 's/\\//g' | tr -s '[:space:]' '\n' | \
  sed "s#^${PROJECT_ROOT}/##" | rg '^native/' | LC_ALL=C sort -u \
  >"${tmp}/native-closure"
LC_ALL=C sort -u \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_native_source_closure.txt" \
  >"${tmp}/expected-native-closure"
cmp "${tmp}/expected-native-closure" "${tmp}/native-closure"

sed -n -E 's/^\(define \((c2-(checkpoint-save|save)-[^ )]+).*/\1/p' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" | LC_ALL=C sort \
  >"${tmp}/definitions"
cmp "${PROJECT_ROOT}/tests/c2/expected/c2_checkpoint_save_definitions.txt" \
  "${tmp}/definitions"
rg -Fx '(define c2-save-stage-components-internal c2-save-stage-components-core)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
rg -U -F $'(define c2-save-pre-touch-p1-identities-internal\n  c2-save-pre-touch-p1-identities)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
rg -U -F $'(define c2-save-pre-touch-tensor-begin-internal\n  state-dict-c2-owned-tensor-borrow-begin-internal)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
rg -U -F $'(define c2-save-pre-touch-tensor-end-internal\n  state-dict-c2-owned-tensor-borrow-end-internal)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
rg -Fx '(define c2-save-measure-staged-internal c2-save-measure-staged)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
rg -Fx '(define c2-save-encode-staged-into-internal! c2-save-encode-staged-into!)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
rg -U -F $'(define c2-save-destination-allocate-internal\n  (lambda (file-bytes) (make-bytevector file-bytes 0)))' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
if rg '^\(provide|c2-checkpoint-load|c2-[^ ]*capability|[kK]2-' \
    "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" \
    "${PROJECT_ROOT}/native/c2_checkpoint_save_source_closure.txt" >/dev/null; then
  die "private C2 save composed a public/load/capability/K2 surface"
fi

compile_probe() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/probe-${label}"
  (cd "${tmp}/probe-${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/probe-cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "${tmp}/probe-${label}/save.d" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_compile_probe.esk" \
        -o "${tmp}/probe-${label}/save.o" \
        >"${tmp}/probe-${label}/compile.stdout" \
        2>"${tmp}/probe-${label}/compile.stderr")
  test ! -s "${tmp}/probe-${label}/compile.stderr"
  sed -e 's/^[^:]*://' -e 's/\\//g' "${tmp}/probe-${label}/save.d" | \
    tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
    sed "s#^${PROJECT_ROOT}/##" | \
    grep -v '^tests/c2/c2_checkpoint_save_compile_probe.esk$' \
    >"${tmp}/probe-${label}/source-closure"
  cmp "${PROJECT_ROOT}/native/c2_checkpoint_save_source_closure.txt" \
    "${tmp}/probe-${label}/source-closure"
  nm -u --format=posix "${tmp}/probe-${label}/save.o" | \
    awk '{ print $1 }' | rg '^et_c2_private_checkpoint_save_' | \
    LC_ALL=C sort -u >"${tmp}/probe-${label}/save-symbols"
  cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_save_bridge_symbols.txt" \
    "${tmp}/probe-${label}/save-symbols"
}

compile_probe clang-a "${cxx}"
compile_probe clang-b "${cxx}"
cmp "${tmp}/probe-clang-a/save.o" "${tmp}/probe-clang-b/save.o"
if [[ -x /usr/bin/g++ ]]; then compile_probe gcc /usr/bin/g++; fi
fi

rg -U -F $'(define c2-save-pre-touch-entry-shells-internal\n  state-dict-c2-owned-entry-shells-internal)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
sed -n '/^(define (c2-save-pre-touch-p1-identities /,/^;;; Focused tests/p' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" \
  >"${tmp}/pre-touch-source.esk"
rg -F '(c2-save-pre-touch-entry-shells-internal' \
  "${tmp}/pre-touch-source.esk" >/dev/null
if rg -F '(with-region' "${tmp}/pre-touch-source.esk" >/dev/null || \
   rg -F '(state-dict-c2-owned-projection-internal' \
     "${tmp}/pre-touch-source.esk" >/dev/null; then
  die "P1 identity preparation entered scratch or used the deep projection"
fi

phase_begin native_build
runtime="${tmp}/runtime"
mkdir -p "${runtime}"
runtime_cflags=("${cflags[@]}" -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
runtime_native_inputs=(
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c"
  "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c"
  "${PROJECT_ROOT}/native/p1_identity.c"
  "${PROJECT_ROOT}/native/data_io.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
  "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/f32_tensor.c"
  "${PROJECT_ROOT}/native/i64_tensor.c"
  "${PROJECT_ROOT}/native/d2_native.c"
  "${PROJECT_ROOT}/native/o2_optimizer.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_codec.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/checkpoint_io.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c"
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c"
)
"${cc}" "${runtime_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY -c \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" -o "${runtime}/i2.o"
"${cc}" "${runtime_cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -c \
  "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" -o "${runtime}/o2.o"
"${cc}" "${runtime_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -DET_P1_TEST_HOOKS=1 -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${runtime}/p1.o"
for source in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor d2_native \
              o2_optimizer c2_x1_canonical c2_checkpoint_codec \
              c2_checkpoint_format; do
  "${cc}" "${runtime_cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${runtime_cflags[@]}" -DET_CHECKPOINT_IO_TESTING -c \
  "${PROJECT_ROOT}/native/checkpoint_io.c" -o "${runtime}/checkpoint_io.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_SAVE_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" -o "${runtime}/save.o"
"${cc}" "${runtime_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c" \
  -o "${runtime}/test.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_save.a" "${runtime}"/*.o
phase_end

phase_begin resources
(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.d2.prepare_public_resources --output "${tmp}/resources-a")
if [[ "${development}" == 0 ]]; then
  (cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 python3 \
    -m tests.d2.prepare_public_resources --output "${tmp}/resources-b")
  diff -ru "${tmp}/resources-a" "${tmp}/resources-b"
fi
phase_end

write_development_manifest() {
  local depfile=$1 archive=$2
  {
    printf 'mode=C2_SAVE_DEVELOPMENT_RUNTIME_ONLY\n'
    printf 'acceptance=false\n'
    printf 'artifact-reuse=false\n'
    printf 'runner=%s\n' "${runner}"
    sha256sum "${runner}" "${BASH_SOURCE[0]}" "${archive}"
    printf 'eshkol-source-commit=%s\n' \
      "$(git -C "${PROJECT_ROOT}/.deps/eshkol-src" rev-parse HEAD)"
    printf 'native-compiler=%s\n' "${cc}"
    "${cc}" --version
    printf 'aot-compiler=%s\n' "${cxx}"
    "${cxx}" --version
    printf 'compile-environment=env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=%q ESHKOL_LIB_DIR=%q ESHKOL_CXX_COMPILER=%q\n' \
      "${tmp}/cache-clang-a" "${PROJECT_ROOT}/lib" "${cxx}"
    printf 'run-environment=ESHKOL_ARENA_POISON=1\n'
    printf 'native-cflags='; printf ' %q' "${runtime_cflags[@]}"; printf '\n'
    printf 'aot-flags=--strict-types --optimize 0 --no-stdlib'
    printf ' -I %q' "${PROJECT_ROOT}/internal/p1/lib" \
      "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t2/lib" \
      "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/internal/d2/lib" \
      "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/native"
    printf ' -L %q --lib eshkol_transformer_c2_save\n' "${runtime}"
    printf 'aot-source=%s\n' \
      "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_runtime.esk"
    printf 'aot-depfile=%s\naot-output=%s\n' "${depfile}" \
      "${tmp}/clang-a/save"
    for artifact in "${tmp}/clang-a/save" "${tmp}/clang-a/save.ll"; do
      [[ -f "${artifact}" ]] && sha256sum "${artifact}"
    done
    printf 'source-and-native-inputs:\n'
    sha256sum "${runtime_native_inputs[@]}"
    if [[ -s "${depfile}" ]]; then
      sed -e 's/^[^:]*://' -e 's/\\//g' "${depfile}" | \
        tr -s '[:space:]' '\n' | while IFS= read -r dependency; do
          [[ -f "${dependency}" ]] && sha256sum "${dependency}"
        done
    fi
    find "${runtime}" -maxdepth 1 -type f -name '*.o' -print0 | \
      sort -z | xargs -0 -r sha256sum
  } >"${tmp}/manifest.txt"
}

write_lifetime_witness() {
  local ir="${tmp}/clang-a/save.ll"
  local pre_touch_ir="${tmp}/clang-a/pre-touch.ll"
  local borrow_ir="${tmp}/clang-a/borrow-begin.ll"
  local state_barrier_line active_borrow_line active_owner_line
  local active_access_line owner_barrier_line region_line region_block return_line
  test -s "${ir}"
  sed -n '/^define .*@c2-save-pre-touch-p1-identities(/,/^}/p' \
    "${ir}" >"${pre_touch_ir}"
  test -s "${pre_touch_ir}"
  if rg -F 'call ptr @region_create' "${pre_touch_ir}" >/dev/null || \
     rg -F 'call void @eshkol_region_unwind_to' "${pre_touch_ir}" >/dev/null; then
    die "compiled P1 identity preparation entered a temporary region"
  fi
  sed -n '/^define .*@c2-training-state-borrow-begin-internal(/,/^}/p' \
    "${ir}" >"${borrow_ir}"
  test -s "${borrow_ir}"
  state_barrier_line="$(rg -n -m1 -F \
    'call void @eshkol_region_write_barrier_into' "${borrow_ir}")"
  state_barrier_line="${state_barrier_line%%:*}"
  active_borrow_line="$(rg -n -m1 -F \
    '%active-borrow = alloca' "${borrow_ir}")"
  active_borrow_line="${active_borrow_line%%:*}"
  active_owner_line="$(rg -n -m1 -F \
    '%active-owner-token = alloca' "${borrow_ir}")"
  active_owner_line="${active_owner_line%%:*}"
  active_access_line="$(rg -n -m1 -F \
    '%active-access-token = alloca' "${borrow_ir}")"
  active_access_line="${active_access_line%%:*}"
  owner_barrier_line="$(rg -n -F \
    'call void @eshkol_region_write_barrier_into' "${borrow_ir}" | \
    sed -n '2{s/:.*//;p;}')"
  region_line="$(rg -n -m1 -F \
    '%region_mark = call i64 @eshkol_region_mark()' "${borrow_ir}")"
  region_line="${region_line%%:*}"
  region_block="$(sed -n "1,${region_line}p" "${borrow_ir}" | awk '
    /^[[:alnum:]_.-]+:/ { block = $1; sub(/:$/, "", block) }
    END { print block }
  ')"
  return_line="$(rg -n -m1 -F \
    'ret %eshkol_tagged_value %active-borrow.load' "${borrow_ir}")"
  return_line="${return_line%%:*}"
  (( state_barrier_line < active_borrow_line &&
     active_borrow_line < active_owner_line &&
     active_owner_line < active_access_line &&
     active_access_line < owner_barrier_line &&
     active_access_line < region_line && region_line < return_line )) || \
    die "compiled borrow lifetime ordering is not canonical"
  sed -n "${owner_barrier_line},$((owner_barrier_line + 6))p" \
    "${borrow_ir}" | rg -F "br label %${region_block}" >/dev/null || \
    die "compiled owner-link barrier does not precede activation scratch"
  rg -F 'store %eshkol_tagged_value { i8 3, i8 0, i16 0, i32 0, i64 1 }, ptr %region_result_slot' \
    "${borrow_ir}" >/dev/null
  rg -F 'call void @eshkol_region_unwind_to' "${borrow_ir}" >/dev/null
  if sed -n "$((state_barrier_line + 1)),\$p" "${borrow_ir}" | \
       rg -F -e '%borrow.load' -e '%owner-token.load' \
         -e '%access-token.load' -e '%tensor-access.load' >/dev/null; then
    die "compiled borrow path reused a pre-barrier regional identity"
  fi
  {
    printf 'pre-touch-temporary-region=false\n'
    printf 'source-helper=state-dict-c2-owned-entry-shells-internal/3\n'
    printf 'borrow-state-slot-write-barrier-line=%s\n' "${state_barrier_line}"
    printf 'borrow-canonical-readback-line=%s\n' "${active_borrow_line}"
    printf 'borrow-owner-link-write-barrier-line=%s\n' "${owner_barrier_line}"
    printf 'borrow-activation-region-line=%s\n' "${region_line}"
    printf 'borrow-canonical-return-line=%s\n' "${return_line}"
    printf 'borrow-post-barrier-original-use=false\n'
    printf 'borrow-region-escape=immediate-true\n'
    sha256sum "${ir}" "${pre_touch_ir}" "${borrow_ir}" \
      "${tmp}/pre-touch-source.esk"
  } >"${tmp}/lifetime-witness.txt"
}

compile_runtime() {
  local label=$1 compiler=$2
  local compile_status=0
  local depfile_args=() ir_args=()
  mkdir -p "${tmp}/${label}/output"
  printf old >"${tmp}/${label}/output/existing.c2"
  printf old >"${tmp}/${label}/output/precommit.c2"
  printf old >"${tmp}/${label}/output/postcommit.c2"
  if [[ "${development}" == 1 ]]; then
    depfile_args=(--emit-depfile "${tmp}/${label}/save.d")
    ir_args=(--dump-ir)
  fi
  phase_begin aot_compile
  (cd "${tmp}/${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 720s "${runner}" \
        --strict-types --optimize 0 --no-stdlib \
        "${depfile_args[@]}" "${ir_args[@]}" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" -L "${runtime}" \
        --lib eshkol_transformer_c2_save \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_runtime.esk" \
        -o "${tmp}/${label}/save" >"${tmp}/${label}/compile.stdout" \
        2>"${tmp}/${label}/compile.stderr") || compile_status=$?
  if [[ "${development}" == 1 ]]; then
    write_development_manifest "${tmp}/${label}/save.d" \
      "${runtime}/libeshkol_transformer_c2_save.a"
    if (( compile_status == 0 )); then write_lifetime_witness; fi
  fi
  if (( compile_status != 0 )); then
    phase_end "${compile_status}"
    return "${compile_status}"
  fi
  test ! -s "${tmp}/${label}/compile.stderr"
  phase_end
  phase_begin poison_run
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    360s "${tmp}/${label}/save" \
      "${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
      "${tmp}/resources-a" "${tmp}/${label}/output" \
      >"${tmp}/${label}/run.stdout" 2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^C2 PRIVATE CHECKPOINT SAVE PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
  if [[ "${development}" == 1 ]]; then
    printf 'poison-runtime-owner-region-check=true\n' \
      >>"${tmp}/lifetime-witness.txt"
  fi
  phase_end
}

compile_runtime clang-a "${cxx}"
if [[ "${development}" == 1 ]]; then
  printf '%s artifact-dir=%s\n' \
    'C2 SAVE DEVELOPER NON-ACCEPTANCE PASS:' "${tmp}"
  exit 0
fi
compile_runtime clang-b "${cxx}"
cmp "${tmp}/clang-a/save" "${tmp}/clang-b/save"
cmp "${tmp}/clang-a/run.stdout" "${tmp}/clang-b/run.stdout"
for image in a.c2 b.c2 postcommit.c2; do
  cmp "${tmp}/clang-a/output/${image}" "${tmp}/clang-b/output/${image}"
done
cmp <(printf old) "${tmp}/clang-a/output/existing.c2"
cmp <(printf old) "${tmp}/clang-a/output/precommit.c2"

if [[ -x /usr/bin/g++ ]]; then
  compile_runtime gcc /usr/bin/g++
  cmp "${tmp}/clang-a/run.stdout" "${tmp}/gcc/run.stdout"
  for image in a.c2 b.c2 postcommit.c2; do
    cmp "${tmp}/clang-a/output/${image}" "${tmp}/gcc/output/${image}"
  done
fi

python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_save_output.py" \
  "${tmp}/clang-a/output/a.c2" "${tmp}/clang-a/output/b.c2" \
  "${tmp}/clang-a/output/postcommit.c2" "${tmp}/clang-b/output/a.c2"
"${cc}" "${cflags[@]}" "${PROJECT_ROOT}/native/c2_checkpoint_format.c" \
  "${PROJECT_ROOT}/native/c2_x1_canonical.c" \
  "${PROJECT_ROOT}/tests/c2/c2_format_driver.c" -o "${tmp}/format-driver"
"${tmp}/format-driver" "${tmp}/clang-a/output/a.c2" 2 0 \
  >"${tmp}/format-driver.stdout"
grep -E '^0 0 0 1 3 [0-9]+ [0-9]+$' "${tmp}/format-driver.stdout" >/dev/null

"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer -DET_C2_CHECKPOINT_SAVE_TESTING \
  "${bridge_sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/test_checkpoint_save_bridge.c" \
  -o "${tmp}/bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${tmp}/bridge-san" "${tmp}/fixtures-a/valid.c2" \
    >"${tmp}/bridge-san.stdout"
cmp "${tmp}/bridge-clang.stdout" "${tmp}/bridge-san.stdout"

printf '%s\n' \
  'C2 PRIVATE CHECKPOINT SAVE PASS: strict Clang/GCC/C++, deterministic AOT/artifacts, full parser, failpoints, sanitizers, exact source/symbol closure'
