#!/usr/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp diff find nm python3 rg sed sha256sum sort tee timeout \
               tr xargs; do
  require_command "${command}"
done

cc="${CC:-/usr/bin/clang}"
cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
development="${C2_OPERATIONAL_DEVELOPMENT_JOINT_ONLY:-0}"
if [[ "${development}" != 0 && "${development}" != 1 ]]; then
  die "C2_OPERATIONAL_DEVELOPMENT_JOINT_ONLY must be 0 or 1"
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
    'C2 OPERATIONAL DEVELOPER MODE: NON-ACCEPTANCE; one Clang AOT and one intact joint lifecycle only'
else
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-operational.XXXXXX")"
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
    printf 'C2 OPERATIONAL DEVELOPER ARTIFACTS: %s\n' "${tmp}" >&2
    return "${status}"
  fi
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,300p' {} \; >&2 || true
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
test_flags=(
  -DET_C2_CHECKPOINT_CORE_TESTING -DET_C2_CHECKPOINT_READER_TESTING
  -DET_CHECKPOINT_IO_TESTING
)

phase_begin fixtures
fixture_labels=(a b)
if [[ "${development}" == 1 ]]; then fixture_labels=(a); fi
for label in "${fixture_labels[@]}"; do
  PYTHONDONTWRITEBYTECODE=1 python3 \
    "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_operational_fixtures.py" \
    "${tmp}/fixtures-${label}"
done
if [[ "${development}" == 0 ]]; then
  diff -ru "${tmp}/fixtures-a" "${tmp}/fixtures-b"
fi
phase_end
fixtures="${tmp}/fixtures-a"
fixture_args=(
  "${fixtures}/joint-exact.c2"
  "${fixtures}/file-one.c2"
  "${fixtures}/metadata-exact.c2"
  "${fixtures}/metadata-one.c2"
  "${fixtures}/tensor-exact.c2"
  "${fixtures}/tensor-one.c2"
  "${fixtures}/count-exact.c2"
  "${fixtures}/count-one.c2"
  "${fixtures}/joint-corrupt.c2"
)

native_sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/native/checkpoint_io.c"
)
build_native() {
  local compiler=$1 output=$2
  "${compiler}" "${cflags[@]}" "${test_flags[@]}" \
    "${native_sources[@]}" \
    "${PROJECT_ROOT}/tests/c2/test_checkpoint_operational.c" -o "${output}"
}
if [[ "${development}" == 0 ]]; then
build_native "${cc}" "${tmp}/native-clang"
"${tmp}/native-clang" "${fixture_args[@]}" >"${tmp}/native-clang.stdout"
rg -x 'C2 operational native PASS: 340 checks' \
  "${tmp}/native-clang.stdout" >/dev/null
if [[ -x /usr/bin/gcc ]]; then
  build_native /usr/bin/gcc "${tmp}/native-gcc"
  "${tmp}/native-gcc" "${fixture_args[@]}" >"${tmp}/native-gcc.stdout"
  cmp "${tmp}/native-clang.stdout" "${tmp}/native-gcc.stdout"
fi

"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${test_flags[@]}" "${native_sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/test_checkpoint_operational.c" \
  -o "${tmp}/native-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${tmp}/native-san" "${fixture_args[@]}" >"${tmp}/native-san.stdout"
cmp "${tmp}/native-clang.stdout" "${tmp}/native-san.stdout"
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
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c"
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c"
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_operational_metrics.c"
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
for source in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor \
              d2_native o2_optimizer c2_x1_canonical c2_checkpoint_codec \
              c2_checkpoint_format; do
  "${cc}" "${runtime_cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${runtime_cflags[@]}" -DET_CHECKPOINT_IO_TESTING -c \
  "${PROJECT_ROOT}/native/checkpoint_io.c" -o "${runtime}/checkpoint_io.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_SAVE_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" -o "${runtime}/save.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_READER_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c" -o "${runtime}/reader.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_CORE_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c" -o "${runtime}/core.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_LOAD_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c" -o "${runtime}/load.o"
"${cc}" "${runtime_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c" \
  -o "${runtime}/test.o"
"${cc}" "${runtime_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_operational_metrics.c" \
  -o "${runtime}/metrics.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_operational.a" \
  "${runtime}"/*.o
phase_end

# The two size witnesses are test-only: production translations must not
# export them, while the test translations used above must export exactly one.
if [[ "${development}" == 0 ]]; then
"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/f32_tensor.c" \
  -o "${tmp}/f32-production.o"
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -c \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${tmp}/i2-production.o"
if nm -g --defined-only --format=posix "${tmp}/f32-production.o" \
    "${tmp}/i2-production.o" | \
    rg 'et_(f32_tensor_test_control|i2_test_decode_builder_control)_bytes_v1' \
    >/dev/null; then
  die "operational ABI-size witnesses escaped a production translation"
fi
test "$(nm -g --defined-only --format=posix "${runtime}/f32_tensor.o" | \
  awk '$1 == "et_f32_tensor_test_control_bytes_v1" { count++ } END { print count+0 }')" \
  -eq 1
test "$(nm -g --defined-only --format=posix "${runtime}/i2.o" | \
  awk '$1 == "et_i2_test_decode_builder_control_bytes_v1" { count++ } END { print count+0 }')" \
  -eq 1

for compiler in "${cxx}" /usr/bin/g++; do
  if [[ -x "${compiler}" ]]; then
    "${compiler}" -x c++ -std=c++20 -Wall -Wextra -Werror -Wpedantic \
      -Wconversion -Wsign-conversion -Wshadow -Wno-c11-extensions \
      -Wno-missing-field-initializers "${test_flags[@]}" \
      -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" -c \
      "${PROJECT_ROOT}/tests/c2/test_checkpoint_operational.c" \
      -o "${tmp}/driver-$(basename "${compiler}").o"
    "${compiler}" -x c++ -std=c++20 -Wall -Wextra -Werror -Wpedantic \
      -Wconversion -Wsign-conversion -Wshadow -c \
      "${PROJECT_ROOT}/tests/c2/c2_checkpoint_operational_metrics.c" \
      -o "${tmp}/metrics-$(basename "${compiler}").o"
    "${compiler}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
      -Wconversion -Wsign-conversion -Wshadow \
      "${PROJECT_ROOT}/tests/c2/checkpoint_operational_cpp.cpp" \
      -L "${runtime}" -leshkol_transformer_c2_operational -lm \
      -o "${tmp}/witness-$(basename "${compiler}")"
    "${tmp}/witness-$(basename "${compiler}")"
  fi
done
fi

write_development_manifest() {
  local depfile=$1 archive=$2
  {
    printf 'mode=C2_OPERATIONAL_DEVELOPMENT_JOINT_ONLY\n'
    printf 'acceptance=false\n'
    printf 'artifact-reuse=false\n'
    printf 'runner=%s\n' "${runner}"
    sha256sum "${runner}" "${BASH_SOURCE[0]}" "${archive}"
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
    printf ' -L %q --lib eshkol_transformer_c2_operational\n' "${runtime}"
    printf 'aot-source=%s\n' \
      "${PROJECT_ROOT}/tests/c2/c2_checkpoint_operational_runtime.esk"
    printf 'aot-depfile=%s\naot-output=%s\n' "${depfile}" \
      "${tmp}/clang-a/operational"
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

compile_runtime() {
  local label=$1 compiler=$2
  local compile_status=0
  local depfile_args=()
  mkdir -p "${tmp}/${label}/output"
  if [[ "${development}" == 1 ]]; then
    depfile_args=(--emit-depfile "${tmp}/${label}/operational.d")
  fi
  phase_begin aot_compile
  (cd "${tmp}/${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 900s "${runner}" \
        --strict-types --optimize 0 --no-stdlib \
        "${depfile_args[@]}" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" -L "${runtime}" \
        --lib eshkol_transformer_c2_operational \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_operational_runtime.esk" \
        -o "${tmp}/${label}/operational" \
        >"${tmp}/${label}/compile.stdout" \
        2>"${tmp}/${label}/compile.stderr") || compile_status=$?
  if [[ "${development}" == 1 ]]; then
    write_development_manifest "${tmp}/${label}/operational.d" \
      "${runtime}/libeshkol_transformer_c2_operational.a"
  fi
  if (( compile_status != 0 )); then
    phase_end "${compile_status}"
    return "${compile_status}"
  fi
  test ! -s "${tmp}/${label}/compile.stderr"
  phase_end
}

run_runtime() {
  local label=$1 mode=$2
  local output="${tmp}/${label}/output/${mode}"
  mkdir -p "${output}"
  phase_begin "${mode}_run"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    1800s "${tmp}/${label}/operational" \
      "${fixtures}/joint-exact.c2" "${fixtures}/file-one.c2" \
      "${fixtures}/metadata-one.c2" "${fixtures}/tensor-one.c2" \
      "${fixtures}/count-one.c2" "${fixtures}/joint-corrupt.c2" \
      "${output}" "${mode}" >"${tmp}/${label}/${mode}.stdout" \
      2>"${tmp}/${label}/${mode}.stderr"
  test ! -s "${tmp}/${label}/${mode}.stderr"
  rg -x "C2 OPERATIONAL RUNTIME PASS: [0-9]+ checks mode=${mode} baseline-kib=[0-9]+ retained-kib=[0-9]+ peak-kib=[0-9]+ elapsed-ms=[0-9]+ f32-retained-bytes=[0-9]+ i2-retained-bytes=[0-9]+ stage-dead-delta=[0-9]+ c2-dead-delta=[0-9]+ p1-first-touch=[0-9]+ p1-tombstone-delta=[0-9]+ rejected-save-max-arena-delta=[0-9]+ exact-save-max-arena-delta=[0-9]+" \
    "${tmp}/${label}/${mode}.stdout" >/dev/null
  if rg -i 'heap usage at|warning' "${tmp}/${label}/compile.stdout" \
      "${tmp}/${label}/compile.stderr" "${tmp}/${label}/${mode}.stdout" \
      "${tmp}/${label}/${mode}.stderr" >/dev/null; then
    die "${label}/${mode} emitted a heap or compiler warning"
  fi
  awk '
    /^C2 OPERATIONAL RUNTIME PASS:/ {
      for (index = 1; index <= NF; ++index) {
        split($index, pair, "=")
        if (pair[1] == "baseline-kib" || pair[1] == "retained-kib" ||
            pair[1] == "peak-kib") {
          if (pair[2] <= 0 || pair[2] >= 524288) exit 1
        }
        if (pair[1] == "elapsed-ms" && (pair[2] < 0 || pair[2] >= 1800000))
          exit 1
      }
      found = 1
    }
    END { if (!found) exit 1 }
  ' "${tmp}/${label}/${mode}.stdout"
  if [[ "${mode}" == joint ]]; then
    rg -x 'C2 OPERATIONAL MEASURE: mode=joint baseline-kib=[0-9]+ retained-kib=[0-9]+ peak-kib=[0-9]+ elapsed-ms=[0-9]+ exact-save-max-arena-delta=[0-9]+' \
      "${tmp}/${label}/${mode}.stdout" >/dev/null
    rg -x 'C2 OPERATIONAL PHASE: mode=joint first-save-arena-delta=[0-9]+ second-save-arena-delta=[0-9]+ overwrite-save-arena-delta=[0-9]+( (measure|encode|publish)-(current|peak)-kib-[123]=[0-9]+){18}' \
      "${tmp}/${label}/${mode}.stdout" >/dev/null
    cmp "${fixtures}/joint-exact.c2" "${output}/joint-a.c2"
    cmp "${fixtures}/joint-exact.c2" "${output}/joint-b.c2"
  else
    test ! -e "${output}/${mode}-reject.c2"
  fi
  phase_end
  sed -n '/^C2 OPERATIONAL RUNTIME PASS:/p' \
    "${tmp}/${label}/${mode}.stdout"
}

compile_runtime clang-a "${cxx}"
if [[ "${development}" == 1 ]]; then
  run_runtime clang-a joint
  printf '%s artifact-dir=%s\n' \
    'C2 OPERATIONAL DEVELOPER NON-ACCEPTANCE PASS:' "${tmp}"
  exit 0
fi
compile_runtime clang-b "${cxx}"
cmp "${tmp}/clang-a/operational" "${tmp}/clang-b/operational"
read -r -a operational_modes <<< \
  "${C2_OPERATIONAL_MODES:-joint file metadata tensor count corrupt}"
for mode in "${operational_modes[@]}"; do
  run_runtime clang-a "${mode}"
  run_runtime clang-b "${mode}"
done
if [[ " ${operational_modes[*]} " == *' joint '* ]]; then
  cmp "${tmp}/clang-a/output/joint/joint-a.c2" \
    "${tmp}/clang-b/output/joint/joint-a.c2"
fi
if [[ -x /usr/bin/g++ ]]; then
  compile_runtime gcc /usr/bin/g++
fi

for gate in test-c2-format.sh test-c2-core.sh test-c2-checkpoint-save.sh \
            test-c2-checkpoint-load.sh; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/${gate}"
done

printf '%s\n' \
  'C2 CHECKPOINT OPERATIONAL EVIDENCE PASS: exact/one-over tuple, early/late precedence, deterministic byte-identical LOAD/SAVE, two fresh measured runs, strict Clang/GCC/C++, sanitizers, test-only ABI closure, affected regressions'
