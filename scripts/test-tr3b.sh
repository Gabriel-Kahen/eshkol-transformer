#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp env timeout; do require_command "${command}"; done
oracle="${M3_ORACLE_PYTHON:-${N2_ORACLE_PYTHON:-}}"
[[ "${oracle}" == /* && -x "${oracle}" ]] || \
  die "M3_ORACLE_PYTHON must name the pinned Python 3.14.6/PyTorch 2.13.0+cpu oracle"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-tr3b.XXXXXX")"
cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then
    rm -rf -- "${tmp}"
  else
    printf 'TR3-B failure evidence retained at %s\n' "${tmp}" >&2
  fi
}
trap cleanup EXIT

mkdir -p "${tmp}/corpus"
cd "${PROJECT_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${tmp}/corpus"

common=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -frounding-math -fPIC -fno-common
  -DET_M3_TESTING -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
  -DET_D2_NATIVE_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src" -I "$(eshkol_source_dir)/inc"
)
sources=(
  native/data_io.c native/checkpoint_io.c native/kernel_abi.c
  src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
  src/eshkol_transformer/m3t_f32_integration.c
  src/eshkol_transformer/m3_model.c native/d2_native.c
  native/n2_primitives_provider.c native/n3k_primitives_provider.c
  native/a2_attention_provider.c native/indexed_cross_entropy.c
  native/l3s_masked_objective_provider.c native/tr3b_objective_bridge.c
)

compile_witness() {
  local mode=$1 directory="${tmp}/$1"
  local -a extra=(-O2) runtime=(env)
  mkdir -p "${directory}/objects" "${directory}/cache"
  if [[ "${mode}" == sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
                 UBSAN_OPTIONS=halt_on_error=1)
  fi

  local source object
  for source in "${sources[@]}"; do
    object="${directory}/objects/$(basename "${source%.c}").o"
    "${cc}" "${common[@]}" "${extra[@]}" -c \
      "${PROJECT_ROOT}/${source}" -o "${object}"
  done
  "${cc}" "${common[@]}" "${extra[@]}" -DET_P1_TRUSTED_BUILD -c \
    "${PROJECT_ROOT}/native/p1_identity.c" \
    -o "${directory}/objects/p1_identity.o"
  "${cc}" "${common[@]}" "${extra[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_M3T_PACKAGE_BUILD -c \
    "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "${directory}/objects/i2_helpers.o"
  ar rcsD "${directory}/libtr3b_native.a" "${directory}"/objects/*.o

  "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
    "${extra[@]}" "${PROJECT_ROOT}/tests/tr3b/test_bridge_native.c" \
    "${directory}/libtr3b_native.a" -lm -o "${directory}/native-validation"
  "${runtime[@]}" "${directory}/native-validation" \
    >"${directory}/native-validation.stdout"
  grep -Fx 'TR3B-NATIVE-VALIDATION-PASS' \
    "${directory}/native-validation.stdout" >/dev/null

  local compiler="${cxx}"
  if [[ "${mode}" == sanitize ]]; then
    cat >"${directory}/sanitize-cxx" <<EOF
#!/usr/bin/env bash
exec "${cxx}" -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
EOF
    chmod 0500 "${directory}/sanitize-cxx"
    compiler="${directory}/sanitize-cxx"
  fi
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${directory}/cache" ESHKOL_CXX_COMPILER="${compiler}" \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/t2/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/internal/d2/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -L "${directory}" --lib tr3b_native \
      "${PROJECT_ROOT}/tests/tr3b/witness.esk" -o "${directory}/witness" \
      >"${directory}/compile.stdout" 2>"${directory}/compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s "${directory}/witness" \
    "${tmp}/corpus" \
    >"${directory}/witness.stdout" 2>"${directory}/witness.stderr"
  test ! -s "${directory}/witness.stderr"
  grep -Fx 'TR3B-WITNESS-PASS' "${directory}/witness.stdout" >/dev/null
}

compile_witness normal
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  "${oracle}" -m tests.tr3b.check_witness "${tmp}/normal/witness.stdout"
# A quiet-NaN numerator used to pass the tolerance comparison because every
# ordered comparison with NaN is false. Prove that the checker now rejects it.
python3 - "${tmp}/normal/witness.stdout" "${tmp}/normal/nonfinite.stdout" <<'PY'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
lines = source.read_text().splitlines()
for index, line in enumerate(lines):
    fields = line.split()
    if fields[:2] == ["TR3B-OBSERVATION", "0"]:
        fields[2:6] = ["0", "0", "192", "127"]  # quiet NaN, little-endian
        lines[index] = " ".join(fields)
        break
else:
    raise SystemExit("missing case-0 observation")
destination.write_text("\n".join(lines) + "\n")
PY
if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
     "${oracle}" -m tests.tr3b.check_witness \
       "${tmp}/normal/nonfinite.stdout" \
       >"${tmp}/normal/nonfinite-check.stdout" \
       2>"${tmp}/normal/nonfinite-check.stderr"; then
  die "TR3-B checker accepted a nonfinite objective observation"
fi
grep -F 'nonfinite observation' \
  "${tmp}/normal/nonfinite-check.stderr" >/dev/null
compile_witness sanitize
cmp "${tmp}/normal/witness.stdout" "${tmp}/sanitize/witness.stdout"
printf 'TR3-B compiled genuine D2/M3 objective bridge PASS\n'
