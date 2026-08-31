#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar awk cmp env nm python3 readelf rg sha256sum strings timeout; do
  require_command "${command}"
done
verify_toolchain

c1_runner="$(eshkol_build_dir)/eshkol-run"
c1_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
c1_cc="$(tsv_value "${c1_provenance}" cc_path)"
c1_cxx="$(tsv_value "${c1_provenance}" cxx_path)"
c1_timeout="${C1_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${c1_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "C1_COMPILER_TIMEOUT_SECONDS must be a positive integer"

c1_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-c1.XXXXXX")"
c1_cleanup() {
  if [[ "${C1_KEEP_TMP:-0}" == 1 ]]; then
    printf 'C1 preserved temporary evidence: %s\n' "${c1_tmp}" >&2
  else
    rm -rf -- "${c1_tmp}"
  fi
}
trap c1_cleanup EXIT

c1_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -fno-common -I "${PROJECT_ROOT}/native"
)

"${c1_cc}" "${c1_cflags[@]}" -DET_CHECKPOINT_IO_TESTING=1 \
  "${PROJECT_ROOT}/native/checkpoint_io.c" \
  "${PROJECT_ROOT}/tests/c1/test_checkpoint_io.c" \
  -o "${c1_tmp}/native-a"
"${c1_tmp}/native-a" >"${c1_tmp}/native-a.stdout"
"${c1_tmp}/native-a" >"${c1_tmp}/native-b.stdout"
cmp "${PROJECT_ROOT}/tests/expected/c1-native.stdout" \
    "${c1_tmp}/native-a.stdout"
cmp "${c1_tmp}/native-a.stdout" "${c1_tmp}/native-b.stdout"

"${c1_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/tests/c1/checkpoint_io_header_cpp.cpp" \
  -o "${c1_tmp}/header-cpp.o"

"${c1_cc}" "${c1_cflags[@]}" -fPIC -fvisibility=hidden \
  -fstack-protector-all -c "${PROJECT_ROOT}/native/checkpoint_io.c" \
  -o "${c1_tmp}/production-a.o"
"${c1_cc}" "${c1_cflags[@]}" -fPIC -fvisibility=hidden \
  -fstack-protector-all -c "${PROJECT_ROOT}/native/checkpoint_io.c" \
  -o "${c1_tmp}/production-b.o"
cmp "${c1_tmp}/production-a.o" "${c1_tmp}/production-b.o"
LC_ALL=C nm -g --defined-only "${c1_tmp}/production-a.o" | \
  awk '$2 ~ /^[A-Z]$/ { print $3 }' | LC_ALL=C sort \
  >"${c1_tmp}/production-symbols.actual"
cmp "${PROJECT_ROOT}/native/checkpoint_io_symbols.txt" \
    "${c1_tmp}/production-symbols.actual"
if nm -a "${c1_tmp}/production-a.o" | rg 'checkpoint_io_test_' >/dev/null; then
  die "C1 production native boundary contains test hooks"
fi

"${c1_cc}" "${c1_cflags[@]}" -DET_CHECKPOINT_IO_TESTING=1 \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/native/checkpoint_io.c" \
  "${PROJECT_ROOT}/tests/c1/test_checkpoint_io.c" \
  -o "${c1_tmp}/native-sanitized"
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  "${c1_tmp}/native-sanitized" >"${c1_tmp}/native-sanitized.stdout"
cmp "${PROJECT_ROOT}/tests/expected/c1-native.stdout" \
    "${c1_tmp}/native-sanitized.stdout"

"${PROJECT_ROOT}/scripts/build-p1-identity.sh" \
  "${c1_tmp}/p1-identity" normal all
mkdir -p "${c1_tmp}/native-link"
ar rcsD "${c1_tmp}/native-link/libeshkol_transformer_checkpoint_io.a" \
  "${c1_tmp}/production-a.o"

compile_c1_aot() {
  local cache="$1" output="$2" log="$3"
  mkdir -p "${cache}"
  if ! timeout --foreground --signal=TERM --kill-after=5s \
    "${c1_timeout}s" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${cache}" \
    "${c1_runner}" --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    -I "${PROJECT_ROOT}/tests/c1/providers" \
    -L "${c1_tmp}/p1-identity/trusted" \
    --lib eshkol_transformer_p1_identity \
    -L "${c1_tmp}/native-link" --lib eshkol_transformer_checkpoint_io \
    "${PROJECT_ROOT}/tests/c1/checkpoint_test.esk" -o "${output}" \
    >"${log}" 2>&1; then
    sed -n '1,260p' "${log}" >&2
    die "C1 AOT compilation failed"
  fi
  [[ -x "${output}" ]] || die "C1 AOT executable is missing"
  ! rg -F 'ERROR:' "${log}" >/dev/null || \
    die "C1 compiler reported ERROR while returning success"
}

compile_c1_production_object() {
  local cache="$1" output="$2" depfile="$3" log="$4"
  mkdir -p "${cache}"
  if ! timeout --foreground --signal=TERM --kill-after=5s \
    "${c1_timeout}s" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${cache}" \
    "${c1_runner}" --strict-types --optimize 0 --emit-object --no-stdlib \
    --emit-depfile "${depfile}" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    "${PROJECT_ROOT}/tests/c1/compile_checkpoint_internal.esk" \
    -o "${output}" >"${log}" 2>&1; then
    sed -n '1,260p' "${log}" >&2
    die "C1 production Eshkol compilation failed"
  fi
  [[ -s "${output}" && -s "${depfile}" ]] || \
    die "C1 production object or depfile is missing"
  ! rg -F 'ERROR:' "${log}" >/dev/null || \
    die "C1 compiler reported ERROR while returning success"
}

mkdir -p "${c1_tmp}/production-a" "${c1_tmp}/production-b"
compile_c1_production_object "${c1_tmp}/cache-production-a" \
  "${c1_tmp}/production-a/c1.o" "${c1_tmp}/production-a/c1.d" \
  "${c1_tmp}/production-c1-a.log"
compile_c1_production_object "${c1_tmp}/cache-production-b" \
  "${c1_tmp}/production-b/c1.o" "${c1_tmp}/production-b/c1.d" \
  "${c1_tmp}/production-c1-b.log"
cmp "${c1_tmp}/production-a/c1.o" "${c1_tmp}/production-b/c1.o" || \
  die "C1 production Eshkol object is not deterministic"
for required_source in \
  "${PROJECT_ROOT}/internal/c1/lib/transformer/checkpoint_internal.esk" \
  "${PROJECT_ROOT}/internal/c1/lib/c1_sha256.esk" \
  "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk" \
  "${PROJECT_ROOT}/native/e1b_error_consumer_private.esk"; do
  grep -F "${required_source}" "${c1_tmp}/production-a/c1.d" >/dev/null || \
    die "C1 production closure omitted ${required_source}"
done
if rg -F "${PROJECT_ROOT}/tests/" "${c1_tmp}/production-a/c1.d" | \
    rg -v -F "${PROJECT_ROOT}/tests/c1/compile_checkpoint_internal.esk" \
    >/dev/null; then
  die "C1 production closure contains a fixture or test dependency"
fi

for run in a b; do
  mkdir -p "${c1_tmp}/cache-negative-${run}"
  if timeout --foreground --signal=TERM --kill-after=5s \
      "${c1_timeout}s" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${c1_tmp}/cache-negative-${run}" \
      "${c1_runner}" --strict-types --optimize 0 --emit-object --no-stdlib \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/c1/negative_checkpoint_internal_public.esk" \
      -o "${c1_tmp}/negative-public.o" \
      >"${c1_tmp}/negative-public-${run}.log" 2>&1; then
    die "C1 trusted implementation unexpectedly compiled from the public root"
  fi
  [[ ! -e "${c1_tmp}/negative-public.o" ]] || \
    die "C1 public-boundary negative left an object"
done
cmp "${c1_tmp}/negative-public-a.log" "${c1_tmp}/negative-public-b.log" || \
  die "C1 public-boundary negative is not deterministic"

mkdir -p "${c1_tmp}/run-a" "${c1_tmp}/run-b"
compile_c1_aot "${c1_tmp}/cache-a" "${c1_tmp}/run-a/c1-test" \
  "${c1_tmp}/compile-a.log"
compile_c1_aot "${c1_tmp}/cache-b" "${c1_tmp}/run-b/c1-test" \
  "${c1_tmp}/compile-b.log"
cmp "${c1_tmp}/run-a/c1-test" "${c1_tmp}/run-b/c1-test"
if ! (cd "${c1_tmp}/run-a" && ./c1-test >c1.stdout); then
  sed -n '1,200p' "${c1_tmp}/run-a/c1.stdout" >&2
  die "C1 first AOT execution failed"
fi
if ! (cd "${c1_tmp}/run-b" && ./c1-test >c1.stdout); then
  sed -n '1,200p' "${c1_tmp}/run-b/c1.stdout" >&2
  die "C1 second AOT execution failed"
fi
cmp "${PROJECT_ROOT}/tests/expected/c1.stdout" "${c1_tmp}/run-a/c1.stdout" || \
  die "C1 AOT output differs from expected"
cmp "${c1_tmp}/run-a/c1.stdout" "${c1_tmp}/run-b/c1.stdout"
cmp "${c1_tmp}/run-a/checkpoint.etcp" "${c1_tmp}/run-b/checkpoint.etcp"
sha256sum "${c1_tmp}/run-a/checkpoint.etcp" | awk '{ print $1 }' \
  >"${c1_tmp}/checkpoint.sha256"
cmp "${PROJECT_ROOT}/tests/expected/c1-checkpoint.sha256" \
  "${c1_tmp}/checkpoint.sha256" || \
  die "C1 checkpoint bytes differ from the reviewed digest"

mkdir -p "${c1_tmp}/cache-validator"
if ! timeout --foreground --signal=TERM --kill-after=5s \
    "${c1_timeout}s" env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${c1_tmp}/cache-validator" \
    "${c1_runner}" --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    -I "${PROJECT_ROOT}/tests/c1/providers" \
    -L "${c1_tmp}/p1-identity/trusted" \
    --lib eshkol_transformer_p1_identity \
    -L "${c1_tmp}/native-link" --lib eshkol_transformer_checkpoint_io \
    "${PROJECT_ROOT}/tests/c1/checkpoint_validator.esk" \
    -o "${c1_tmp}/checkpoint-validator" \
    >"${c1_tmp}/checkpoint-validator.compile.log" 2>&1; then
  sed -n '1,260p' "${c1_tmp}/checkpoint-validator.compile.log" >&2
  die "C1 adversarial validator compilation failed"
fi
[[ -x "${c1_tmp}/checkpoint-validator" ]] || \
  die "C1 checkpoint validator is missing"
! rg -F 'ERROR:' "${c1_tmp}/checkpoint-validator.compile.log" >/dev/null || \
  die "C1 checkpoint-validator compiler reported ERROR while returning success"
python3 "${PROJECT_ROOT}/tests/c1/test_checkpoint_format.py" \
  "${c1_tmp}/checkpoint-validator" \
  "${c1_tmp}/run-a/checkpoint.etcp" --fuzz-cases 192 \
  >"${c1_tmp}/adversarial.stdout"
cmp "${PROJECT_ROOT}/tests/expected/c1-adversarial.stdout" \
  "${c1_tmp}/adversarial.stdout" || \
  die "C1 adversarial validator output differs from expected"

if rg -n -i 'python|pytorch|pickle|eval\(|dlopen|system\(' \
    "${PROJECT_ROOT}/internal/c1" \
    "${PROJECT_ROOT}/native/checkpoint_io.c" \
    "${PROJECT_ROOT}/native/checkpoint_io.h" >/dev/null; then
  die "C1 production path contains a forbidden runtime dependency or evaluator"
fi
if strings -a "${c1_tmp}/production-a.o" | \
    rg -i 'python|pytorch|pickle|dlopen|ET_CHECKPOINT_IO_TESTING' >/dev/null; then
  die "C1 production native artifact contains a forbidden dependency or test marker"
fi

printf 'C1 PASS: native I/O, sanitizers, repeated AOT, 992 adversarial checks, logical round trip, and isolation\n'
