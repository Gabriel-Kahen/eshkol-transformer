#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp diff gcc g++ nm python3 rg sed sort timeout tr; do
  require_command "${command}"
done

clang_cc=
clang_cxx=
resolve_provenance_compilers clang_cc clang_cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-inspect.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,240p' {} \; >&2 || true
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 python3 \
  tests/c2/prepare_checkpoint_inspect_fixtures.py "${tmp}/fixtures-a" && \
  PYTHONDONTWRITEBYTECODE=1 python3 \
  tests/c2/prepare_checkpoint_inspect_fixtures.py "${tmp}/fixtures-b")
diff -ru "${tmp}/fixtures-a" "${tmp}/fixtures-b"
ln -s "${tmp}/fixtures-a/valid.c2" "${tmp}/fixtures-a/link.c2"

sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_inspect_bridge.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/native/checkpoint_io.c"
  "${PROJECT_ROOT}/tests/c2/test_checkpoint_inspect_bridge.c"
)
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
        -fno-common -fvisibility=hidden -I "${PROJECT_ROOT}/include"
        -I "${PROJECT_ROOT}/native"
        -DET_C2_CHECKPOINT_CORE_TESTING -DET_C2_CHECKPOINT_READER_TESTING
        -DET_C2_CHECKPOINT_INSPECT_TESTING)
arguments=("${tmp}/fixtures-a/valid.c2" "${tmp}/fixtures-a/compiler.c2"
  "${tmp}/fixtures-a/checksum.c2" "${tmp}/fixtures-a/hostile.c2"
  "${tmp}/fixtures-a/profile.c2" "${tmp}/fixtures-a/missing.c2")

for compiler in "${clang_cc}" /usr/bin/gcc; do
  name="$(basename "${compiler}")"
  "${compiler}" "${cflags[@]}" "${sources[@]}" -o "${tmp}/bridge-${name}"
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 90s \
      "${tmp}/bridge-${name}" "${arguments[@]}" \
      >"${tmp}/bridge-${name}-${run}.stdout"
  done
  cmp "${tmp}/bridge-${name}-1.stdout" "${tmp}/bridge-${name}-2.stdout"
  rg -x 'C2 private checkpoint inspect bridge: PASS \(60 checks\)' \
    "${tmp}/bridge-${name}-1.stdout" >/dev/null
done

for compiler in "${clang_cxx}" /usr/bin/g++; do
  "${compiler}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
    -I "${PROJECT_ROOT}/native" \
    "${PROJECT_ROOT}/tests/c2/checkpoint_inspect_bridge_header_cpp.cpp" \
    -o "${tmp}/header-$(basename "${compiler}")"
  "${tmp}/header-$(basename "${compiler}")"
done

"${clang_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_inspect_bridge.c" \
  -o "${tmp}/inspect-production.o"
nm -g --defined-only "${tmp}/inspect-production.o" | awk '{print $3}' | sort \
  >"${tmp}/symbols.txt"
cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_inspect_bridge_symbols.txt" \
  "${tmp}/symbols.txt"

"${clang_cc}" -std=c11 -I "${PROJECT_ROOT}/native" -MM \
  "${PROJECT_ROOT}/native/c2_checkpoint_inspect_bridge.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c" \
  "${PROJECT_ROOT}/native/c2_x1_canonical.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c" \
  "${PROJECT_ROOT}/native/checkpoint_io.c" | \
  sed -e 's/^[^:]*://' -e 's/\\//g' | tr -s '[:space:]' '\n' | \
  sed "s#^${PROJECT_ROOT}/##" | rg '^native/' | sort -u \
  >"${tmp}/bridge-source-closure.txt"
cmp "${PROJECT_ROOT}/native/c2_checkpoint_inspect_bridge_source_closure.txt" \
  "${tmp}/bridge-source-closure.txt"

compile_probe() {
  local label=$1
  mkdir -p "${tmp}/probe-${label}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${tmp}/cache-probe-${label}" \
    ESHKOL_CXX_COMPILER="${clang_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 240s "${runner}" \
      --strict-types --optimize 0 --no-stdlib --emit-object \
      --emit-depfile "${tmp}/probe-${label}/inspect.d" \
      -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      "${PROJECT_ROOT}/tests/c2/c2_checkpoint_inspect_compile_probe.esk" \
      -o "${tmp}/probe-${label}/inspect.o" \
      >"${tmp}/probe-${label}/compile.stdout" \
      2>"${tmp}/probe-${label}/compile.stderr"
  test ! -s "${tmp}/probe-${label}/compile.stderr"
  nm -u --format=posix "${tmp}/probe-${label}/inspect.o" | awk '{print $1}' | \
    rg '^et_c2_' | sort -u >"${tmp}/probe-${label}/c2-undefined"
  cmp <(printf '%s\n' \
      et_c2_private_checkpoint_inspect_bridge_v1 \
      et_c2_private_metadata_factory_authenticate_v1 \
      et_c2_private_metadata_factory_register_v1 \
      et_c2_private_policy_factory_authenticate_v1 \
      et_c2_private_policy_factory_register_v1) \
    "${tmp}/probe-${label}/c2-undefined"
  sed -e 's/^[^:]*://' -e 's/\\//g' \
    "${tmp}/probe-${label}/inspect.d" | tr -s '[:space:]' '\n' | \
    rg "^${PROJECT_ROOT}/" | sed "s#^${PROJECT_ROOT}/##" | \
    rg -v '^tests/c2/c2_checkpoint_inspect_compile_probe\.esk$' \
    >"${tmp}/probe-${label}/source-closure"
  cmp "${PROJECT_ROOT}/native/c2_checkpoint_inspect_source_closure.txt" \
    "${tmp}/probe-${label}/source-closure"
}
compile_probe a
compile_probe b
cmp "${tmp}/probe-a/inspect.o" "${tmp}/probe-b/inspect.o"
cmp "${tmp}/probe-a/source-closure" "${tmp}/probe-b/source-closure"

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
for source in c2_checkpoint_inspect_bridge c2_checkpoint_core \
              c2_checkpoint_reader c2_checkpoint_format c2_x1_canonical \
              checkpoint_io kernel_abi f32_tensor; do
  "${clang_cc}" "${cflags[@]}" -fPIC -c \
    "${PROJECT_ROOT}/native/${source}.c" -o "${runtime}/${source}.o"
done
"${clang_cc}" "${cflags[@]}" -fPIC -DET_C2_CARRIER_FACTORIES -c \
  "${PROJECT_ROOT}/native/k2_capabilities.c" \
  -o "${runtime}/k2_capabilities.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_inspect.a" "${runtime}"/*.o

compile_aot() {
  local label=$1
  mkdir -p "${tmp}/${label}"
  (cd "${tmp}/${label}" && env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache-${label}" \
    ESHKOL_CXX_COMPILER="${clang_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 240s "${runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" -L "${runtime}" \
      --lib eshkol_transformer_c2_inspect \
      "${PROJECT_ROOT}/tests/c2/c2_checkpoint_inspect_runtime.esk" \
      -o "${tmp}/${label}/inspect" \
      >"${tmp}/${label}/compile.stdout" \
      2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
}
compile_aot a
compile_aot b
cmp "${tmp}/a/inspect" "${tmp}/b/inspect"
for label in a b; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${tmp}/${label}/inspect" "${arguments[@]:0:5}" \
    "${tmp}/fixtures-a/missing.c2" "${tmp}/fixtures-a" \
    "${tmp}/fixtures-a/link.c2" >"${tmp}/${label}/run.stdout" \
    2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  rg -x 'C2 CHECKPOINT INSPECT PASS: 55 checks' \
    "${tmp}/${label}/run.stdout" >/dev/null
done
cmp "${tmp}/a/run.stdout" "${tmp}/b/run.stdout"

if rg -n '(^|[/_-])k2|^\(provide|checkpoint-load|checkpoint-save!|lib/transformer/persistence\.esk' \
    "${PROJECT_ROOT}/native/c2_checkpoint_inspect_extension.esk" \
    "${PROJECT_ROOT}/native/c2_checkpoint_inspect_source_closure.txt"; then
  die "private C2 inspect closure publishes a facade or composes K2/load/save"
fi

"${clang_cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" -o "${tmp}/bridge-sanitize"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${tmp}/bridge-sanitize" "${arguments[@]}" >"${tmp}/sanitize.stdout"
rg -x 'C2 private checkpoint inspect bridge: PASS \(60 checks\)' \
  "${tmp}/sanitize.stdout" >/dev/null

printf 'C2 CHECKPOINT INSPECT PASS: real same-fd inspect, bounded inert metadata, exact identity/errors/limits/profile, no leaks, strict deterministic AOT, C/C++, ASan/UBSan, symbols/source closure, no K2/public facade\n'
