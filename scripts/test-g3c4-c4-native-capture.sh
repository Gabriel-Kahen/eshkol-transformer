#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Future compiled gate only: source review, supported toolchain and fresh ROOT
# admission must precede invocation. This script never repins production.
verify_toolchain
for command in ar cmp git nm python3 readelf realpath sha256sum timeout; do
  require_command "$command"
done
[[ "${Q0_PYTHON:-}" == /* && -x "${Q0_PYTHON:-}" ]] || \
  die 'absolute Q0_PYTHON is required (Python 3.14.6 / Torch 2.13.0+cpu)'
[[ "${1:-}" == /* && "${2:-}" == /* ]] || \
  die 'usage: test-g3c4-c4-native-capture.sh /absolute/fresh-evidence /absolute/frozen-fixture'
evidence="$1"
fixture="$2"
[[ -f "$fixture" ]] || die "missing frozen fixture: $fixture"
test ! -e "$evidence" || die "capture evidence already exists: $evidence"
mkdir -m 0700 "$evidence"
q0_python="$Q0_PYTHON"
q0_target="$(realpath -e "$q0_python")"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"

q0_identity() {
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
    timeout --signal=TERM --kill-after=5s 30s \
    "$q0_python" - <<'PY'
import hashlib
from pathlib import Path
import sys
import torch
if sys.version_info[:3] != (3, 14, 6) or torch.__version__ != "2.13.0+cpu":
    raise SystemExit("wrong Q0 Python/Torch runtime")
print(sys.executable)
print(torch.__version__)
print(Path(torch.__file__).resolve())
print(hashlib.sha256(Path(torch.__file__).read_bytes()).hexdigest())
PY
}
q0_identity >"$evidence/q0.pre"
sha256sum "$q0_python" "$q0_target" "$cc" "$cxx" "$(command -v ar)" "$(command -v nm)" \
  "$(command -v readelf)" "$PROJECT_ROOT/tests/q0/requirements-oracle.lock" \
  "$fixture" >"$evidence/inputs.pre.sha256"
printf '%s\n' '2f3adec7636cebc39b735d7b6c461e2856e390ec1e9f12fccdc0c3074184f041' |
  cmp - <(sha256sum "$fixture" | cut -d' ' -f1)
test -z "$(git --no-optional-locks -C "$PROJECT_ROOT" status --porcelain=v1)" || \
  die 'capture source checkout is not clean'
git --no-optional-locks -C "$PROJECT_ROOT" rev-parse HEAD HEAD^{tree} \
  >"$evidence/source-git.pre"
while IFS= read -r member; do
  sha256sum "$PROJECT_ROOT/$member"
done <"$PROJECT_ROOT/native/g3c4_c4_native_capture_source_closure.txt" \
  >"$evidence/source.pre.sha256"
timeout --signal=TERM --kill-after=5s 30s \
  python3 "$PROJECT_ROOT/tests/g3c4/check_c4_native_capture.py" \
  >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 90s \
  python3 -m unittest -v \
  tests.g3c4.test_c4_native_transcript \
  tests.g3c4.test_c4_native_capture_source \
  >"$evidence/source-tests.stdout" 2>"$evidence/source-tests.stderr"
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 30s \
  python3 -m unittest -v \
  tests.q0.test_python_isolation \
  >"$evidence/q0-isolation.stdout" 2>"$evidence/q0-isolation.stderr"

PYTHONDONTWRITEBYTECODE=1 ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  timeout --signal=TERM --kill-after=5s 30s \
  "$q0_python" -m tests.g3c4.c4_native_transcript \
  --fixture "$fixture" --input --output "$evidence/request.txt" \
  >"$evidence/request.stdout" 2>"$evidence/request.stderr"

build_mode() {
  local mode="$1" directory="$evidence/$1"
  local -a flags=(-O2) environment=(env)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                     UBSAN_OPTIONS=halt_on_error=1)
  fi
  mkdir -p "$directory/objects"
  local -a sources=(
    "$PROJECT_ROOT/tests/g3c4/c4_native_capture.c" \
    "$PROJECT_ROOT/src/eshkol_transformer/m3_i64_integration.c" \
    "$PROJECT_ROOT/native/t1_i64_shell.c" \
    "$PROJECT_ROOT/native/n3k_primitives_provider.c" \
    "$PROJECT_ROOT/native/g3c4_primitives_provider.c" \
    "$PROJECT_ROOT/native/g3s_sampling_provider.c" \
    "$PROJECT_ROOT/native/g3n_primitives_provider.c" \
    "$PROJECT_ROOT/native/n2_primitives_provider.c" \
    "$PROJECT_ROOT/native/a2_kv_cache.c")
  : >"$directory/compile.stdout"
  : >"$directory/compile.stderr"
  for source in "${sources[@]}"; do
    timeout --signal=TERM --kill-after=5s 600s \
      "$cc" -std=c11 -Wall -Wextra -Werror -Wpedantic \
      -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
      -fstack-protector-all "${flags[@]}" \
      -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" \
      -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/src/eshkol_transformer" \
      -DET_I64_TENSOR_TESTING -DET_A2_KV_CACHE_TESTING \
      -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
      -c "$source" -o "$directory/objects/$(basename "$source" .c).o" \
      >>"$directory/compile.stdout" 2>>"$directory/compile.stderr"
  done
  timeout --signal=TERM --kill-after=5s 60s \
    ar rcsD "$directory/libc4_native_capture.a" "$directory"/objects/*.o
  timeout --signal=TERM --kill-after=5s 600s \
    "$cc" "${flags[@]}" "$directory"/objects/*.o \
    -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1 -lm \
    -o "$directory/c4-native-capture" \
    >>"$directory/compile.stdout" 2>>"$directory/compile.stderr"
  sha256sum "$directory/c4-native-capture" >"$directory/executable.pre.sha256"
  timeout --signal=TERM --kill-after=5s 30s \
    nm -g --defined-only --format=posix "$directory/libc4_native_capture.a" \
    >"$directory/archive-defined.symbols"
  timeout --signal=TERM --kill-after=5s 30s \
    nm -u --format=posix "$directory/libc4_native_capture.a" \
    >"$directory/archive-undefined.symbols"
  timeout --signal=TERM --kill-after=5s 30s \
    nm -g --defined-only --format=posix "$directory/c4-native-capture" \
    >"$directory/defined.symbols"
  timeout --signal=TERM --kill-after=5s 30s \
    nm -u --format=posix "$directory/c4-native-capture" \
    >"$directory/undefined.symbols"
  timeout --signal=TERM --kill-after=5s 30s \
    readelf --wide --symbols "$directory/c4-native-capture" \
    >"$directory/elf.symbols"
  "${environment[@]}" PYTHONDONTWRITEBYTECODE=1 \
    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
    timeout --signal=TERM --kill-after=5s 150s \
    "$q0_python" -m tests.g3c4.c4_native_transcript \
    --fixture "$fixture" --binary "$directory/c4-native-capture" \
    --request "$evidence/request.txt" --evidence "$directory/first" \
    >"$directory/first.launch.stdout" 2>"$directory/first.launch.stderr"
  PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 30s \
    "$q0_python" -m tests.g3c4.c4_native_transcript \
    --fixture "$fixture" --raw "$directory/first.stdout" \
    --output "$directory/first.json" \
    >"$directory/parse.stdout" 2>"$directory/parse.stderr"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
    timeout --signal=TERM --kill-after=5s 120s \
    "$q0_python" -m tests.g3c4.c4_math_oracle \
    --fixture "$fixture" \
    --frozen-sha256 2f3adec7636cebc39b735d7b6c461e2856e390ec1e9f12fccdc0c3074184f041 \
    --observation "$directory/first.json" \
    >"$directory/math.stdout" 2>"$directory/math.stderr"
  sha256sum "$directory/c4-native-capture" >"$directory/executable.post.sha256"
  cmp "$directory/executable.pre.sha256" "$directory/executable.post.sha256"
}

build_mode normal
PYTHONDONTWRITEBYTECODE=1 ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  timeout --signal=TERM --kill-after=5s 150s \
  "$q0_python" -m tests.g3c4.c4_native_transcript \
  --fixture "$fixture" --binary "$evidence/normal/c4-native-capture" \
  --request "$evidence/request.txt" --evidence "$evidence/normal/repeat" \
  >"$evidence/normal/repeat.launch.stdout" \
  2>"$evidence/normal/repeat.launch.stderr"
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 30s \
  "$q0_python" -m tests.g3c4.c4_native_transcript \
  --fixture "$fixture" --raw "$evidence/normal/repeat.stdout" \
  --output "$evidence/normal/repeat.json" \
  >"$evidence/normal/repeat.parse.stdout" \
  2>"$evidence/normal/repeat.parse.stderr"
cmp "$evidence/normal/first.stdout" "$evidence/normal/repeat.stdout"
cmp "$evidence/normal/first.json" "$evidence/normal/repeat.json"
build_mode sanitize
cmp "$evidence/normal/first.stdout" "$evidence/sanitize/first.stdout"
cmp "$evidence/normal/first.json" "$evidence/sanitize/first.json"

q0_identity >"$evidence/q0.post"
cmp "$evidence/q0.pre" "$evidence/q0.post"
sha256sum -c "$evidence/inputs.pre.sha256" >"$evidence/inputs.verify.stdout"
while IFS= read -r member; do
  sha256sum "$PROJECT_ROOT/$member"
done <"$PROJECT_ROOT/native/g3c4_c4_native_capture_source_closure.txt" \
  >"$evidence/source.post.sha256"
cmp "$evidence/source.pre.sha256" "$evidence/source.post.sha256"
test -z "$(git --no-optional-locks -C "$PROJECT_ROOT" status --porcelain=v1)" || \
  die 'capture source checkout drift'
git --no-optional-locks -C "$PROJECT_ROOT" rev-parse HEAD HEAD^{tree} \
  >"$evidence/source-git.post"
cmp "$evidence/source-git.pre" "$evidence/source-git.post"
sha256sum "$evidence/normal/c4-native-capture" \
  "$evidence/sanitize/c4-native-capture" \
  "$evidence"/{normal,sanitize}/libc4_native_capture.a \
  "$evidence"/{normal,sanitize}/objects/*.o \
  >"$evidence/compiled-artifacts.sha256"
sha256sum -c "$evidence/compiled-artifacts.sha256" \
  >"$evidence/compiled-artifacts.verify.stdout"
printf 'C4 native capture evidence: %s\n' "$evidence"
