#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cmp python3 rg sha256sum timeout; do require_command "$command"; done
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.c2.test_save_image_receipt_source tests.c2.test_save_image_receipt_compiled_source
if [[ "${C2_IMAGE_SOURCE_ONLY:-0}" == 1 ]]; then
  printf 'C2 image receipt compiled gate source preparation PASS (runtime not run)\n'
  exit 0
fi

cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/c2-save-image-receipt-compiled}"
mkdir -p "$evidence"
tmp="$(mktemp -d "$evidence/build.XXXXXX")"
export TMPDIR="$tmp"
trap 'status=$?; if (( status != 0 )); then
  printf "C2 image receipt gate failure retained: %s\n" "$tmp" >&2;
fi' EXIT

flags=(-std=c11 -O2 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off -fexcess-precision=standard
  -frounding-math -fPIC -fvisibility=hidden -fno-common
  -fstack-protector-all -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native"
  -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
native="$tmp/native"
mkdir -p "$native"
"$cc" "${flags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -DET_C2_I2_MODEL_COPY \
  -c "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" -o "$native/i2.o"
"$cc" "${flags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -c \
  "$PROJECT_ROOT/native/o2_wave2_package_bridge.c" -o "$native/o2.o"
"$cc" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1 \
  -c "$PROJECT_ROOT/native/p1_identity.c" -o "$native/p1.o"
for source in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor d2_native \
              o2_optimizer c2_x1_canonical c2_checkpoint_codec \
              c2_checkpoint_format; do
  "$cc" "${flags[@]}" -c "$PROJECT_ROOT/native/$source.c" \
    -o "$native/$source.o"
done
"$cc" "${flags[@]}" -DET_C2_CARRIER_FACTORIES -c \
  "$PROJECT_ROOT/native/k2_capabilities.c" -o "$native/k2.o"
"$cc" "${flags[@]}" -DET_CHECKPOINT_IO_TESTING -c \
  "$PROJECT_ROOT/native/checkpoint_io.c" -o "$native/checkpoint_io.o"
"$cc" "${flags[@]}" -DET_C2_CHECKPOINT_SAVE_TESTING -c \
  "$PROJECT_ROOT/native/c2_checkpoint_save_bridge.c" -o "$native/save.o"
"$cc" "${flags[@]}" -DET_C2_SAVE_IMAGE_RECEIPT_PRIVATE \
  -DET_C2_SAVE_IMAGE_RECEIPT_TESTING -c \
  "$PROJECT_ROOT/native/c2_save_image_receipt.c" -o "$native/image.o"
"$cc" "${flags[@]}" -c \
  "$PROJECT_ROOT/tests/c2/c2_checkpoint_save_test_bridge.c" \
  -o "$native/test.o"
ar rcsD "$native/libeshkol_transformer_c2_image.a" "$native"/*.o

(cd "$PROJECT_ROOT" && PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.d2.prepare_public_resources --output "$tmp/resources")
mkdir -p "$tmp/compile"
(cd "$tmp/compile" &&
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$tmp/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    ESHKOL_CXX_COMPILER="$cxx" \
    timeout --foreground --signal=TERM --kill-after=5s 720s "$runner" \
      --strict-types --optimize 0 --no-stdlib \
      --emit-depfile "$tmp/compile/image.d" \
      -I "$PROJECT_ROOT/internal/p1/lib" \
      -I "$PROJECT_ROOT/internal/c1/lib" \
      -I "$PROJECT_ROOT/internal/t2/lib" \
      -I "$PROJECT_ROOT/internal/t1/lib" \
      -I "$PROJECT_ROOT/internal/d2/lib" \
      -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" \
      -I "$PROJECT_ROOT/native" -L "$native" \
      --lib eshkol_transformer_c2_image \
      "$PROJECT_ROOT/tests/c2/c2_save_image_receipt_runtime.esk" \
      -o "$tmp/compile/image" >"$tmp/compile/stdout" \
      2>"$tmp/compile/stderr")
test ! -s "$tmp/compile/stderr"
python3 - "$PROJECT_ROOT" "$tmp/compile/image.d" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
depfile = Path(sys.argv[2]).read_text().replace('\\\n', ' ')
actual = []
for source in depfile.split(':', 1)[1].split():
    path = Path(source)
    if not path.is_absolute():
        path = Path(sys.argv[2]).parent / path
    path = path.resolve()
    if path.is_relative_to(root):
        name = path.relative_to(root).as_posix()
        if name != 'tests/c2/c2_save_image_receipt_runtime.esk':
            actual.append(name)
expected = (root / 'native/c2_save_image_receipt_source_closure.txt').read_text().splitlines()
assert actual == expected, (actual, expected)
PY

for iteration in a b; do
  mkdir -p "$tmp/$iteration"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    360s "$tmp/compile/image" \
      "$PROJECT_ROOT/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
      "$tmp/resources" "$tmp/$iteration" normal \
      >"$tmp/$iteration/stdout" 2>"$tmp/$iteration/stderr"
  test ! -s "$tmp/$iteration/stderr"
  rg '^C2 PRIVATE SAVE IMAGE RECEIPT PASS: [1-9][0-9]* checks$' \
    "$tmp/$iteration/stdout" >/dev/null
  test ! -e "$tmp/$iteration/pre.c2"
  test -f "$tmp/$iteration/post.c2"
  cmp "$tmp/$iteration/public.c2" "$tmp/$iteration/image.c2"
  python3 - "$tmp/$iteration/stdout" "$tmp/$iteration/image.c2" <<'PY'
import hashlib, pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
values = [line for line in lines if line.startswith('C2 IMAGE DIGEST:')]
assert len(values) == 1
digest = bytes(int(part) for part in values[0].split(':', 1)[1].split())
assert len(digest) == 32
assert digest == hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).digest()
PY
done
cmp "$tmp/a/stdout" "$tmp/b/stdout"
cmp "$tmp/a/image.c2" "$tmp/b/image.c2"

mkdir -p "$tmp/fail-stop"
if ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    360s "$tmp/compile/image" \
      "$PROJECT_ROOT/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
      "$tmp/resources" "$tmp/fail-stop" fail-stop \
      >"$tmp/fail-stop/stdout" 2>"$tmp/fail-stop/stderr"; then
  die 'uncertain receipt release unexpectedly returned'
else
  status=$?
  test "$status" -eq 134 || die "unexpected fail-stop status: $status"
fi
test ! -s "$tmp/fail-stop/stderr"
printf 'C2 image receipt compiled normal/repeat/fail-stop gate PASS\n' \
  | tee "$evidence/result.txt"
