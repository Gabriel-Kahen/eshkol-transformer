#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "G3-R SAVE admission requires a clean source tree"
for command in ar cmp cut nm python3 realpath rg sha256sum stat timeout; do
  require_command "$command"
done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" && ! -L "$out" ]] || die "G3-R SAVE output must be fresh"
mkdir -p "$out"
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
{
  if [[ -d "$PROJECT_ROOT/.git" || -f "$PROJECT_ROOT/.git" ]] && \
      command -v git >/dev/null; then
    printf 'source_head\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    printf 'source_tree\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})"
  else
    printf 'source_tree\tpackaged source; verify package manifest\n'
  fi
  printf 'cc\t%s\n' "$cc"
  printf 'cxx\t%s\n' "$cxx"
  printf 'runner_sha256\t%s\n' "$(sha256sum "$runner" | cut -d ' ' -f 1)"
} >"$out/provenance.tsv"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "$out/compiler-provenance.tsv"

PYTHONDONTWRITEBYTECODE=1 python3 -m tests.c2.prepare_checkpoint_load_fixtures \
  "$out/seed"
/usr/bin/bash "$PROJECT_ROOT/scripts/build-c2.sh" "$out/c2" \
  >"$out/c2-build.stdout" 2>"$out/c2-build.stderr"
archive="$out/c2/libeshkol_transformer_wave2.a"
facade="$out/c2/facades/transformer/trainer.esk"
cmp "$PROJECT_ROOT/tests/c2/legacy_facades/transformer/trainer.esk" "$facade"
cmp "$PROJECT_ROOT/native/c2_wave2_source_closure.txt" \
  "$out/c2/c2_wave2.o.evidence/source-closure.txt"
nm -g --defined-only "$archive" | \
  rg 'et_e1b_public_c2_checkpoint_save_v1' \
  >"$out/public-save-symbol.txt"
sha256sum "$archive" "$out/c2/c2_wave2.o" "$facade" \
  "$PROJECT_ROOT/tests/c2/public_runtime.esk" \
  "$PROJECT_ROOT/tests/g3r/public_without_save.esk" \
  >"$out/producer-sources.sha256"

compile_public() {
  local name="$1" source="$2" directory="$out/$1"
  mkdir -p "$directory/cache"
  printf '%q ' "$runner" --strict-types --no-stdlib --dump-ir \
    -I "$out/c2/facades" -I "$PROJECT_ROOT/lib" \
    -L "$out/c2" --lib eshkol_transformer_wave2 "$source" \
    -o "$directory/program" >"$directory/build.argv"
  printf '\n' >>"$directory/build.argv"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    ESHKOL_CXX_COMPILER="$cxx" \
    timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
      --strict-types --no-stdlib --dump-ir -I "$out/c2/facades" \
      -I "$PROJECT_ROOT/lib" -L "$out/c2" \
      --lib eshkol_transformer_wave2 "$source" -o "$directory/program" \
      >"$directory/build.stdout" 2>"$directory/build.stderr"
  test -s "$directory/program"
  test -s "$directory/program.ll"
}
compile_public no-save "$PROJECT_ROOT/tests/g3r/public_without_save.esk"
compile_public producer "$PROJECT_ROOT/tests/c2/public_runtime.esk"
rg 'et_e1b_public_c2_checkpoint_save_v1' \
  "$out/producer/program.ll" >"$out/producer/save-call-ir.txt"
if rg 'et_e1b_public_c2_checkpoint_save_v1' \
    "$out/no-save/program.ll" >"$out/no-save/unexpected-save-ir.txt"; then
  die "missing-SAVE control unexpectedly references public SAVE"
fi

saved="$out/roundtrip.c2"
test ! -e "$saved" && test ! -L "$saved"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
  "$out/no-save/program" "$out/seed/valid.c2" "$saved" \
  "$out/seed/nonf32-buffer.c2" \
  >"$out/no-save/run.stdout" 2>"$out/no-save/run.stderr"
test ! -s "$out/no-save/run.stderr"
rg -x 'G3-R NO-SAVE CONTROL PASS' "$out/no-save/run.stdout" >/dev/null
test ! -e "$saved" && test ! -L "$saved" || \
  die "missing-SAVE control unexpectedly produced the image"
printf 'absent after compiled LOAD/release without SAVE\n' \
  >"$out/no-save/output-absence.txt"

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
  "$out/producer/program" "$out/seed/valid.c2" "$saved" \
  "$out/seed/nonf32-buffer.c2" \
  >"$out/producer/run.stdout" 2>"$out/producer/run.stderr"
test ! -s "$out/producer/run.stderr"
rg -x 'C2 PUBLIC RUNTIME PASS: 10 checks' \
  "$out/producer/run.stdout" >/dev/null
test -f "$saved" && test ! -L "$saved" && test -s "$saved"
size="$(stat -c %s "$saved")"
(( size <= 1048576 )) || die "saved C2 exceeds private 1 MiB policy"
cmp "$out/seed/valid.c2" "$saved" >"$out/producer/cmp.stdout"
printf 'byte-identical to canonical LOAD seed\n' \
  >"$out/producer/cmp.result"
printf 'saved_c2_bytes\t%s\n' "$size" >>"$out/provenance.tsv"
sha256sum "$out/seed/valid.c2" "$saved" \
  >"$out/c2-image.sha256"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "$PROJECT_ROOT/tests/g3r/prepare_pair_file_inputs.py" \
  --c2-image "$saved" --output "$out/companions"
python3 - "$saved" "$out/companions/valid.g3r" <<'PY'
import hashlib
from pathlib import Path
import sys

saved, companion = (Path(arg) for arg in sys.argv[1:])
assert companion.read_bytes()[224:256] == hashlib.sha256(saved.read_bytes()).digest()
PY
sha256sum "$saved" "$out/companions/valid.g3r" \
  >"$out/pair-before-reader.sha256"

common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off -fexcess-precision=standard
  -frounding-math -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native")
test_flags=(-DET_C2_CHECKPOINT_LOAD_TESTING -DET_C2_CHECKPOINT_CORE_TESTING
  -DET_C2_CHECKPOINT_READER_TESTING -DET_CHECKPOINT_IO_TESTING)

build_reader() {
  local mode="$1" directory="$out/private-$1" name sanitizer_link=
  local -a flags=(-O2) runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    sanitizer_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1
                 LSAN_OPTIONS=exitcode=23:report_objects=1)
  fi
  mkdir -p "$directory/objects" "$directory/cache"
  {
  set -x
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -c \
    "$PROJECT_ROOT/native/i2_wave2_package_bridge.c" \
    -o "$directory/objects/i2.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE -c \
    "$PROJECT_ROOT/native/o2_wave2_package_bridge.c" \
    -o "$directory/objects/o2.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_P1_TRUSTED_BUILD=1 -c "$PROJECT_ROOT/native/p1_identity.c" \
    -o "$directory/objects/p1.o"
  for name in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor \
              d2_native o2_optimizer c2_x1_canonical c2_checkpoint_format \
              checkpoint_io c2_checkpoint_reader c2_checkpoint_core; do
    "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
      -c "$PROJECT_ROOT/native/$name.c" \
      -o "$directory/objects/$name.o"
  done
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_C2_CARRIER_FACTORIES -DET_K2_TESTING -c \
    "$PROJECT_ROOT/native/k2_capabilities.c" \
    -o "$directory/objects/k2.o"
  "$cc" "${common[@]}" "${flags[@]}" "${test_flags[@]}" \
    -DET_G3R_C2_STAGE_SHA_PRIVATE -c \
    "$PROJECT_ROOT/native/c2_checkpoint_load_bridge.c" \
    -o "$directory/objects/load.o"
  ar rcsD "$directory/libg3r_c2_sha_stage.a" "$directory/objects"/*.o
  set +x
  } >"$directory/native-build.stdout" 2>"$directory/native-build.stderr"
  nm -g --defined-only "$directory/objects/load.o" | \
    rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
    >"$directory/feature-on-symbol.txt"
  nm -g --defined-only "$directory/objects/checkpoint_io.o" | \
    rg 'et_checkpoint_io_test_(reset|fail)_v1' \
    >"$directory/io-fault-symbols.txt"
  sha256sum "$directory/libg3r_c2_sha_stage.a" \
    >"$directory/native-archive.sha256"
  printf '#!/usr/bin/env bash\nexec %q %s "$@"\n' \
    "$cxx" "$sanitizer_link" >"$directory/cxx-wrap"
  chmod 0500 "$directory/cxx-wrap"
  local -a inputs=(
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
    -I "$PROJECT_ROOT/internal/t2/lib" -I "$PROJECT_ROOT/internal/t1/lib"
    -I "$PROJECT_ROOT/internal/d2/lib" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" -L "$directory"
    --lib g3r_c2_sha_stage "$PROJECT_ROOT/tests/g3r/pair_file_read_runtime.esk")
  printf '%q ' "$runner" --strict-types --optimize 0 --no-stdlib --dump-ir \
    "${inputs[@]}" -o "$directory/pair-file-reader" \
    >"$directory/build.argv"
  printf '\n' >>"$directory/build.argv"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
      --strict-types --optimize 0 --no-stdlib --dump-ir "${inputs[@]}" \
      -o "$directory/pair-file-reader" \
      >"$directory/build.stdout" 2>"$directory/build.stderr"
  test -s "$directory/pair-file-reader"
  test -s "$directory/pair-file-reader.ll"
  test ! -s "$directory/build.stderr"
  nm -u --format=posix "$directory/pair-file-reader" \
    >"$directory/linked-undefined-symbols.txt"
  # Executable mode in the pinned compiler does not emit dependency files.
  # One matching object build authenticates the loaded Eshkol source closure.
  if [[ "$mode" == normal ]]; then
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$directory/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "$directory/reader.d" "${inputs[@]}" \
        -o "$directory/reader.o" \
        >"$directory/object-build.stdout" 2>"$directory/object-build.stderr"
    test ! -s "$directory/object-build.stderr"
    sed -e 's/^[^:]*://' -e 's/\\//g' "$directory/reader.d" | \
      tr -s '[:space:]' '\n' | rg "^$PROJECT_ROOT/" | \
      sed "s#^$PROJECT_ROOT/##" | \
      rg -v '^tests/g3r/pair_file_read_runtime\.esk$' \
      >"$directory/source-closure.txt"
    cmp "$PROJECT_ROOT/native/g3r_c2_pair_file_read_source_closure.txt" \
      "$directory/source-closure.txt"
    nm -u --format=posix "$directory/reader.o" \
      >"$directory/eshkol-object-undefined-symbols.txt"
    rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
      "$directory/eshkol-object-undefined-symbols.txt" >/dev/null
  fi
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 360s \
    "$directory/pair-file-reader" "$saved" "$out/companions" \
    >"$directory/run.stdout" 2>"$directory/run.stderr"
  test ! -s "$directory/run.stderr"
  rg -x 'G3-R companion file read PASS: [1-9][0-9]* checks' \
    "$directory/run.stdout" >/dev/null
}
build_reader normal
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 360s \
  "$out/private-normal/pair-file-reader" "$saved" "$out/companions" \
  >"$out/private-normal/repeat.stdout" \
  2>"$out/private-normal/repeat.stderr"
test ! -s "$out/private-normal/repeat.stderr"
cmp "$out/private-normal/run.stdout" "$out/private-normal/repeat.stdout"
build_reader sanitize
cmp "$out/private-normal/run.stdout" "$out/private-sanitize/run.stdout"

build_tokenizer_bind() {
  local mode="$1" directory="$out/private-$1" binder="$out/tokenizer-$1"
  local -a runtime=(env)
  if [[ "$mode" == sanitize ]]; then
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1
                 LSAN_OPTIONS=exitcode=23:report_objects=1)
  fi
  mkdir -p "$binder/cache"
  local -a inputs=(
    -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
    -I "$PROJECT_ROOT/internal/t2/lib" -I "$PROJECT_ROOT/internal/t1/lib"
    -I "$PROJECT_ROOT/internal/d2/lib" -I "$PROJECT_ROOT/src"
    -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" -L "$directory"
    --lib g3r_c2_sha_stage
    "$PROJECT_ROOT/tests/g3r/pair_tokenizer_bind_runtime.esk")
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$binder/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
    ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
      --strict-types --optimize 0 --no-stdlib --dump-ir "${inputs[@]}" \
      -o "$binder/caller" >"$binder/build.stdout" 2>"$binder/build.stderr"
  test -s "$binder/caller" && test -s "$binder/caller.ll"
  test ! -s "$binder/build.stderr"
  nm -u --format=posix "$binder/caller" >"$binder/linked-undefined-symbols.txt"
  rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
    "$binder/caller.ll" >"$binder/c2-sha-ir.txt"
  if [[ "$mode" == normal ]]; then
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$binder/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
      ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
      timeout --foreground --signal=TERM --kill-after=5s 1200s "$runner" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "$binder/caller.d" "${inputs[@]}" \
        -o "$binder/caller.o" \
        >"$binder/object.stdout" 2>"$binder/object.stderr"
    test ! -s "$binder/object.stderr"
    sed -e 's/^[^:]*://' -e 's/\\//g' "$binder/caller.d" | \
      tr -s '[:space:]' '\n' | rg "^$PROJECT_ROOT/" | \
      sed "s#^$PROJECT_ROOT/##" | \
      rg -v '^tests/g3r/pair_tokenizer_bind_runtime\.esk$' \
      >"$binder/source-closure.txt"
    cmp "$PROJECT_ROOT/native/g3r_c2_pair_tokenizer_bind_source_closure.txt" \
      "$binder/source-closure.txt"
    nm -u --format=posix "$binder/caller.o" \
      >"$binder/eshkol-object-undefined-symbols.txt"
    rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' \
      "$binder/eshkol-object-undefined-symbols.txt" >/dev/null
  fi
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 360s \
    "$binder/caller" "$saved" "$out/companions" \
    >"$binder/run.stdout" 2>"$binder/run.stderr"
  test ! -s "$binder/run.stderr"
  rg -x 'G3-R C2/FILE tokenizer bind PASS: checks=[1-9][0-9]*' \
    "$binder/run.stdout" >/dev/null
}
build_tokenizer_bind normal
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 360s \
  "$out/tokenizer-normal/caller" "$saved" "$out/companions" \
  >"$out/tokenizer-normal/repeat.stdout" \
  2>"$out/tokenizer-normal/repeat.stderr"
test ! -s "$out/tokenizer-normal/repeat.stderr"
cmp "$out/tokenizer-normal/run.stdout" "$out/tokenizer-normal/repeat.stdout"
build_tokenizer_bind sanitize
cmp "$out/tokenizer-normal/run.stdout" "$out/tokenizer-sanitize/run.stdout"
sha256sum "$saved" "$out/companions/valid.g3r" \
  >"$out/pair-after-reader.sha256"
cmp "$out/pair-before-reader.sha256" "$out/pair-after-reader.sha256" || \
  die "reader mutated the authoritative C2 image or companion"
printf 'G3-R authentic C2 SAVE pair-file gate PASS: normal/repeat/sanitize\n'
