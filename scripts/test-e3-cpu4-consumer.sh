#!/usr/bin/env bash
# Direct uninstalled candidate-OFF consumer. Primary owns heavy.lock; one job.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 1 && "$1" == /* && ! -e "$1" && ! -L "$1" ]] || die 'usage: test-e3-cpu4-consumer.sh FRESH_ABSOLUTE_OUTPUT'
out=$1
mkdir -m 700 "$out"
seal() {
  local status=$?
  printf '%s\n' "$status" >"$out/gate.exit"
  python3 -B - "$out" <<'PYSEAL'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for path in sorted(root.rglob('*')):
    if path.is_file() and path!=root/'SHA256SUMS':
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
        rows.append(digest.hexdigest()+'  '+str(path.relative_to(root))+'\n')
(root/'SHA256SUMS').write_text(''.join(rows))
PYSEAL
  return "$status"
}
trap seal EXIT
cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die 'consumer source must be clean and committed'
git rev-parse HEAD HEAD^{tree} >"$out/source.txt"
printf 'candidate_mode=off\ncompile_jobs=1\nparameter_control_bytes=64\n' >"$out/profile.txt"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx "$(lock_value supported_cc)" "$(lock_value supported_cxx)"
runner="$(eshkol_build_dir)/eshkol-run"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "$out/toolchain.tsv"
run() {
  local label=$1 status
  shift
  set +e
  "$@" >"$out/$label.stdout" 2>"$out/$label.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/$label.exit"
  [[ $status == 0 ]] || die "$label failed: $status"
}
run source python3 -B -m unittest -v tests.e3_cpu4_consumer.test_source
run resources python3 -B -m tests.e3_cpu4_consumer.resources "$out/resources"
sha256sum --quiet -c tests/e3_private/asan_ir_sources.sha256
cp tests/e3_private/asan_ir_sources.sha256 "$out/asan-ir-sources.sha256"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
  -fno-common -fstack-protector-all -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
  -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc"
  -DET_G3C4_NATIVE_OWNER_PRIVATE -DET_CPU4_TRAINING_PRIVATE
  -DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3R_CANDIDATE_RETIRE_PRIVATE)
includes=(-I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib"
  -I "$PROJECT_ROOT/internal/t2/lib" -I "$PROJECT_ROOT/internal/t1/lib"
  -I "$PROJECT_ROOT/internal/d2/lib" -I "$PROJECT_ROOT/internal/e3/lib"
  -I "$PROJECT_ROOT/src" -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native")
for mode in normal sanitize; do
  mkdir "$out/$mode" "$out/$mode/objects" "$out/$mode/cache"
  selected=(-O2) link_flags=()
  if [[ "$mode" == sanitize ]]; then
    selected=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    link_flags=(-fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  while IFS= read -r source; do
    extra=()
    case "$source" in
      native/i2_wave2_package_bridge.c) extra=(-DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE) ;;
      native/p1_identity.c) extra=(-DET_P1_TRUSTED_BUILD=1) ;;
      native/e3_d2_native.c) extra=(-DET_TR3_C_D2_RESTORE) ;;
    esac
    object=${source//\//_}
    run "$mode.native.$object" timeout --foreground --signal=TERM --kill-after=5s 120s \
      "$cc" "${flags[@]}" "${selected[@]}" "${extra[@]}" -MMD \
      -MF "$out/$mode/objects/$object.d" -c "$PROJECT_ROOT/$source" -o "$out/$mode/objects/$object.o"
    [[ ! -s "$out/$mode.native.$object.stderr" ]] || die 'native compiler diagnostics'
  done < <(cat native/e3_cpu4_consumer_native_sources.txt; printf '%s\n' tests/e3_cpu4_consumer/helper.c)
  run "$mode.archive" ar rcsD "$out/$mode/libcpu4_consumer.a" "$out/$mode/objects/"*.o
  printf '#!/usr/bin/env bash\nexec %q' "$cxx" >"$out/$mode/cxx-link"
  if (( ${#link_flags[@]} )); then printf ' %q' "${link_flags[@]}" >>"$out/$mode/cxx-link"; fi
  printf ' "$@"\n' >>"$out/$mode/cxx-link"
  chmod 0500 "$out/$mode/cxx-link"
  clean_env=(env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0
    XDG_CACHE_HOME="$out/$mode/cache" ESHKOL_CXX_COMPILER="$out/$mode/cxx-link"
    ESHKOL_LIB_DIR="$PROJECT_ROOT/lib")
  if [[ "$mode" == normal ]]; then
    run raw.object "${clean_env[@]}" timeout --foreground --signal=TERM --kill-after=5s 1200s \
      "$runner" --strict-types --no-stdlib --optimize 0 "${includes[@]}" \
      -L "$out/$mode" --lib cpu4_consumer --emit-object --dump-ir \
      --emit-depfile "$out/runtime.d" "$PROJECT_ROOT/tests/e3_cpu4_consumer/runtime.esk" -o "$out/runtime.raw.o"
    [[ -s "$out/runtime.raw.o.ll" && -s "$out/runtime.raw.o.bc" && -s "$out/runtime.d" ]] || die 'raw compiler artifacts missing'
    object="$out/runtime.raw.o"
  else
    run asan.prepare python3 -B tests/shared/prepare_eshkol_asan_ir.py \
      "$out/runtime.raw.o.ll" "$out/runtime.asan.ll" "$out/runtime.asan.json"
    run asan.compile "$cc" "${selected[@]}" -x ir -c "$out/runtime.asan.ll" -o "$out/runtime.asan.o"
    run asan.imports nm -u --format=posix "$out/runtime.asan.o"
    [[ ! -s "$out/asan.imports.stderr" ]] || die 'ASan nm diagnostics'
    run asan.proof python3 -B tests/e3_private/check_asan_ir.py \
      "$out/runtime.raw.o.ll" "$out/runtime.asan.ll" "$out/runtime.asan.json" \
      "$out/runtime.asan.o" "$out/asan.imports.stdout"
    sha256sum --quiet -c tests/e3_private/asan_ir_sources.sha256
    object="$out/runtime.asan.o"
  fi
  run "$mode.link" "${clean_env[@]}" timeout --foreground --signal=TERM --kill-after=5s 120s \
    "$runner" --no-stdlib -L "$out/$mode" --lib cpu4_consumer "$object" -o "$out/$mode/runtime"
  for gradient in absent present; do
    for repeat in 1 2; do
      label="$mode.$gradient.$repeat"
      run "$label" env ESHKOL_ARENA_POISON=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
        "$out/$mode/runtime" "$out/resources/corpus" "$gradient"
      [[ ! -s "$out/$label.stderr" ]] || die 'consumer runtime diagnostics'
      run "$label.observation" python3 -B -m tests.e3_cpu4_consumer.observation "$out/$label.stdout"
    done
    cmp "$out/$mode.$gradient.1.stdout" "$out/$mode.$gradient.2.stdout"
  done
done
for gradient in absent present; do
  cmp "$out/normal.$gradient.1.stdout" "$out/sanitize.$gradient.1.stdout"
done
git rev-parse HEAD HEAD^{tree} >"$out/source-post.txt"
cmp "$out/source.txt" "$out/source-post.txt"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die 'source changed during gate'
printf 'CPU4 direct consumer normal/sanitized mechanical gate complete; independent result review pending\n'
