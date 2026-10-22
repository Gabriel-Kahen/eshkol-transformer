#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence=${1:?usage: test-cpu4-candidate-producer.sh FRESH_EVIDENCE [normal|sanitize]}
mode=${2:-normal}
optimization=(-O2)
link_flags=()
runtime=(env ESHKOL_ARENA_POISON=1)
case "$mode" in
  normal) ;;
  sanitize)
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    link_flags=(-fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ESHKOL_ARENA_POISON=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1) ;;
  *) echo 'expected normal or sanitize' >&2; exit 2 ;;
esac
# Integration owns the heavy.lock and bounded host/container dispatch. Never
# reuse output or remove failed build/IR artifacts from this proving gate.
mkdir -m 700 "$evidence"
mkdir -m 700 "$evidence/objects" "$evidence/cache"
# Host Guile checks are a separately reviewed prerequisite; the pinned image
# has no Guile. Authenticate that frozen receipt and its exact owned sources.
host_checks=${CPU4_CANDIDATE_HOST_SOURCE_EVIDENCE:?mount the frozen host source evidence}
host_seal=${CPU4_CANDIDATE_HOST_SOURCE_SEAL_SHA256:?pin its reviewed SHA256SUMS digest}
python3 -B - "$host_checks" "$host_seal" "$PROJECT_ROOT" \
  >"$evidence/host-source-prerequisite.stdout" <<'VERIFY_HOST'
import hashlib,json,pathlib,subprocess,sys
receipt=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; project=pathlib.Path(sys.argv[3])
assert hashlib.sha256((receipt/'SHA256SUMS').read_bytes()).hexdigest()==expected
subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=receipt,check=True)
result=json.loads((receipt/'RESULTS.json').read_text())
assert result['results'][0]['exit']==0
assert 'CPU4 candidate source/planner PASS:' in (receipt/'0.stdout').read_text()
original=pathlib.Path(result['worktree'])
for line in (receipt/'OWNED-SOURCES.sha256').read_text().splitlines():
    digest,path=line.split('  ',1)
    target=project/pathlib.Path(path).relative_to(original)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest,str(target)
print('Frozen host Guile/source prerequisite PASS; Guile not run in container')
VERIFY_HOST
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
  -fno-common -fstack-protector-all -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
  -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc"
  -DET_G3C4_NATIVE_OWNER_PRIVATE -DET_CPU4_CANDIDATE_PRIVATE
  -DET_CPU4_TRAINING_PRIVATE -DET_G3R_CANDIDATE_RETIRE_PRIVATE)
# The predecessor C4 model-authority gate substitutes M3T F32 and M3 I64
# integration TUs exactly once. Add O2 only for native candidate exclusion.
while IFS= read -r source; do
  [[ -n "$source" && "$source" != \#* ]] || continue
  extra=()
  case "$source" in
    native/i2_wave2_package_bridge.c)
      extra=(-DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE) ;;
    native/p1_identity.c) extra=(-DET_P1_TRUSTED_BUILD=1) ;;
  esac
  object=${source//\//_}
  "$cc" "${flags[@]}" "${optimization[@]}" "${extra[@]}" \
    -MMD -MF "$evidence/objects/$object.d" \
    -c "$PROJECT_ROOT/$source" -o "$evidence/objects/$object.o" \
    >"$evidence/objects/$object.stdout" 2>"$evidence/objects/$object.stderr"
done <"$PROJECT_ROOT/native/cpu4_candidate_native_sources.txt"
ar rcsD "$evidence/libcpu4_candidate.a" "$evidence"/objects/*.o
cat >"$evidence/cxx-link" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") ${link_flags[*]} "\$@"
WRAPPER
chmod 0500 "$evidence/cxx-link"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$evidence/cache" ESHKOL_CXX_COMPILER="$evidence/cxx-link" \
  ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "$(eshkol_build_dir)/eshkol-run" --dump-ir --strict-types --optimize 0 --no-stdlib \
  -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" \
  -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/src" \
  -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" \
  -L "$evidence" --lib cpu4_candidate \
  "$PROJECT_ROOT/tests/cpu4_candidate/producer_test.esk" \
  -o "$evidence/producer" >"$evidence/compile.stdout" 2>"$evidence/compile.stderr"
test -s "$evidence/producer.ll"
"${runtime[@]}" timeout --foreground --signal=TERM --kill-after=5s 240s \
  "$evidence/producer" >"$evidence/run.stdout" 2>"$evidence/run.stderr"
test ! -s "$evidence/run.stderr"
grep -E '^CPU4 genuine candidate producer PASS: checks=[1-9][0-9]*$' "$evidence/run.stdout"
if [[ "$mode" == normal ]]; then
  "${runtime[@]}" timeout --foreground --signal=TERM --kill-after=5s 240s \
    "$evidence/producer" >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
  test ! -s "$evidence/repeat.stderr"
  cmp "$evidence/run.stdout" "$evidence/repeat.stdout"
fi
