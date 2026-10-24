#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
# Invoke only inside the orchestrator's supported-image, caller-owned heavy-lock
# gate. The host launcher must acquire heavy.lock before Docker and hold it
# through Docker exit/cleanup; the lock FD cannot cross the container boundary.
# This recipe does not acquire a second lock or launch parallel compiler jobs.
capture_identity() {
 git -C "$PROJECT_ROOT" diff --quiet HEAD -- || return 1
 [[ -z $(git -C "$PROJECT_ROOT" ls-files --others --exclude-standard) ]] || return 1
 local provenance cc_path cxx_path runtime
 provenance=$(eshkol_build_dir)/eshkol-transformer-provenance.tsv
 cc_path=$(tsv_value "$provenance" cc_path) || return 1
 cxx_path=$(tsv_value "$provenance" cxx_path) || return 1
 runtime=$(eshkol_build_dir)/eshkol-run
 printf 'source_commit\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
 printf 'source_tree\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse 'HEAD^{tree}')"
 sha256sum -- "$provenance" "$cc_path" "$cxx_path" "$runtime"
}
if [[ ${1:-} != --worker ]]; then
 out=${1:?usage: test-cpu4-eshkol-numerical.sh ABSOLUTE_FRESH_EVIDENCE [normal|sanitize]}
 mode=${2:-normal}
 [[ $out = /* ]] || die 'evidence directory must be absolute'
 mkdir -m 700 "$out"
 seal_evidence() {
  (cd "$out" && find . -type f ! -name SHA256SUMS -print0 |
    LC_ALL=C sort -z | xargs -0 -r sha256sum) >"$out.SHA256SUMS.tmp" || return 1
  mv -- "$out.SHA256SUMS.tmp" "$out/SHA256SUMS" || return 1
  sha256sum "$out/SHA256SUMS" >"$out.seal.sha256"
 }
 finish() {
  local status=$?
  trap - EXIT
  set +e
  printf '%s\n' "${worker_status:-NOT_RUN}" >"$out/worker.exit"
  if [[ -f "$out/custody.before.tsv" ]]; then
   capture_identity >"$out/custody.after.tsv"
   if (( $? )) || ! cmp -s "$out/custody.before.tsv" "$out/custody.after.tsv"; then
    printf 'FAIL: source/toolchain postflight changed or unavailable\n' >"$out/postflight.txt"
    status=74
   else
    printf 'PASS: source/toolchain postflight identical\n' >"$out/postflight.txt"
   fi
  else
   printf 'NOT_RUN: entry custody failed\n' >"$out/postflight.txt"
  fi
  printf '%d\n' "$status" >"$out/gate.exit"
  if ! seal_evidence; then
   status=74
   printf '%d\n' "$status" >"$out/gate.exit"
   seal_evidence || :
  fi
  exit "$status"
 }
 trap finish EXIT
 capture_identity >"$out/custody.before.tsv" || die 'committed source/toolchain identity required'
 {
  printf 'source_commit\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git -C "$PROJECT_ROOT" rev-parse 'HEAD^{tree}')"
  printf 'runner\t%s\n' 'scripts/test-cpu4-eshkol-numerical.sh'
  printf 'python_invocation\t%s\n' 'python3 -E -B tests/cpu4_numerical/check.py'
  printf 'mode\t%s\n' "$mode"
  printf 'lock_contract\t%s\n' 'caller-held host heavy.lock through Docker cleanup'
 } >"$out/source.tsv"
 cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "$out/toolchain.tsv"
 worker_status=125
 set +e
 bash "$(realpath "$0")" --worker "$out" "$mode"
 worker_status=$?
 set -e
 exit "$worker_status"
fi
shift
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx "$(lock_value supported_cc)" "$(lock_value supported_cxx)"
out=${1:?worker requires evidence directory}
mode=${2:-normal}
mkdir "$out/objects" "$out/cache" "$out/resources"
cd "$PROJECT_ROOT"
python3 -E -B - "$out/resources" <<'PY_RESOURCE'
from pathlib import Path
import sys
from tests.d2.test_resources import write_d1_resource
write_d1_resource(Path(sys.argv[1]), ((3,197,3,41,7,2,9,5,1),),
 fingerprint='sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704',vocab=256)
PY_RESOURCE
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
 -fno-common -fstack-protector-all -ffp-contract=off -fexcess-precision=standard -fno-fast-math
 -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src"
 -I "$PROJECT_ROOT/src/eshkol_transformer" -I "$(eshkol_source_dir)/inc"
 -DET_G3C4_NATIVE_OWNER_PRIVATE -DET_CPU4_TRAINING_PRIVATE -DET_CPU4_OBJECTIVE_PRIVATE -DET_G3R_CANDIDATE_RETIRE_PRIVATE)
optimization=(-O2) link_flags=()
case "$mode" in
 normal) ;;
 sanitize) optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
           link_flags=(-fsanitize=address,undefined -fno-omit-frame-pointer) ;;
 *) die 'expected normal or sanitize' ;;
esac
while IFS= read -r source; do
 extra=()
 case "$source" in
  native/i2_wave2_package_bridge.c) extra=(-DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE) ;;
  native/p1_identity.c) extra=(-DET_P1_TRUSTED_BUILD=1) ;;
  native/d2_native.c) extra=(-DET_TR3_C_D2_RESTORE) ;;
 esac
 object=${source//\//_}
 "$cc" "${flags[@]}" "${optimization[@]}" "${extra[@]}" -MMD -MF "$out/objects/$object.d" \
  -c "$PROJECT_ROOT/$source" -o "$out/objects/$object.o" \
  >"$out/objects/$object.stdout" 2>"$out/objects/$object.stderr"
done <"$PROJECT_ROOT/native/cpu4_training_native_sources.txt"
"$cc" "${flags[@]}" "${optimization[@]}" -MMD \
 -MF "$out/objects/cpu4_numerical_observer.d" \
 -c "$PROJECT_ROOT/tests/cpu4_numerical/observer.c" \
 -o "$out/objects/cpu4_numerical_observer.o" \
 >"$out/objects/cpu4_numerical_observer.stdout" \
 2>"$out/objects/cpu4_numerical_observer.stderr"
ar rcsD "$out/libcpu4_binding.a" "$out"/objects/*.o
printf '#!/usr/bin/env bash\nexec %q' "$cxx" >"$out/cxx-link"
if (( ${#link_flags[@]} )); then printf ' %q' "${link_flags[@]}" >>"$out/cxx-link"; fi
printf ' "$@"\n' >>"$out/cxx-link"
chmod 0500 "$out/cxx-link"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
 XDG_CACHE_HOME="$out/cache" ESHKOL_CXX_COMPILER="$out/cxx-link" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
 timeout --foreground --signal=TERM --kill-after=5s 1200s "$(eshkol_build_dir)/eshkol-run" \
 --dump-ir --strict-types --optimize 0 --no-stdlib \
 -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/internal/c1/lib" -I "$PROJECT_ROOT/internal/t2/lib" \
 -I "$PROJECT_ROOT/internal/t1/lib" -I "$PROJECT_ROOT/internal/d2/lib" -I "$PROJECT_ROOT/src" \
 -I "$PROJECT_ROOT/lib" -I "$PROJECT_ROOT/native" -L "$out" --lib cpu4_binding \
 "$PROJECT_ROOT/tests/cpu4_numerical/runtime.esk" -o "$out/binding" \
 >"$out/compile.stdout" 2>"$out/compile.stderr"
for repeat in 1 2; do
 env ESHKOL_ARENA_POISON=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
 timeout --foreground --signal=TERM --kill-after=5s 120s "$out/binding" \
 "$out/resources" \
 >"$out/run-$repeat.stdout" 2>"$out/run-$repeat.stderr"
 test ! -s "$out/run-$repeat.stderr"
done
cmp "$out/run-1.stdout" "$out/run-2.stdout"
grep -E '^CPU4 genuine Eshkol training microbatch PASS: checks=[1-9][0-9]*$' "$out/run-1.stdout"
python3 -E -B "$PROJECT_ROOT/tests/cpu4_numerical/check.py" \
 "$out/run-1.stdout" >"$out/oracle.stdout"
python3 -E -B - "$PROJECT_ROOT/tests/cpu4_numerical/check.py" \
 >"$out/nonfinite-negative.stdout" 2>"$out/nonfinite-negative.stderr" <<'PY_NONFINITE'
import importlib.util
import math
import sys
spec = importlib.util.spec_from_file_location("cpu4_oracle", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if sys.flags.optimize != 0:
    raise SystemExit("Python optimization changed numerical assertions")
try:
    module.near(0.0, math.nan, 0.0, 0.0, "nonfinite expected")
except AssertionError:
    print("CPU4 nonfinite-reference negative PASS")
else:
    raise SystemExit("nonfinite reference was accepted")
PY_NONFINITE
python3 -E -B - "$out/run-1.stdout" "$out/mutated-logit.stdout" \
 >"$out/mutation-prepare.stdout" 2>"$out/mutation-prepare.stderr" <<'PY_MUTATE'
from pathlib import Path
import sys
original = Path(sys.argv[1]).read_text()
lines = original.splitlines(keepends=True)
matches = [i for i, line in enumerate(lines) if line.startswith("NUM LOGIT 0 0 ")]
if len(matches) != 1:
    raise SystemExit("expected exactly one first-logit transcript row")
lines[matches[0]] = "NUM LOGIT 0 0 1232348160\n"  # finite f32 1,000,000
Path(sys.argv[2]).write_text("".join(lines))
PY_MUTATE
if python3 -E -B "$PROJECT_ROOT/tests/cpu4_numerical/check.py" \
 "$out/mutated-logit.stdout" >"$out/mutant.stdout" 2>"$out/mutant.stderr"; then
 die 'mutated-logit transcript was accepted'
fi
grep -q 'logit 0:' "$out/mutant.stderr" || die 'mutation failed for the wrong reason'
printf 'CPU4 mutated-logit negative PASS\n' >"$out/mutant-verdict.stdout"
cat "$out/oracle.stdout"
cat "$out/nonfinite-negative.stdout" "$out/mutant-verdict.stdout"
