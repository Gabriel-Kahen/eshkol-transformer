#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx "$(lock_value supported_cc)" "$(lock_value supported_cxx)"
out=${1:?usage: test-cpu4-training-schedule.sh FRESH_EVIDENCE [normal|sanitize]}
mode=${2:-normal}
mkdir -m 700 "$out"
mkdir "$out/objects" "$out/cache" "$out/resources"
cd "$PROJECT_ROOT"
python3 - "$out/resources" <<'PY_RESOURCE'
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
 "$PROJECT_ROOT/tests/cpu4_objective/schedule_test.esk" -o "$out/binding" \
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
