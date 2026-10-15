#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-current-held.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
common=(-std=c11 -Wall -Wextra -Werror -Wno-unused-parameter
        -Wno-misleading-indentation -ffunction-sections -fdata-sections
        -I "$root/include" -I "$root/native" -I "$root/src"
        -I "$root/src/eshkol_transformer")
cc "${common[@]}" -O0 -c "$root/src/eshkol_transformer/g3t_transport.c" -o "$out/off.o"
if nm -g --defined-only "$out/off.o" | grep -Fq et_g3r_private_g3t_save_begin_v1; then
  echo "held owner escaped feature-off object" >&2
  exit 1
fi
# Causal regression: removing the no-replay arm must break the held-owner
# translation unit, while the unmodified source compiles in the modes below.
python3 - "$root" "$out" <<'PY'
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:])
source = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
arm = "#else\n         c->history_pending_g1\n#endif"
assert source.count(arm) == 1
mutant = out / "g3t_missing_no_replay_arm.c"
mutant.write_text(source.replace(arm, "#endif", 1))
fixture = (root / "tests/g3r/current_history_held_owner_test.c").read_text()
include = '#include "../../src/eshkol_transformer/g3t_transport.c"'
assert fixture.count(include) == 1
(out / "missing_arm_test.c").write_text(
    fixture.replace(include, f'#include "{mutant}"', 1))
PY
if cc "${common[@]}" -O0 -c "$out/missing_arm_test.c" \
  -o "$out/missing_arm.o" 2>"$out/missing_arm.stderr"; then
  echo "held-owner compile accepted missing no-replay history arm" >&2
  exit 1
fi
grep -F 'g3t_history_append_preflight' "$out/missing_arm.stderr" >/dev/null
for mode in normal repeat sanitizer; do
  flags=(-O0)
  if [[ "$mode" == sanitizer ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  cc "${common[@]}" "${flags[@]}" \
    "$root/tests/g3r/current_history_held_owner_test.c" \
    -Wl,--gc-sections -o "$out/$mode"
  if [[ "$mode" == sanitizer ]]; then
    ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 "$out/$mode"
  else
    "$out/$mode"
  fi
done
