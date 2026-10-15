#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mktemp -d /tmp/e3-n4-mask-census.XXXXXX)
trap 'rm -rf -- "$out"' EXIT
cd -- "$root"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.e3_n4_mask.test_source -v
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.e3_n4_mask.prepare --output "$out/resources"
python3 - "$root/native/d2_native.c" "$out/d2-mutant.c" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
old = "batch->mask_data[destination + index] = 1u;"
new = "batch->mask_data[destination + index + (count == 1u ? 1u : 0u)] = 1u;"
if source.count(old) != 1:
    raise SystemExit("D2 mask writer changed; mutant cannot be attributed")
Path(sys.argv[2]).write_text(source.replace(old, new))
PY
cc=${CC:-cc}
flags=(-std=c11 -O1 -Wall -Wextra -Werror -Wpedantic
       -I "$root/include" -I "$root/native")
for mode in normal mutant; do
  source="$root/native/d2_native.c"
  if [[ "$mode" == mutant ]]; then source="$out/d2-mutant.c"; fi
  "$cc" "${flags[@]}" -c "$source" -o "$out/$mode-d2.o"
  "$cc" "${flags[@]}" -Wno-maybe-uninitialized -c "$root/native/i64_tensor.c" \
    -o "$out/$mode-i64.o"
  "$cc" "${flags[@]}" -c "$root/native/kernel_abi.c" -o "$out/$mode-k1.o"
  "$cc" "${flags[@]}" -c "$root/tests/e3_n4_mask/native_mask_census.c" \
    -o "$out/$mode-test.o"
  "$cc" "$out/$mode-d2.o" "$out/$mode-i64.o" "$out/$mode-k1.o" \
    "$out/$mode-test.o" -o "$out/$mode"
done
"$out/normal" "$out/resources" >"$out/normal.stdout"
grep -Fx 'E3 N4 D1/D2 MASK CENSUS PASS: 4 rows' "$out/normal.stdout" >/dev/null
set +e
"$out/mutant" "$out/resources" >"$out/mutant.stdout" 2>"$out/mutant.stderr"
status=$?
set -e
test "$status" -eq 71
test ! -s "$out/mutant.stderr"
grep -Fx 'MASK FIRST DIVERGENCE p2 01 expected 10' "$out/mutant.stdout" >/dev/null
echo 'E3 N4 MASK CENSUS HOST PASS: 4 checked rows; mutant first p2 mask'
