#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

G3C4_TEST_ROOT="$root" GUILE_AUTO_COMPILE=0 \
  guile -s "$root/tests/g3c4/p2_g2_typed_wrapper_rollback_test.scm"

python3 - "$root" "$tmp" <<'PY'
from pathlib import Path
import sys
root, tmp = map(Path, sys.argv[1:])
source = (root / "native/g3c4_p2g2_typed_result_extension.esk").read_text()
mutants = {
    "no-registry-rollback": (
        "(vector-set! g3c4-registry 0 old-registry)", "#t"),
    "unchecked-create-copy": (
        "(g3c4-native-check\n            (g3c4-p2g2-native-typed-copy native count bytes extent)\n            operation)",
        "(g3c4-p2g2-native-typed-copy native count bytes extent)"),
    "unchecked-fresh-copy": (
        "(g3c4-native-check\n          (g3c4-p2g2-native-typed-copy native count bytes (+ 8 payload))\n          operation)",
        "(g3c4-p2g2-native-typed-copy native count bytes (+ 8 payload))"),
}
for name, (before, after) in mutants.items():
    assert source.count(before) == 1, name
    (tmp / f"{name}.esk").write_text(source.replace(before, after, 1))
PY

for mutant in no-registry-rollback unchecked-create-copy unchecked-fresh-copy; do
  if G3C4_TEST_ROOT="$root" G3C4_TEST_SOURCE="$tmp/$mutant.esk" \
       GUILE_AUTO_COMPILE=0 \
       guile -s "$root/tests/g3c4/p2_g2_typed_wrapper_rollback_test.scm" \
       >"$tmp/$mutant.stdout" 2>"$tmp/$mutant.stderr"; then
    echo "undetected typed wrapper mutant: $mutant" >&2
    exit 1
  fi
done
printf 'G3-C4 private typed wrapper host mutants detected: 3\n'
