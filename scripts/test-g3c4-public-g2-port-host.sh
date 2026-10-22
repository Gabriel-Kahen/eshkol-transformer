#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export G3C4_TEST_ROOT="$root" GUILE_AUTO_COMPILE=0
guile -s "$root/tests/g3c4/p2_g2_public_ticket_host_test.scm"
guile -s "$root/tests/g3c4/p2_g2_public_typed_result_host_test.scm"
for mode in reinstall alias-roots held-abort; do
  set +e
  G3C4_TICKET_CASE="$mode" guile -s "$root/tests/g3c4/p2_g2_public_ticket_host_test.scm" \
    >"$tmp/$mode.stdout" 2>"$tmp/$mode.stderr"
  status=$?
  set -e
  [[ "$status" == 134 ]] || { cat "$tmp/$mode.stderr" >&2; exit 1; }
done
python3 - "$root" "$tmp" <<'PY'
from pathlib import Path
import sys
root, tmp = map(Path, sys.argv[1:])
source = (root / 'native/g3c4_p2g2_terminal_commit_extension.esk').read_text()
marker = '(vector-set! barrier 0 #t)'
assert source.count(marker) == 1
(tmp / 'postzero-fault.esk').write_text(source.replace(marker,
    marker + '\n                  (g3c4-fail \'invalid-state operation "postzero fault")', 1))
PY
set +e
G3C4_TICKET_CASE=postzero-fault G3C4_TEST_SOURCE="$tmp/postzero-fault.esk" \
  guile -s "$root/tests/g3c4/p2_g2_public_ticket_host_test.scm" \
  >"$tmp/postzero.stdout" 2>"$tmp/postzero.stderr"
status=$?
set -e
[[ "$status" == 134 ]] || { cat "$tmp/postzero.stderr" >&2; exit 1; }
printf 'G3-C4 public ticket fail-stop controls PASS: cases=4\n'
