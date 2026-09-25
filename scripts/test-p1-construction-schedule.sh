#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in python3 sha256sum timeout; do require_command "$command"; done
evidence="${1:-$(project_build_dir)/p1-construction-schedule}"
mkdir -p "$evidence/cache"
python3 - "$PROJECT_ROOT/internal/p1/lib/transformer/module.esk" \
  "$PROJECT_ROOT/tests/p1/construction_schedule_identity_test.esk" \
  "$evidence/witness.esk" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
anchor = '  )))\n\n(define (module-parameters module)'
assert source.count(anchor) == 1
probe = '''  ;; Test-only probes; the production trusted surface is unchanged.
  (lambda (actual retained) (construction-schedule-equal? actual retained))
  (lambda ()
    (if (= 0 (p1-native-handle-create native-context))
        (p1-native-result-ptr native-context) #f))
'''
Path(sys.argv[3]).write_text(source.replace(anchor, probe + anchor, 1)
                              + '\n' + Path(sys.argv[2]).read_text())
PY
"$PROJECT_ROOT/scripts/build-p1-identity.sh" "$evidence/identity" normal trusted \
  >"$evidence/build.stdout"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="$evidence/cache" \
  ESHKOL_CXX_COMPILER="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)" \
  ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s \
  "$(eshkol_build_dir)/eshkol-run" --strict-types --no-stdlib \
  -I "$PROJECT_ROOT/internal/p1/lib" -I "$PROJECT_ROOT/lib" \
  -I "$PROJECT_ROOT/native" -L "$evidence/identity/trusted" \
  --lib eshkol_transformer_p1_identity \
  "$evidence/witness.esk" \
  -o "$evidence/caller" >"$evidence/compile.stdout" \
  2>"$evidence/compile.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "$evidence/caller" >"$evidence/stdout" 2>"$evidence/stderr"
test ! -s "$evidence/stderr"
grep -Fx 'P1 construction schedule PASS: 6' "$evidence/stdout" >/dev/null
sha256sum "$evidence/caller" >"$evidence/binary.sha256"
cat "$evidence/stdout"
