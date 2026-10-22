#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
out=${1:?usage: test-cpu4-typed-control-alias.sh EVIDENCE_DIR}
mkdir -p "$out"
case "${2:-lease}" in
  lease) candidate=0 ;;
  candidate) candidate=1 ;;
  *) echo 'expected lease or candidate alias profile' >&2; exit 1 ;;
esac
cc=$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)
python3 - "$out/cc-cpu4" "$cc" <<'PY'
from pathlib import Path
import shlex, sys
p=Path(sys.argv[1])
p.write_text('#!/usr/bin/env bash\nexec '+shlex.quote(sys.argv[2])+' -DET_CPU4_TRAINING_PRIVATE "$@"\n')
p.chmod(0o755)
PY
CPU4_TYPED_CANDIDATE_TESTING="$candidate" CC="$out/cc-cpu4" bash "$PROJECT_ROOT/scripts/test-g3c4-p2g2-typed-result.sh" "$out/combined"
