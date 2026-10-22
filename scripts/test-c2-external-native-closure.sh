#!/usr/bin/env bash
# Exercise the real E1B closure collector without compiling a package.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
repository=$PROJECT_ROOT
scratch="$(mktemp -d "${TMPDIR:-/tmp}/c2-native-closure.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$repository/scripts/build-e1b-consumer.sh" "$scratch" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
start = source.index('if [[ -n "${package_native_source_closure}" ]]; then')
end = source.index('\nfi\n', start) + len('\nfi\n')
collector = source[start:end]
Path(sys.argv[2], 'collector.sh').write_text(collector)
old = collector.replace('      "${package_policy}" == c2-wave2-aggregate || \\\n', '', 1)
assert old != collector
Path(sys.argv[2], 'old-collector.sh').write_text(old)
PY
PROJECT_ROOT="$scratch/repo"
e1b_tmp="$scratch/output"
mkdir -p "$PROJECT_ROOT/native" "$e1b_tmp" "$scratch/external/inc/eshkol"
touch "$PROJECT_ROOT/native/header.h"
headers=(eshkol.h exhaustive_dispatch.h memory_abi_v2.h ad_node_registry.def)
for name in "${headers[@]}"; do touch "$scratch/external/inc/eshkol/$name"; done
mkdir -p "$PROJECT_ROOT/.deps"
cp -a "$scratch/external" "$PROJECT_ROOT/.deps/eshkol-src"
raw_private_root="$PROJECT_ROOT/unused-root"
raw_package_bridge="$PROJECT_ROOT/unused-bridge"
raw_package_renames="$PROJECT_ROOT/unused-renames"
raw_public_exports="$PROJECT_ROOT/unused-exports"
raw_include_dirs=()
source "$repository/scripts/m3t-package-policy.sh"
package_policy=c2-wave2-aggregate
package_dependency_prefix=m3t
g3g_package_policy=0
e3_zero_discovery=0
package_native_source_closure="$scratch/expected.txt"
package_native_depfiles=("$scratch/input.d")
printf 'native/header.h\n' >"$package_native_source_closure"
for name in "${headers[@]}"; do
  printf '.deps/eshkol-src/inc/eshkol/%s\n' "$name" >>"$package_native_source_closure"
done
write_depfile() {
  printf 'bridge.o: %s/native/header.h' "$PROJECT_ROOT" >"$scratch/input.d"
  for name in "${headers[@]}"; do
    printf ' %s/inc/eshkol/%s' "$e1b_source" "$name" >>"$scratch/input.d"
  done
  printf '\n' >>"$scratch/input.d"
}
for e1b_source in "$PROJECT_ROOT/.deps/eshkol-src" "$scratch/external"; do
  write_depfile
  (source "$scratch/collector.sh")
  cmp "$package_native_source_closure" "$e1b_tmp/native-source-closure.txt"
done
# The old policy silently drops all authenticated external compiler headers.
if (source "$scratch/old-collector.sh") >"$scratch/old.stdout" 2>"$scratch/old.stderr"; then
  die 'old C2 collector unexpectedly retained external compiler headers'
fi
rg -F 'trusted native source closure drifted' "$scratch/old.stderr" >/dev/null
# A symlink alias and an unrelated source root must still fail closed.
ln -s external "$scratch/compiler-alias"
reject_path() {
  local bad=$1 expected=$2
  printf 'bridge.o: %s\n' "$bad" >"$scratch/input.d"
  if (source "$scratch/collector.sh") >"$scratch/bad.stdout" 2>"$scratch/bad.stderr"; then
    die 'unsafe C2 dependency was accepted'
  fi
  rg -Fx "error: $expected" "$scratch/bad.stderr" >/dev/null
}
touch "$scratch/foreign.h"
reject_path "$scratch/compiler-alias/inc/eshkol/eshkol.h" \
  'M3T native depfile uses a symlink alias'
reject_path "$scratch/foreign.h" \
  'M3T native dependency is outside reviewed source roots'
printf 'C2 external native closure PASS: canonical/external/old-policy/alias/foreign\n'
