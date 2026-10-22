#!/usr/bin/env bash
# Native no-grad C4/lease/frame evidence. Caller owns TEAM heavy.lock.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
[[ $# == 1 ]] || die "usage: test-e3-cpu4-frame.sh FRESH_OUTPUT_DIRECTORY"
out=$(realpath -m -- "$1")
[[ ! -e "$out" ]] || die "CPU4 frame output must be fresh"
mkdir -p "$out"
seal() {
  local status=$?
  printf '%s\n' "$status" >"$out/gate.exit"
  python3 -B - "$out" <<'PY'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1])
with (root/'SHA256SUMS').open('w') as stream:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name!='SHA256SUMS':
            stream.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(root))+'\n')
PY
  return "$status"
}
trap seal EXIT
cd "$PROJECT_ROOT"
git diff --quiet HEAD -- || die "CPU4 frame source is dirty"
[[ -z "$(git ls-files --others --exclude-standard)" ]] || die "CPU4 frame source has untracked files"
git rev-parse HEAD HEAD^{tree} >"$out/source.txt"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "$out/toolchain.tsv"
cc=$(tsv_value "$out/toolchain.tsv" cc_path)
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fno-fast-math -fstack-protector-all
  -I "$PROJECT_ROOT/include" -I "$PROJECT_ROOT/native" -I "$PROJECT_ROOT/src" -I "$out")
python3 -B tests/e3_cpu4_frame/reference.py "$out/reference.h"
python3 -B tests/e3_cpu4_frame/reference.py "$out/reference-repeat.h"
cmp "$out/reference.h" "$out/reference-repeat.h"
for profile in normal sanitize; do
  selected=(-O2)
  [[ "$profile" != sanitize ]] || selected=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  timeout --foreground --signal=TERM --kill-after=5s 120s "$cc" "${flags[@]}" "${selected[@]}" \
    tests/e3_cpu4_frame/native.c tests/m3t/kernel_fail_allocator.c \
    native/n3k_primitives_provider.c native/n2_primitives_provider.c \
    native/a2_attention_provider.c native/indexed_cross_entropy.c \
    native/l3s_masked_objective_provider.c native/e3_evaluation_metrics_provider.c \
    -lm -o "$out/$profile" >"$out/compile-$profile.stdout" 2>"$out/compile-$profile.stderr"
  [[ ! -s "$out/compile-$profile.stderr" ]] || die "CPU4 frame compile diagnostics"
  for repeat in 1 2; do
    env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s "$out/$profile" \
      >"$out/$profile-$repeat.stdout" 2>"$out/$profile-$repeat.stderr"
    [[ ! -s "$out/$profile-$repeat.stderr" ]] || die "CPU4 frame runtime diagnostics"
  done
  cmp "$out/$profile-1.stdout" "$out/$profile-2.stdout"
done
cmp "$out/normal-1.stdout" "$out/sanitize-1.stdout"
cat "$out/normal-1.stdout"
