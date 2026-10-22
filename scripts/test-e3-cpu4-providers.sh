#!/usr/bin/env bash
# Standalone provider evidence only; no model/frame/trainer acceptance.
# Caller holds TEAM.md's shared heavy.lock, including across a Docker launch.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar awk cmp cp env git grep nm objdump python3 realpath sha256sum timeout; do require_command "$command"; done
[[ $# == 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "$out" ]] || die "CPU4 provider output must be fresh"
mkdir -p "$out"
seal() {
  local status=$?
  printf '%s\n' "$status" >"$out/gate.exit"
  python3 -B - "$out" <<'PY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1])
with (root/'SHA256SUMS').open('w') as stream:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path!=root/'SHA256SUMS':
            digest=hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    digest.update(chunk)
            stream.write(f'{digest.hexdigest()}  {path.relative_to(root)}\n')
PY
  return "$status"
}
trap seal EXIT
git -C "$PROJECT_ROOT" diff --quiet HEAD -- || die "CPU4 source is dirty"
[[ -z "$(git -C "$PROJECT_ROOT" ls-files --others --exclude-standard)" ]] || die "CPU4 source has untracked files"
git -C "$PROJECT_ROOT" rev-parse HEAD HEAD^{tree} >"$out/source.txt"
git -C "$PROJECT_ROOT" ls-files -z | python3 -B -c '
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1])
for name in sorted(sys.stdin.buffer.read().split(b"\0")):
    if name:
        path=root/name.decode();print(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+name.decode())
' "$PROJECT_ROOT" >"$out/source.sha256"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cp "$provenance" "$out/toolchain.tsv"
cc="$(tsv_value "$provenance" cc_path)"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -frounding-math -fstack-protector-all
  -fPIC -I "$PROJECT_ROOT/include" -I "$out")
log() {
  local name=$1 status=0;shift
  "$@" >"$out/$name.stdout" 2>"$out/$name.stderr" || status=$?
  printf '%s\n' "$status" >"$out/$name.exit"
  [[ "$status" == 0 ]] || die "CPU4 $name failed; evidence retained"
}
cd "$PROJECT_ROOT"
log source-check python3 -B -m unittest discover -s tests/e3_cpu4 -p test_source.py -v
for repeat in a b; do
  log "oracle-$repeat" python3 -B tests/e3_cpu4/reference.py "$out/reference-$repeat.h"
done
cmp "$out/reference-a.h" "$out/reference-b.h"
cp "$out/reference-a.h" "$out/reference.h"
sources=(kernel_abi indexed_cross_entropy l3s_masked_objective_provider e3_evaluation_metrics_provider)
for profile in a b sanitized; do
  mkdir "$out/$profile"
  selected=(-O2)
  [[ "$profile" != sanitized ]] || selected=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  objects=()
  for source in "${sources[@]}"; do
    object="$out/$profile/$source.o";objects+=("$object")
    log "compile-$profile-$source" timeout --foreground --signal=TERM --kill-after=5s 60s \
      "$cc" "${flags[@]}" "${selected[@]}" -MMD -MF "$out/$profile/$source.d" -MT "$source.o" \
      -c "$PROJECT_ROOT/native/$source.c" -o "$object"
  done
  ar rcsD "$out/$profile/libeshkol_transformer_l3s.a" "$out/$profile/l3s_masked_objective_provider.o"
  ar rcsD "$out/$profile/libeshkol_transformer_e3_metrics.a" "$out/$profile/e3_evaluation_metrics_provider.o"
  for pair in cpu4:tests/e3_cpu4/native.c l3s:tests/l3s/test_l3s.c metrics:tests/e3_metrics/native_test.c numerical:tests/l3s/oracle_runner.c; do
    name=${pair%%:*};file=${pair#*:}
    log "link-$profile-$name" timeout --foreground --signal=TERM --kill-after=5s 60s \
      "$cc" "${flags[@]}" "${selected[@]}" "$PROJECT_ROOT/$file" "${objects[@]}" -lm -o "$out/$profile/$name"
    for repeat in 1 2; do
      log "run-$profile-$name-$repeat" env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 timeout --foreground --signal=TERM --kill-after=5s 60s "$out/$profile/$name"
      [[ ! -s "$out/run-$profile-$name-$repeat.stderr" ]] || die "runtime diagnostics from $profile/$name"
    done
    cmp "$out/run-$profile-$name-1.stdout" "$out/run-$profile-$name-2.stdout"
  done
done
for source in "${sources[@]}"; do cmp "$out/a/$source.o" "$out/b/$source.o";cmp "$out/a/$source.d" "$out/b/$source.d";done
for archive in libeshkol_transformer_l3s.a libeshkol_transformer_e3_metrics.a; do cmp "$out/a/$archive" "$out/b/$archive";done
for name in cpu4 l3s metrics numerical; do
  cmp "$out/a/$name" "$out/b/$name"
  cmp "$out/run-a-$name-1.stdout" "$out/run-b-$name-1.stdout"
done
for source in l3s_masked_objective_provider e3_evaluation_metrics_provider; do
  log "ir-$source" "$cc" "${flags[@]}" -O2 -S -emit-llvm "$PROJECT_ROOT/native/$source.c" -o "$out/$source.ll"
  if grep -E 'llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)|\b(fast|reassoc|afn|arcp)\b' "$out/$source.ll"; then die "CPU4 strict-f32 IR violation";fi
  nm -g --defined-only --format=posix "$out/a/$source.o" | awk '{print $1}' | LC_ALL=C sort >"$out/$source.exports"
  nm -u --format=posix "$out/a/$source.o" | awk '{print $1}' | LC_ALL=C sort -u >"$out/$source.undefined"
  objdump -d "$out/a/$source.o" >"$out/$source.assembly"
  if grep -E '\b(v?fmadd|v?fmsub|v?fnmadd|v?fnmsub)' "$out/$source.assembly";then die "CPU4 FMA instruction";fi
done
cmp native/l3s_public_exports.txt "$out/l3s_masked_objective_provider.exports"
cmp native/l3s_undefined_symbols.txt "$out/l3s_masked_objective_provider.undefined"
printf 'et_e3_metrics_kernel_provider_v1\n' | cmp - "$out/e3_evaluation_metrics_provider.exports"
cmp tests/e3_metrics/expected/undefined_symbols.txt "$out/e3_evaluation_metrics_provider.undefined"
log l3s-package python3 -B tests/l3s/test_package_contract.py "$out/a"
log metrics-package python3 -B tests/e3_metrics/test_package_contract.py "$out/a" "$out/e3_evaluation_metrics_provider.ll"
log capability-report-link "$cc" "${flags[@]}" -O2 tests/e3_metrics/report_provider.c \
  "$out/a/kernel_abi.o" "$out/a/e3_evaluation_metrics_provider.o" -lm -o "$out/report"
log capability-report "$out/report"
cmp tests/e3_metrics/expected/provider_report.json "$out/capability-report.stdout"
log mutations python3 -B tests/e3_cpu4/mutations.py --cc "$cc" --output "$out/mutations" --objects "$out/a" --reference-dir "$out"
printf 'CPU4-PROVIDERS-PASS: exact T2/T4 provider gate; model/evaluator integration remains pending\n'
