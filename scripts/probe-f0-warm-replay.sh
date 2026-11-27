#!/usr/bin/env bash
set -euo pipefail

# Offline comparison of the F0 fix against its immediate predecessor.
if [[ $# != 3 ]]; then
  printf 'usage: %s SEED_DEPS IMAGE_SHA256 NEW_RECEIPT_DIR\n' "${0##*/}" >&2
  exit 2
fi
seed="$(realpath -e -- "$1")"
image="$2"
receipt="$(realpath -m -- "$3")"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fixed=0d8bd51a7a3e23818d23a0edad1a7ae79be15f47
baseline="$(git -C "$repo" rev-parse "${fixed}^")"
scripts=(scripts/bootstrap-eshkol.sh scripts/common.sh)
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "$(docker image inspect --format '{{.Id}}' "$image")" == "$image" ]]
[[ -d "$seed/eshkol-src/.git" && -f "$seed/eshkol-build/.ninja_log" ]]
[[ "$receipt" != "$seed" && "$receipt" != "$seed/"* ]]
grep -Fxq 'CMAKE_HOME_DIRECTORY:INTERNAL=/workspace/.deps/eshkol-src' "$seed/eshkol-build/CMakeCache.txt"
grep -Fxq 'CMAKE_CACHEFILE_DIR:INTERNAL=/workspace/.deps/eshkol-build' "$seed/eshkol-build/CMakeCache.txt"
git -C "$repo" diff --exit-code "$baseline" "$fixed" -- Makefile toolchain/eshkol.lock

exec 9<>"$HOME/.cache/eshkol-transformer-orchestration-build.lock"
exec 8<>/tmp/eshkol-transformer-heavy-build.lock
flock -n 9 && flock -n 8 || { echo 'Shared Eshkol build slot is busy.' >&2; exit 1; }

mkdir -- "$receipt"
scratch="$(mktemp -d -- "$(dirname -- "$receipt")/.f0-warm-replay.XXXXXX")"
trap 'chmod -R u+rwX -- "$scratch"; rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/bin"

# The same observer delegates both phases to the real CMake without modifying
# the production scripts. Read-only observations bracket the bootstrap patch.
cat > "$scratch/bin/cmake" <<'OBSERVER'
#!/usr/bin/bash
set -euo pipefail
build=/workspace/.deps/eshkol-build
header="$build/generated/eshkol/build_config.h"
record() {
  sha256sum -- "$header"
  TZ=UTC stat -c '%y' -- "$header"
}
case "${1:-}" in
  -S)
    record > /receipt/initial-header.txt
    /usr/bin/time -f 'elapsed_seconds=%e' -o /receipt/configure.time /usr/bin/cmake "$@"
    record > /receipt/configured-header.txt
    ;;
  --build)
    record > /receipt/pre-build-header.txt
    (cd "$build" && sha256sum CMakeCache.txt build.ninja CMakeFiles/rules.ninja) > /receipt/configuration.sha256
    cp -- "$build/.ninja_log" /receipt/initial.ninja_log
    /usr/bin/time -f 'elapsed_seconds=%e' -o /receipt/build.time /usr/bin/cmake "$@"
    cp -- "$build/.ninja_log" /receipt/final.ninja_log
    record > /receipt/final-header.txt
    ;;
  *) exec /usr/bin/cmake "$@" ;;
esac
OBSERVER
chmod +x "$scratch/bin/cmake"
cp "$scratch/bin/cmake" "$receipt/cmake-observer.sh"
cp -- "${BASH_SOURCE[0]}" "$receipt/probe-f0-warm-replay.sh"

fingerprint() {
  # PAX retains nanosecond mtimes. Access and inode change times may differ on copies.
  LC_ALL=C tar --sort=name --format=pax \
    --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime \
    --numeric-owner -cf - -C "$1" eshkol-src eshkol-build | sha256sum | cut -d ' ' -f1
}

{
  printf 'seed\t%s\nimage\t%s\n' "$seed" "$image"
  printf 'host\t%s\n' "$(uname -srmo)"
  printf 'docker_context\t%s\n' "$(docker context show)"
  printf 'docker_server\t%s\n' "$(docker version --format '{{.Server.Version}}')"
} > "$receipt/host.tsv"
fingerprint "$seed" > "$receipt/initial-seed.sha256"

run() {
  local name="$1" revision="$2" copy="$scratch/$1" output="$receipt/$1"
  mkdir -p "$copy/deps" "$copy/workspace/.deps/"{eshkol-src,eshkol-build} "$output"
  cp -a --reflink=auto -- "$seed/eshkol-src" "$seed/eshkol-build" "$copy/deps/"
  fingerprint "$copy/deps" > "$output/initial-deps.sha256"
  cmp "$receipt/initial-seed.sha256" "$output/initial-deps.sha256"
  git -C "$repo" archive "$revision" Makefile toolchain/eshkol.lock "${scripts[@]}" |
    tar -xf - -C "$copy/workspace"
  printf '%s\n' "$revision" > "$output/bootstrap-commit.txt"
  (cd "$copy/workspace" && sha256sum Makefile toolchain/eshkol.lock "${scripts[@]}") > "$output/bootstrap-inputs.sha256"
  docker run --rm --pull=never --network=none --entrypoint /usr/bin/env \
    --mount "type=bind,source=$copy/workspace,target=/workspace,readonly" \
    --mount "type=bind,source=$copy/deps/eshkol-src,target=/workspace/.deps/eshkol-src,readonly" \
    --mount "type=bind,source=$copy/deps/eshkol-build,target=/workspace/.deps/eshkol-build" \
    --mount "type=bind,source=$scratch/bin,target=/probe/bin,readonly" \
    --mount "type=bind,source=$output,target=/receipt" \
    --workdir /workspace "$image" -i HOME=/root LC_ALL=C GIT_OPTIONAL_LOCKS=0 \
    PATH=/probe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    /usr/bin/bash -eu -o pipefail -c '
      git config --global --add safe.directory /workspace/.deps/eshkol-src
      /usr/bin/time -f "elapsed_seconds=%e\nmax_rss_kib=%M" -o /receipt/toolchain.time make toolchain
      source scripts/common.sh
      verify_eshkol_provenance
      cp .deps/eshkol-build/eshkol-transformer-provenance.tsv /receipt/provenance.tsv
      cp /etc/os-release /receipt/os-release
    ' > "$output/toolchain.log" 2>&1
  grep -Fq 'toolchain ready:' "$output/toolchain.log"
  # Ninja prints a [0/N] glob check even when it has no build work.
  awk '/^\[[1-9][0-9]*\/[0-9]+\] / { count++ } END { print count + 0 }' "$output/toolchain.log" > "$output/edges.txt"
}

run baseline "$baseline"
run fixed "$fixed"
fingerprint "$seed" > "$receipt/final-seed.sha256"
cmp "$receipt/initial-seed.sha256" "$receipt/final-seed.sha256"

before="$receipt/baseline"
after="$receipt/fixed"
cmp "$before/initial-header.txt" "$after/initial-header.txt"
cmp "$before/configuration.sha256" "$after/configuration.sha256"
cmp <(sed '/^eshkol_binary_sha256\t/d' "$before/provenance.tsv") \
    <(sed '/^eshkol_binary_sha256\t/d' "$after/provenance.tsv")
for run_dir in "$before" "$after"; do
  [[ "$(head -n 1 "$run_dir/initial-header.txt")" == "$(head -n 1 "$run_dir/configured-header.txt")" ]]
  [[ "$(tail -n 1 "$run_dir/configured-header.txt")" > "$(tail -n 1 "$run_dir/initial-header.txt")" ]]
  cmp "$run_dir/pre-build-header.txt" "$run_dir/final-header.txt"
done
cmp "$before/configured-header.txt" "$before/pre-build-header.txt"
cmp "$after/initial-header.txt" "$after/pre-build-header.txt"
[[ "$(cat "$before/edges.txt")" -gt 0 && "$(cat "$after/edges.txt")" -eq 0 ]]
ninja_build_log() {
  awk -F '\t' '$4 != "/workspace/.deps/eshkol-build/CMakeFiles/cmake.verify_globs"' "$1"
}
cmp "$before/initial.ninja_log" "$after/initial.ninja_log"
if cmp -s <(ninja_build_log "$before/initial.ninja_log") <(ninja_build_log "$before/final.ninja_log"); then
  echo 'Baseline Ninja log contains no new build work.' >&2
  exit 1
fi
cmp <(ninja_build_log "$after/initial.ninja_log") <(ninja_build_log "$after/final.ninja_log")
grep -Fxq 'ninja: no work to do.' "$after/toolchain.log"
printf 'baseline_edges=%s fixed_edges=0\n' "$(cat "$before/edges.txt")" > "$receipt/result.txt"
(cd "$receipt" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$receipt/SHA256SUMS"
printf 'PASS %s\n' "$receipt"
