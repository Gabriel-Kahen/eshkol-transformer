#!/usr/bin/env bash
set -euo pipefail

root=/home/gabe/Documents/ChatGPT/eshkol-pa4-wider-current-timing-20261125
evidence=/home/gabe/.codex/evidence/pa4-wider-current-timing-20261125
image=eshkol-r0-debug:20260928
cd "$root"
test "$(git rev-parse HEAD)" = 63a1d9f7a0fe473c7dff25ce82f63d2001c0062e
test -z "$(git status --porcelain --untracked-files=no)"

{
  date -u --iso-8601=seconds
  git rev-parse HEAD 'HEAD^{tree}'
  git status --short --branch
  git ls-tree HEAD native/pa4_wider_attention_provider.c tests/pa4_wider/bench_pa4_wider.c tests/pa4_wider/fixture.h scripts/build-pa4-wider.sh
  sha256sum native/pa4_wider_attention_provider.c tests/pa4_wider/bench_pa4_wider.c tests/pa4_wider/fixture.h scripts/build-pa4-wider.sh
  docker image inspect "$image" --format '{{.Id}} {{.RepoDigests}}'
  uname -a
  lscpu | grep -E 'Model name|CPU\(s\):|Thread\(s\) per core|Core\(s\) per socket'
  lscpu -e=CPU,CORE,ONLINE
  cat /proc/loadavg
  grep '^cpu6 \|^cpu14 ' /proc/stat
  docker run --rm --network none --cpuset-cpus 6 "$image" bash -ec 'cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID"; clang-21 --version | head -1; grep Cpus_allowed_list /proc/self/status'
} > "$evidence/conditions-before.txt"

docker run --rm --network none --cpuset-cpus 6 \
  -v "$root:$root" -w "$root" "$image" bash -ec \
  'CC=clang-21 bash scripts/build-pa4-wider.sh && clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off -I include -O2 tests/pa4_wider/bench_pa4_wider.c build/pa4_wider/libeshkol_transformer_pa4_wider.a native/kernel_abi.c -lm -o build/pa4_wider/bench-pa4-wider' \
  > "$evidence/build.log" 2>&1
sha256sum build/pa4_wider/libeshkol_transformer_pa4_wider.a build/pa4_wider/bench-pa4-wider > "$evidence/binary-sha256.txt"

{
  date -u --iso-8601=seconds
  cat /proc/loadavg
  grep '^cpu6 \|^cpu14 ' /proc/stat
  ps -eLo pid,tid,psr,pcpu,comm --no-headers | awk '$3==6 || $3==14 {print}' | sort -k3,3n
} > "$evidence/measurement-before.txt"
docker run --rm --network none --cpuset-cpus 6 \
  -v "$root:$root:ro" -w "$root" "$image" bash -ec \
  'grep Cpus_allowed_list /proc/self/status >&2; /usr/bin/time -v build/pa4_wider/bench-pa4-wider' \
  > "$evidence/benchmark-raw.log" 2> "$evidence/benchmark-stderr.log"
{
  date -u --iso-8601=seconds
  cat /proc/loadavg
  grep '^cpu6 \|^cpu14 ' /proc/stat
  ps -eLo pid,tid,psr,pcpu,comm --no-headers | awk '$3==6 || $3==14 {print}' | sort -k3,3n
} > "$evidence/measurement-after.txt"
