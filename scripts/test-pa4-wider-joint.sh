#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
if [[ $# -gt 1 || ($# -eq 1 && $1 != --time) ]]; then
  printf 'Usage: %s [--time]\n' "$0" >&2
  exit 2
fi
cc="$(realpath -- "$(command -v "${CC:-clang}")")"
out="${root}/build/pa4_wider"
mkdir -p "${out}"
scratch="$(mktemp -d "${out}/.joint.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-lto -I include)
sources=(tests/pa4_wider/joint_attention.c native/a2_attention_provider.c
         native/pa4_wider_attention_provider.c native/kernel_abi.c)
"${cc}" "${flags[@]}" -O2 "${sources[@]}" -lm -o "${scratch}/joint-attention"
if [[ $# -eq 1 ]]; then
  printf 'commit=%s tree=%s\n' "$(git rev-parse HEAD)" "$(git rev-parse 'HEAD^{tree}')"
  git status --short
  printf 'compiler=%s flags=%s -O2\n' "${cc}" "${flags[*]}"
  "${cc}" --version
  uname -srmo
  cat /etc/os-release
  awk -F ': *' '/^model name/ {print "cpu=" $2; exit}' /proc/cpuinfo
  taskset -pc "$$"
  sha256sum "${cc}" "${scratch}/joint-attention" "${sources[@]}" \
    include/eshkol_transformer/{a2_attention_abi,kernel_abi,pa4_wider_attention_abi}.h \
    tests/pa4_wider/fixture.h "${BASH_SOURCE[0]}"
  flock -n -E 75 /tmp/eshkol-transformer-heavy-build.lock \
    "${scratch}/joint-attention" --time
  exit
fi

"${scratch}/joint-attention"
"${cc}" "${flags[@]}" -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined -fno-sanitize-recover=all "${sources[@]}" \
  -lm -o "${scratch}/joint-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${scratch}/joint-sanitized"
printf 'PA4-Wider joint ASan/UBSan/LSan: PASS\n'
