#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in awk cat cmp git grep nm python3 sort timeout; do
  require_command "${command}"
done
python3 "${PROJECT_ROOT}/scripts/check-g3r-p1-roster.py"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
[[ "${cc}" == /* && -x "${cc}" && "${cxx}" == /* && -x "${cxx}" ]] || \
  die "provenance does not select executable absolute Clang paths"

baseline=90f3c42025556596a206b125138de49e09e3f7d3
[[ "$(git -C "${PROJECT_ROOT}" rev-parse "${baseline}^{tree}")" == \
   1fc04e8299610d5fe508d44c888f1f4b1c9a4021 ]] || \
  die "P1 predecessor tree differs"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/g3r-p1-roster.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT
git -C "${PROJECT_ROOT}" show "${baseline}:native/p1_identity.c" \
  >"${temporary}/p1_identity.c"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
       -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
       -I "${PROJECT_ROOT}/native")
for role in public trusted; do
  role_flags=()
  [[ "${role}" == trusted ]] && role_flags=(-DET_P1_TRUSTED_BUILD=1)
  "${cc}" "${flags[@]}" "${role_flags[@]}" -c \
    "${PROJECT_ROOT}/native/p1_identity.c" -o "${temporary}/current-${role}.o"
  "${cc}" "${flags[@]}" "${role_flags[@]}" -c \
    "${temporary}/p1_identity.c" -o "${temporary}/baseline-${role}.o"
  cmp "${temporary}/baseline-${role}.o" "${temporary}/current-${role}.o"
  nm -g --defined-only "${temporary}/current-${role}.o" | \
    awk '$2 ~ /^[A-Z]$/ {print $3}' | LC_ALL=C sort \
    >"${temporary}/${role}-symbols"
  cmp "${PROJECT_ROOT}/native/p1_identity_${role}_symbols.txt" \
    "${temporary}/${role}-symbols"
done

if "${cc}" "${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 -c \
    "${PROJECT_ROOT}/native/p1_identity.c" -o "${temporary}/forbidden-public.o" \
    >"${temporary}/forbidden-public.stdout" \
    2>"${temporary}/forbidden-public.stderr"; then
  die "candidate roster compiled in the public P1 role"
fi
grep -F 'requires the trusted P1 build' \
  "${temporary}/forbidden-public.stderr" >/dev/null

"${cc}" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 -c \
  "${PROJECT_ROOT}/native/p1_identity.c" -o "${temporary}/candidate.o"
nm -g --defined-only "${temporary}/candidate.o" | \
  awk '$2 ~ /^[A-Z]$/ {print $3}' | LC_ALL=C sort \
  >"${temporary}/candidate-symbols"
{
  cat "${PROJECT_ROOT}/native/p1_identity_trusted_symbols.txt"
  printf '%s\n' et_p1_private_candidate_construction_begin_v1 \
    et_p1_private_candidate_graph_preflight_v1 \
    et_p1_private_candidate_graph_revoke_v1
} | LC_ALL=C sort >"${temporary}/expected-candidate-symbols"
cmp "${temporary}/expected-candidate-symbols" "${temporary}/candidate-symbols"

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -DET_P1_CPP_TRUSTED_PROBE=1 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -I "${PROJECT_ROOT}/native" -fsyntax-only \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_header.cpp"

for run in a b; do
  "${cc}" "${flags[@]}" \
    "${PROJECT_ROOT}/tests/p1/test_g3r_candidate_roster.c" \
    -Wl,--wrap=calloc -o "${temporary}/roster-${run}"
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${temporary}/roster-${run}" >"${temporary}/run-${run}.stdout" \
    2>"${temporary}/run-${run}.stderr"
  [[ ! -s "${temporary}/run-${run}.stderr" ]] || die "native roster stderr is nonempty"
done
cmp "${temporary}/run-a.stdout" "${temporary}/run-b.stdout"
grep -Fx 'G3-R P1 candidate roster PASS: 19 retired tokens, exact native enrollment and retained-roster baseline' \
  "${temporary}/run-a.stdout" >/dev/null

"${cc}" "${flags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/p1/test_g3r_candidate_roster.c" \
  -Wl,--wrap=calloc -o "${temporary}/roster-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${temporary}/roster-sanitized" >"${temporary}/san.stdout" \
  2>"${temporary}/san.stderr"
cmp "${temporary}/run-a.stdout" "${temporary}/san.stdout"
[[ ! -s "${temporary}/san.stderr" ]] || die "sanitized native roster stderr is nonempty"
printf 'G3-R P1 ROSTER GATE PASS: normal/repeat/sanitizers, default-off byte/symbol parity\n'
