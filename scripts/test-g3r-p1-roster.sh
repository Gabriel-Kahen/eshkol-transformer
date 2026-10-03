#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(( $# <= 1 )) || die "usage: $0 [fresh-absolute-evidence-directory]"
if (( $# == 1 )); then
  temporary=$1
  [[ "${temporary}" == /* && ! -e "${temporary}" && ! -L "${temporary}" ]] || \
    die "evidence directory must be a fresh absolute path"
  mkdir -- "${temporary}"
else
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/g3r-p1-roster.XXXXXX")"
  trap 'rm -rf -- "${temporary}"' EXIT
fi

verify_toolchain >"${temporary}/toolchain.stdout" 2>"${temporary}/toolchain.stderr"
for command in awk cat cmp git grep nm python3 sort timeout; do
  require_command "${command}"
done
python3 "${PROJECT_ROOT}/scripts/check-g3r-p1-roster.py" \
  >"${temporary}/static.stdout" 2>"${temporary}/static.stderr"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
[[ "${cc}" == /* && -x "${cc}" && "${cxx}" == /* && -x "${cxx}" ]] || \
  die "provenance does not select executable absolute Clang paths"
cat "${provenance}" >"${temporary}/compiler-provenance.tsv"
printf 'cc\t%s\ncxx\t%s\n' "${cc}" "${cxx}" \
  >"${temporary}/compiler-selected.tsv"
"${cc}" --version >"${temporary}/cc.version" 2>"${temporary}/cc.version.stderr"
"${cxx}" --version >"${temporary}/cxx.version" 2>"${temporary}/cxx.version.stderr"

baseline=90f3c42025556596a206b125138de49e09e3f7d3
[[ "$(git -C "${PROJECT_ROOT}" rev-parse "${baseline}^{tree}")" == \
   1fc04e8299610d5fe508d44c888f1f4b1c9a4021 ]] || \
  die "P1 predecessor tree differs"
git -C "${PROJECT_ROOT}" show "${baseline}:native/p1_identity.c" \
  >"${temporary}/p1_identity.c"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
       -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
       -I "${PROJECT_ROOT}/native")
for role in public trusted; do
  role_flags=()
  [[ "${role}" == trusted ]] && role_flags=(-DET_P1_TRUSTED_BUILD=1)
  "${cc}" "${flags[@]}" "${role_flags[@]}" -c \
    "${PROJECT_ROOT}/native/p1_identity.c" -o "${temporary}/current-${role}.o" \
    >"${temporary}/current-${role}.compile.stdout" \
    2>"${temporary}/current-${role}.compile.stderr"
  "${cc}" "${flags[@]}" "${role_flags[@]}" -c \
    "${temporary}/p1_identity.c" -o "${temporary}/baseline-${role}.o" \
    >"${temporary}/baseline-${role}.compile.stdout" \
    2>"${temporary}/baseline-${role}.compile.stderr"
  cmp "${temporary}/baseline-${role}.o" "${temporary}/current-${role}.o" \
    >"${temporary}/${role}.object-cmp.stdout" \
    2>"${temporary}/${role}.object-cmp.stderr"
  nm -g --defined-only "${temporary}/current-${role}.o" | \
    awk '$2 ~ /^[A-Z]$/ {print $3}' | LC_ALL=C sort \
    >"${temporary}/${role}-symbols"
  cmp "${PROJECT_ROOT}/native/p1_identity_${role}_symbols.txt" \
    "${temporary}/${role}-symbols" \
    >"${temporary}/${role}.symbol-cmp.stdout" \
    2>"${temporary}/${role}.symbol-cmp.stderr"
done

if "${cc}" "${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 -c \
    "${PROJECT_ROOT}/native/p1_identity.c" -o "${temporary}/forbidden-public.o" \
    >"${temporary}/forbidden-public.stdout" \
    2>"${temporary}/forbidden-public.stderr"; then
  printf '0\n' >"${temporary}/forbidden-public.exit"
  die "candidate roster compiled in the public P1 role"
else
  printf '%d\n' "$?" >"${temporary}/forbidden-public.exit"
fi
grep -F 'requires the trusted P1 build' \
  "${temporary}/forbidden-public.stderr" >/dev/null

"${cc}" "${flags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 -c \
  "${PROJECT_ROOT}/native/p1_identity.c" -o "${temporary}/candidate.o" \
  >"${temporary}/candidate.compile.stdout" \
  2>"${temporary}/candidate.compile.stderr"
nm -g --defined-only "${temporary}/candidate.o" | \
  awk '$2 ~ /^[A-Z]$/ {print $3}' | LC_ALL=C sort \
  >"${temporary}/candidate-symbols"
{
  cat "${PROJECT_ROOT}/native/p1_identity_trusted_symbols.txt"
  printf '%s\n' et_p1_private_candidate_construction_begin_v1 \
    et_p1_private_candidate_graph_preflight_v1 \
    et_p1_private_candidate_graph_revoke_v1
} | LC_ALL=C sort >"${temporary}/expected-candidate-symbols"
cmp "${temporary}/expected-candidate-symbols" "${temporary}/candidate-symbols" \
  >"${temporary}/candidate.symbol-cmp.stdout" \
  2>"${temporary}/candidate.symbol-cmp.stderr"

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -DET_P1_CPP_TRUSTED_PROBE=1 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -I "${PROJECT_ROOT}/native" -fsyntax-only \
  "${PROJECT_ROOT}/tests/p1/test_p1_identity_header.cpp" \
  >"${temporary}/cpp-header.stdout" 2>"${temporary}/cpp-header.stderr"

for run in a b; do
  "${cc}" "${flags[@]}" \
    "${PROJECT_ROOT}/tests/p1/test_g3r_candidate_roster.c" \
    -Wl,--wrap=calloc -o "${temporary}/roster-${run}" \
    >"${temporary}/roster-${run}.compile.stdout" \
    2>"${temporary}/roster-${run}.compile.stderr"
  set +e
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${temporary}/roster-${run}" >"${temporary}/run-${run}.stdout" \
    2>"${temporary}/run-${run}.stderr"
  status=$?
  set -e
  printf '%d\n' "${status}" >"${temporary}/run-${run}.exit"
  [[ "${status}" -eq 0 ]] || die "native roster ${run} exited ${status}"
  [[ ! -s "${temporary}/run-${run}.stderr" ]] || die "native roster stderr is nonempty"
done
cmp "${temporary}/run-a.stdout" "${temporary}/run-b.stdout" \
  >"${temporary}/repeat-cmp.stdout" 2>"${temporary}/repeat-cmp.stderr"
grep -Fx 'G3-R P1 candidate roster PASS: 19 retired tokens, exact native enrollment and retained-roster baseline' \
  "${temporary}/run-a.stdout" >/dev/null

"${cc}" "${flags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/p1/test_g3r_candidate_roster.c" \
  -Wl,--wrap=calloc -o "${temporary}/roster-sanitized" \
  >"${temporary}/roster-sanitized.compile.stdout" \
  2>"${temporary}/roster-sanitized.compile.stderr"
set +e
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${temporary}/roster-sanitized" >"${temporary}/san.stdout" \
  2>"${temporary}/san.stderr"
status=$?
set -e
printf '%d\n' "${status}" >"${temporary}/san.exit"
[[ "${status}" -eq 0 ]] || die "sanitized native roster exited ${status}"
cmp "${temporary}/run-a.stdout" "${temporary}/san.stdout" \
  >"${temporary}/san-cmp.stdout" 2>"${temporary}/san-cmp.stderr"
[[ ! -s "${temporary}/san.stderr" ]] || die "sanitized native roster stderr is nonempty"
printf 'G3-R P1 ROSTER GATE PASS: normal/repeat/sanitizers, default-off byte/symbol parity\n'
