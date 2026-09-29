#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp nm sha256sum timeout; do require_command "${command}"; done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-prompt-t1-borrow.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3c4-t1-i1-exact-pair-test}"
mkdir -p "${evidence}"
test ! -e "${evidence}/SHA256SUMS"
cp "${PROJECT_ROOT}/native/g3c4_t1_i1_exact_pair_source_closure.txt" \
  "${evidence}/source-closure.txt"
while IFS= read -r source; do
  printf '%s\t%s\n' \
    "$(sha256sum "${PROJECT_ROOT}/${source}" | awk '{print $1}')" "${source}"
done <"${PROJECT_ROOT}/native/g3c4_t1_i1_exact_pair_source_closure.txt" \
  >"${evidence}/source-sha256.tsv"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-t1-i1.XXXXXX")"
trap 'rm -rf -- "${tmp}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
       -fno-common -fstack-protector-all
       -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
       -DET_T1_I64_SHELL_TESTING -DET_I64_TENSOR_TESTING
       -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
sources=("${PROJECT_ROOT}/tests/g3c4/t1_i1_exact_pair.c"
         "${PROJECT_ROOT}/native/t1_i64_shell.c"
         "${PROJECT_ROOT}/native/i64_tensor.c"
         "${PROJECT_ROOT}/native/kernel_abi.c")

"${cc}" "${flags[@]}" -O2 "${sources[@]}" -o "${tmp}/normal"
"${tmp}/normal" >"${evidence}/native-normal.stdout"
"${tmp}/normal" >"${evidence}/native-repeat.stdout"
cmp "${evidence}/native-normal.stdout" "${evidence}/native-repeat.stdout"
grep -Fx 'G3-C4 T1/I1 EXACT PAIR PASS: 197 checks' \
  "${evidence}/native-normal.stdout" >/dev/null
"${cc}" "${flags[@]}" -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined "${sources[@]}" -o "${tmp}/sanitize"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${tmp}/sanitize" >"${evidence}/native-sanitize.stdout"
cmp "${evidence}/native-normal.stdout" "${evidence}/native-sanitize.stdout"

"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/t1_i64_shell.c" -o "${tmp}/t1-off.o"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/i64_tensor.c" -o "${tmp}/i1-off.o"
if nm -g --defined-only "${tmp}/t1-off.o" "${tmp}/i1-off.o" | \
   grep -E 'et_t1_i64_shell_private_c4_read_v1|et_i64_tensor_private_t1_pair_validate_v1|et_t1_i64_shell_test_(corrupt|restore)_v1'; then
  die "feature-off T1/I1 objects define exact-pair symbols"
fi
printf 'feature-off exact-pair symbols absent\n' \
  >"${evidence}/feature-off.stdout"

G3C4_EXACT_PAIR_PRIVATE=1 G3C4_SKIP_T1_EVAL_REGRESSION=1 \
  "${PROJECT_ROOT}/scripts/test-g3c4-prompt-t1-borrow.sh" \
  "${evidence}/linked" >"${evidence}/linked-runner.stdout" 2>"${evidence}/linked-runner.stderr"
test ! -s "${evidence}/linked-runner.stderr"
cmp "${evidence}/linked/normal-exact-pair-runtime.stdout" \
    "${evidence}/linked/sanitize-exact-pair-runtime.stdout"

printf 'private native normal/repeat/sanitizer and linked normal/sanitizer PASS\n' \
  >"${evidence}/result.stdout"
if git -C "${PROJECT_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  git -C "${PROJECT_ROOT}" rev-parse HEAD HEAD^{tree} \
    >"${evidence}/source-git.stdout"
  git -C "${PROJECT_ROOT}" diff --check
else
  declared_gitdir="$(sed -n 's/^gitdir: //p' "${PROJECT_ROOT}/.git" 2>/dev/null || true)"
  [[ "${declared_gitdir}" == /* && ! -e "${declared_gitdir}" ]] || \
    die "source Git failed for a reason other than inaccessible worktree metadata"
  printf 'git metadata inaccessible in runner; host check required\n' \
    >"${evidence}/source-git.stdout"
fi
(
  cd "${evidence}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)
printf 'G3-C4 T1/I1 exact-pair supported gate: PASS\n'
