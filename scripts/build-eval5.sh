#!/usr/bin/env bash
set -euo pipefail

eval5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${eval5_root}"
eval5_output="${1:-build/eval5}"
eval5_mode="${2:-normal}"
eval5_cc="${CC:-cc}"
eval5_cxx="${CXX:-c++}"
eval5_ar="${AR:-ar}"
eval5_nm="${NM:-nm}"
eval5_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -I include
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
)
case "${eval5_mode}" in
  normal) eval5_flags+=(-O2) ;;
  sanitize)
    eval5_flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
      "-ffile-prefix-map=${eval5_root}=.")
    ;;
  *) printf 'unknown EVAL5 build mode: %s\n' "${eval5_mode}" >&2; exit 2 ;;
esac

mkdir -p -- "${eval5_output}"
eval5_tmp="$(mktemp -d "${eval5_output}/.eval5-build.XXXXXX")"
trap 'rm -rf -- "${eval5_tmp}"' EXIT

"${eval5_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I include \
  -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/eval5_benchmark_abi.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_eval5_manifest>);
static_assert(std::is_same_v<et_eval5_status, int32_t>);
CPP

for eval5_trial in first second; do
  eval5_trial_dir="${eval5_tmp}/${eval5_trial}"
  mkdir -- "${eval5_trial_dir}"
  "${eval5_cc}" "${eval5_flags[@]}" -c native/eval5_benchmark.c \
    -o "${eval5_trial_dir}/eval5_benchmark.o"
  "${eval5_ar}" rcsD "${eval5_trial_dir}/libeshkol_eval5.a" \
    "${eval5_trial_dir}/eval5_benchmark.o"
done
eval5_archive="${eval5_tmp}/first/libeshkol_eval5.a"
cmp "${eval5_archive}" "${eval5_tmp}/second/libeshkol_eval5.a"
eval5_digest="$(sha256sum "${eval5_archive}")"
printf 'EVAL5 %s deterministic archive SHA-256 %s\n' "${eval5_mode}" "${eval5_digest%% *}"

cat > "${eval5_tmp}/expected-symbols" <<'SYMBOLS'
et_eval5_audit_contamination
et_eval5_manifest_format
et_eval5_manifest_parse
et_eval5_manifest_read_file
et_eval5_manifest_write_atomic
et_eval5_score_lm
et_eval5_score_mc
SYMBOLS
"${eval5_nm}" -g --defined-only -P "${eval5_archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${eval5_mode}" |
  LC_ALL=C sort > "${eval5_tmp}/actual-symbols"
diff -u "${eval5_tmp}/expected-symbols" "${eval5_tmp}/actual-symbols"
mv -f -- "${eval5_archive}" "${eval5_output}/libeshkol_eval5.a"
printf 'EVAL5 %s C11 build, C++17 header, and archive symbol audit passed\n' "${eval5_mode}"
