#!/usr/bin/env bash
set -euo pipefail

eval5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${eval5_root}"
eval5_output="${1:-build/eval5_aot}"
eval5_mode="${2:-normal}"
eval5_cc="${CC:-cc}"
eval5_cxx="${CXX:-c++}"
eval5_ar="${AR:-ar}"
eval5_nm="${NM:-nm}"

eval5_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
  -I include -I native
)
case "${eval5_mode}" in
  normal)
    eval5_flags+=(-O2)
    ;;
  sanitize)
    eval5_flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
      "-ffile-prefix-map=${eval5_root}=.")
    ;;
  *)
    printf 'unknown EVAL5 AOT build mode: %s\n' "${eval5_mode}" >&2
    exit 2
    ;;
esac

mkdir -p -- "${eval5_output}"
eval5_tmp="$(mktemp -d "${eval5_output}/.eval5-aot-build.XXXXXX")"
trap 'rm -rf -- "${eval5_tmp}"' EXIT

"${eval5_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I include -I native \
  -fsyntax-only tests/eval5/header_bridge_cpp.cpp

for eval5_trial in first second; do
  eval5_trial_dir="${eval5_tmp}/${eval5_trial}"
  mkdir -- "${eval5_trial_dir}"
  "${eval5_cc}" "${eval5_flags[@]}" -c native/eval5_eshkol_bridge.c \
    -o "${eval5_trial_dir}/eval5_eshkol_bridge.o"
  "${eval5_ar}" rcsD "${eval5_trial_dir}/libeshkol_transformer_eval5_aot.a" \
    "${eval5_trial_dir}/eval5_eshkol_bridge.o"
done

eval5_archive="${eval5_tmp}/first/libeshkol_transformer_eval5_aot.a"
cmp "${eval5_archive}" "${eval5_tmp}/second/libeshkol_transformer_eval5_aot.a"
eval5_digest="$(sha256sum "${eval5_archive}")"
printf 'EVAL5 AOT %s deterministic archive SHA-256 %s\n' "${eval5_mode}" "${eval5_digest%% *}"

cat > "${eval5_tmp}/expected-symbols" <<'SYMBOLS'
et_eval5_aot_audit_contamination
et_eval5_aot_bytes_equal
et_eval5_aot_corpus_add_doc
et_eval5_aot_corpus_create
et_eval5_aot_corpus_destroy
et_eval5_aot_corpus_doc_count
et_eval5_aot_load_f64
et_eval5_aot_load_i64
et_eval5_aot_manifest_create
et_eval5_aot_manifest_destroy
et_eval5_aot_manifest_format
et_eval5_aot_manifest_get_digests
et_eval5_aot_manifest_get_metrics
et_eval5_aot_manifest_parse
et_eval5_aot_manifest_read_file
et_eval5_aot_manifest_set_audit
et_eval5_aot_manifest_set_digests
et_eval5_aot_manifest_set_identity
et_eval5_aot_manifest_set_lm
et_eval5_aot_manifest_set_mc
et_eval5_aot_manifest_verify_clean
et_eval5_aot_manifest_write_atomic
et_eval5_aot_mc_add_choice
et_eval5_aot_mc_begin_item
et_eval5_aot_mc_create
et_eval5_aot_mc_destroy
et_eval5_aot_score_lm
et_eval5_aot_score_mc
et_eval5_aot_sha256_bytes
et_eval5_aot_sha256_file
et_eval5_aot_store_f32
et_eval5_aot_store_u32
SYMBOLS

"${eval5_nm}" -g --defined-only -P "${eval5_archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${eval5_mode}" |
  LC_ALL=C sort > "${eval5_tmp}/actual-symbols"
diff -u "${eval5_tmp}/expected-symbols" "${eval5_tmp}/actual-symbols"

if [[ "${eval5_mode}" == normal ]]; then
  imports="$("${eval5_nm}" -u "${eval5_tmp}/first/eval5_eshkol_bridge.o" | awk 'NF {print $NF}')"
  while IFS= read -r symbol; do
    case "${symbol}" in
      ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
      bcmp|calloc|close|free|fstat|lstat|malloc|memcmp|memcpy|memset|open|read|realloc|strlen|\
      et_eval5_audit_contamination|et_eval5_manifest_format|et_eval5_manifest_parse|\
      et_eval5_manifest_read_file|et_eval5_manifest_write_atomic|\
      et_eval5_score_lm|et_eval5_score_mc) ;;
      *)
        printf 'unexpected EVAL5 AOT bridge dependency: %s\n' "${symbol}" >&2
        exit 1
        ;;
    esac
  done <<< "${imports}"
fi

mv -f -- "${eval5_archive}" "${eval5_output}/libeshkol_transformer_eval5_aot.a"
printf 'EVAL5 AOT %s C11 build, C++17 header, and archive symbol audit passed\n' "${eval5_mode}"
