#!/usr/bin/env bash
set -euo pipefail

ug_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ug_root}"

ug_output="${1:-build/unigram_tok}"
ug_mode="${2:-normal}"

export CC="${CC:-clang-21}"
export CXX="${CXX:-clang++-21}"
ug_ar="${AR:-ar}"
ug_nm="${NM:-nm}"

ug_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
  -I include -I native
)

case "${ug_mode}" in
  normal)
    ug_flags+=(-O2)
    ;;
  sanitize)
    ug_flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
      "-ffile-prefix-map=${ug_root}=.")
    ;;
  *)
    printf 'unknown unigram_tok build mode: %s\n' "${ug_mode}" >&2
    exit 2
    ;;
esac

mkdir -p -- "${ug_output}"
ug_tmp="$(mktemp -d "${ug_output}/.unigram-tok-build.XXXXXX")"
trap 'rm -rf -- "${ug_tmp}"' EXIT

# C++17 header syntax check covering both public ABI and Eshkol AOT bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I include -I native \
  -fsyntax-only tests/unigram_tok/header_unigram_cpp.cpp

# Build deterministic core and bridge archives twice and verify byte-identity.
for trial in first second; do
  trial_dir="${ug_tmp}/${trial}"
  mkdir -- "${trial_dir}"
  "${CC}" "${ug_flags[@]}" -c native/unigram_tok.c \
    -o "${trial_dir}/unigram_tok.o"
  "${ug_ar}" rcsD "${trial_dir}/libeshkol_transformer_unigram_tok.a" \
    "${trial_dir}/unigram_tok.o"
  "${CC}" "${ug_flags[@]}" -c native/unigram_tok_eshkol_bridge.c \
    -o "${trial_dir}/unigram_tok_eshkol_bridge.o"
  "${ug_ar}" rcsD "${trial_dir}/libeshkol_transformer_unigram_tok_aot.a" \
    "${trial_dir}/unigram_tok_eshkol_bridge.o"
done

core_archive="${ug_tmp}/first/libeshkol_transformer_unigram_tok.a"
bridge_archive="${ug_tmp}/first/libeshkol_transformer_unigram_tok_aot.a"
cmp "${core_archive}" "${ug_tmp}/second/libeshkol_transformer_unigram_tok.a"
cmp "${bridge_archive}" "${ug_tmp}/second/libeshkol_transformer_unigram_tok_aot.a"

core_digest="$(sha256sum "${core_archive}")"
bridge_digest="$(sha256sum "${bridge_archive}")"
printf 'UNIGRAM_TOK %s deterministic core archive SHA-256 %s\n' "${ug_mode}" "${core_digest%% *}"
printf 'UNIGRAM_TOK %s deterministic bridge archive SHA-256 %s\n' "${ug_mode}" "${bridge_digest%% *}"

cat > "${ug_tmp}/expected-core-symbols" <<'SYMBOLS'
et_unigram_corpus_fb_v1
et_unigram_create_v1
et_unigram_decode_v1
et_unigram_deserialize_v1
et_unigram_destroy_v1
et_unigram_encode_ex_v1
et_unigram_encode_v1
et_unigram_forward_backward_v1
et_unigram_get_info_v1
et_unigram_get_piece_v1
et_unigram_load_file_v1
et_unigram_sample_v1
et_unigram_save_file_v1
et_unigram_serialize_v1
et_unigram_train_v1
SYMBOLS

"${ug_nm}" -g --defined-only -P "${core_archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${ug_mode}" |
  LC_ALL=C sort > "${ug_tmp}/actual-core-symbols"
diff -u "${ug_tmp}/expected-core-symbols" "${ug_tmp}/actual-core-symbols"

cat > "${ug_tmp}/expected-bridge-symbols" <<'SYMBOLS'
et_unigram_aot_active_count
et_unigram_aot_bytes_equal
et_unigram_aot_corpus_add_bytes
et_unigram_aot_corpus_add_cstr
et_unigram_aot_corpus_create
et_unigram_aot_corpus_destroy
et_unigram_aot_corpus_doc_count
et_unigram_aot_cstr_bytes_equal
et_unigram_aot_cstr_len
et_unigram_aot_decode_bytes
et_unigram_aot_destroy
et_unigram_aot_encode_bytes
et_unigram_aot_encode_cstr
et_unigram_aot_f64_bits_equal
et_unigram_aot_forward_backward_bytes
et_unigram_aot_forward_backward_cstr
et_unigram_aot_learned_count
et_unigram_aot_load
et_unigram_aot_load_f64
et_unigram_aot_load_i64
et_unigram_aot_load_u32
et_unigram_aot_load_u8
et_unigram_aot_perturbed_log_z_cstr
et_unigram_aot_piece_log_prob
et_unigram_aot_probe_fp_env_trap
et_unigram_aot_probe_nonfinite_create
et_unigram_aot_sample_cstr
et_unigram_aot_save
et_unigram_aot_special_ids
et_unigram_aot_store_u32
et_unigram_aot_store_u8
et_unigram_aot_train
et_unigram_aot_u32_equal
et_unigram_aot_vocab_size
SYMBOLS

"${ug_nm}" -g --defined-only -P "${bridge_archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${ug_mode}" |
  LC_ALL=C sort > "${ug_tmp}/actual-bridge-symbols"
diff -u "${ug_tmp}/expected-bridge-symbols" "${ug_tmp}/actual-bridge-symbols"

if [[ "${ug_mode}" == normal ]]; then
  core_imports="$("${ug_nm}" -u "${ug_tmp}/first/unigram_tok.o" | awk 'NF {print $NF}')"
  while IFS= read -r symbol; do
    case "${symbol}" in
      ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
      bcmp|calloc|close|exp|fegetround|floor|free|fstat|fsync|log|log1p|\
      lstat|malloc|memcmp|memcpy|memset|mkstemp|open|qsort|read|rename|\
      strrchr|unlink|write) ;;
      *)
        printf 'unexpected unigram_tok core dependency: %s\n' "${symbol}" >&2
        exit 1
        ;;
    esac
  done <<< "${core_imports}"

  bridge_imports="$("${ug_nm}" -u "${ug_tmp}/first/unigram_tok_eshkol_bridge.o" | awk 'NF {print $NF}')"
  while IFS= read -r symbol; do
    case "${symbol}" in
      ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
      bcmp|calloc|fegetround|fesetround|free|malloc|memcmp|memcpy|memset|\
      realloc|strlen|\
      et_unigram_create_v1|et_unigram_decode_v1|et_unigram_destroy_v1|\
      et_unigram_encode_v1|et_unigram_forward_backward_v1|\
      et_unigram_get_info_v1|et_unigram_get_piece_v1|\
      et_unigram_load_file_v1|et_unigram_sample_v1|et_unigram_save_file_v1|\
      et_unigram_train_v1) ;;
      *)
        printf 'unexpected unigram_tok bridge dependency: %s\n' "${symbol}" >&2
        exit 1
        ;;
    esac
  done <<< "${bridge_imports}"
fi

mv -f -- "${core_archive}" "${ug_output}/libeshkol_transformer_unigram_tok.a"
mv -f -- "${bridge_archive}" "${ug_output}/libeshkol_transformer_unigram_tok_aot.a"
printf 'UNIGRAM_TOK %s C11 build, C++17 header, and archive symbol audit: PASS\n' "${ug_mode}"
