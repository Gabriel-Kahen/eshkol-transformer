#!/usr/bin/env bash
set -euo pipefail

t2_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${t2_root}"

t2_output="${1:-build/t2_trainer}"
t2_mode="${2:-normal}"

export CC="${CC:-clang-21}"
export CXX="${CXX:-clang++-21}"
t2_ar="${AR:-ar}"
t2_nm="${NM:-nm}"

t2_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
  -I include -I native
)

case "${t2_mode}" in
  normal)
    t2_flags+=(-O2)
    ;;
  sanitize)
    t2_flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
      "-ffile-prefix-map=${t2_root}=.")
    ;;
  *)
    printf 'unknown T2 trainer build mode: %s\n' "${t2_mode}" >&2
    exit 2
    ;;
esac

mkdir -p -- "${t2_output}"
t2_tmp="$(mktemp -d "${t2_output}/.t2-trainer-build.XXXXXX")"
trap 'rm -rf -- "${t2_tmp}"' EXIT

# C++17 header syntax check covering both public ABI and Eshkol AOT bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I include -I native \
  -fsyntax-only tests/t2_trainer/header_trainer_cpp.cpp

# Build deterministic core archive twice and verify byte-identity.
for trial in first second; do
  trial_dir="${t2_tmp}/${trial}"
  mkdir -- "${trial_dir}"
  "${CC}" "${t2_flags[@]}" -c native/t2_bpe_trainer.c \
    -o "${trial_dir}/t2_bpe_trainer.o"
  "${t2_ar}" rcsD "${trial_dir}/libeshkol_transformer_t2_trainer.a" \
    "${trial_dir}/t2_bpe_trainer.o"
  "${CC}" "${t2_flags[@]}" -c native/t2_bpe_trainer_eshkol_bridge.c \
    -o "${trial_dir}/t2_bpe_trainer_eshkol_bridge.o"
  "${t2_ar}" rcsD "${trial_dir}/libeshkol_transformer_t2_trainer_aot.a" \
    "${trial_dir}/t2_bpe_trainer_eshkol_bridge.o"
done

core_archive="${t2_tmp}/first/libeshkol_transformer_t2_trainer.a"
bridge_archive="${t2_tmp}/first/libeshkol_transformer_t2_trainer_aot.a"
cmp "${core_archive}" "${t2_tmp}/second/libeshkol_transformer_t2_trainer.a"
cmp "${bridge_archive}" "${t2_tmp}/second/libeshkol_transformer_t2_trainer_aot.a"

core_digest="$(sha256sum "${core_archive}")"
bridge_digest="$(sha256sum "${bridge_archive}")"
printf 'T2_TRAINER %s deterministic core archive SHA-256 %s\n' "${t2_mode}" "${core_digest%% *}"
printf 'T2_TRAINER %s deterministic bridge archive SHA-256 %s\n' "${t2_mode}" "${bridge_digest%% *}"

cat > "${t2_tmp}/expected-core-symbols" <<'SYMBOLS'
et_t2_bpe_decode
et_t2_bpe_deserialize
et_t2_bpe_destroy
et_t2_bpe_encode
et_t2_bpe_encode_ex
et_t2_bpe_get_info
et_t2_bpe_get_merge
et_t2_bpe_load_file
et_t2_bpe_save_file
et_t2_bpe_serialize
et_t2_bpe_token_bytes
et_t2_bpe_train
SYMBOLS

"${t2_nm}" -g --defined-only -P "${core_archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${t2_mode}" |
  LC_ALL=C sort > "${t2_tmp}/actual-core-symbols"
diff -u "${t2_tmp}/expected-core-symbols" "${t2_tmp}/actual-core-symbols"

cat > "${t2_tmp}/expected-bridge-symbols" <<'SYMBOLS'
et_t2_bpe_aot_bytes_equal
et_t2_bpe_aot_corpus_add_bytes
et_t2_bpe_aot_corpus_add_cstr
et_t2_bpe_aot_corpus_create
et_t2_bpe_aot_corpus_destroy
et_t2_bpe_aot_corpus_doc_count
et_t2_bpe_aot_count_tokens_cstr
et_t2_bpe_aot_cstr_bytes_equal
et_t2_bpe_aot_cstr_len
et_t2_bpe_aot_decode_bytes
et_t2_bpe_aot_destroy
et_t2_bpe_aot_encode_bytes
et_t2_bpe_aot_encode_cstr
et_t2_bpe_aot_get_merge
et_t2_bpe_aot_load
et_t2_bpe_aot_load_i64
et_t2_bpe_aot_load_u32
et_t2_bpe_aot_load_u8
et_t2_bpe_aot_merge_count
et_t2_bpe_aot_save
et_t2_bpe_aot_special_ids
et_t2_bpe_aot_store_u32
et_t2_bpe_aot_store_u8
et_t2_bpe_aot_train
et_t2_bpe_aot_train_bytes
et_t2_bpe_aot_vocab_size
SYMBOLS

"${t2_nm}" -g --defined-only -P "${bridge_archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${t2_mode}" |
  LC_ALL=C sort > "${t2_tmp}/actual-bridge-symbols"
diff -u "${t2_tmp}/expected-bridge-symbols" "${t2_tmp}/actual-bridge-symbols"

if [[ "${t2_mode}" == normal ]]; then
  core_imports="$("${t2_nm}" -u "${t2_tmp}/first/t2_bpe_trainer.o" | awk 'NF {print $NF}')"
  while IFS= read -r symbol; do
    case "${symbol}" in
      ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
      bcmp|calloc|close|free|fstat|fsync|lstat|malloc|memcmp|memcpy|memset|\
      mkstemp|open|read|rename|strrchr|unlink|write) ;;
      *)
        printf 'unexpected T2 trainer core dependency: %s\n' "${symbol}" >&2
        exit 1
        ;;
    esac
  done <<< "${core_imports}"

  bridge_imports="$("${t2_nm}" -u "${t2_tmp}/first/t2_bpe_trainer_eshkol_bridge.o" | awk 'NF {print $NF}')"
  while IFS= read -r symbol; do
    case "${symbol}" in
      ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
      bcmp|calloc|free|malloc|memcmp|memcpy|realloc|strlen|\
      et_t2_bpe_decode|et_t2_bpe_destroy|et_t2_bpe_encode|et_t2_bpe_get_info|\
      et_t2_bpe_get_merge|et_t2_bpe_load_file|et_t2_bpe_save_file|et_t2_bpe_train) ;;
      *)
        printf 'unexpected T2 trainer bridge dependency: %s\n' "${symbol}" >&2
        exit 1
        ;;
    esac
  done <<< "${bridge_imports}"
fi

mv -f -- "${core_archive}" "${t2_output}/libeshkol_transformer_t2_trainer.a"
mv -f -- "${bridge_archive}" "${t2_output}/libeshkol_transformer_t2_trainer_aot.a"
printf 'T2_TRAINER %s C11 build, C++17 header, and archive symbol audit: PASS\n' "${t2_mode}"
