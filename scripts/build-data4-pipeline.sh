#!/usr/bin/env bash
set -euo pipefail

data4_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${data4_root}"

data4_output="${1:-build/data4_pipeline}"
data4_mode="${2:-normal}"

export CC="${CC:-clang-21}"
export CXX="${CXX:-clang++-21}"
data4_ar="${AR:-ar}"
data4_nm="${NM:-nm}"

data4_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
  -fstack-protector-all -fPIC
  -I "${data4_root}/include" -I "${data4_root}/native"
)

case "${data4_mode}" in
  normal)
    data4_cflags+=(-O2)
    ;;
  sanitize)
    data4_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
      "-ffile-prefix-map=${data4_root}=.")
    ;;
  *)
    printf 'unknown DATA4 pipeline build mode: %s\n' "${data4_mode}" >&2
    exit 2
    ;;
esac

mkdir -p -- "${data4_output}"
data4_tmp="$(mktemp -d "${data4_output}/.data4-build.XXXXXX")"
trap 'rm -rf -- "${data4_tmp}"' EXIT

# C++17 header syntax and layout check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${data4_root}/include" -I "${data4_root}/native" \
  -fsyntax-only "${data4_root}/tests/data4/header_pipeline_cpp.cpp"

# Build static archive twice in fresh directories and assert bit-exact parity.
for trial in first second; do
  trial_dir="${data4_tmp}/${trial}"
  mkdir -- "${trial_dir}"
  "${CC}" "${data4_cflags[@]}" -c "${data4_root}/native/data4_pipeline.c" \
    -o "${trial_dir}/data4_pipeline.o"
  "${CC}" "${data4_cflags[@]}" -c "${data4_root}/native/data4_pipeline_eshkol_bridge.c" \
    -o "${trial_dir}/data4_pipeline_eshkol_bridge.o"
  "${data4_ar}" rcsD "${trial_dir}/libeshkol_transformer_data4_pipeline.a" \
    "${trial_dir}/data4_pipeline.o" \
    "${trial_dir}/data4_pipeline_eshkol_bridge.o"
done

archive="${data4_tmp}/first/libeshkol_transformer_data4_pipeline.a"
cmp "${archive}" "${data4_tmp}/second/libeshkol_transformer_data4_pipeline.a"
archive_sha="$(sha256sum "${archive}" | awk '{print $1}')"
printf 'DATA4 pipeline %s deterministic archive SHA-256: %s\n' "${data4_mode}" "${archive_sha}"

# Exact defined-symbol audit.
cat > "${data4_tmp}/expected-symbols" <<'SYMBOLS'
et_data4_aot_add_doc
et_data4_aot_buf_load_u8
et_data4_aot_buf_store_u8
et_data4_aot_copy_report
et_data4_aot_copy_report_sha256
et_data4_aot_get_doc_disposition
et_data4_aot_get_doc_jaccard_den
et_data4_aot_get_doc_jaccard_num
et_data4_aot_get_doc_matched_ordinal
et_data4_aot_get_stat
et_data4_aot_run
et_data4_aot_session_create
et_data4_aot_session_destroy
et_data4_aot_set_limits
et_data4_aot_set_near_dedup
et_data4_aot_set_quality
et_data4_aot_verify_report
et_data4_compute_minhash
et_data4_config_default
et_data4_disposition_name
et_data4_exact_jaccard
et_data4_pipeline_run
et_data4_report_format
et_data4_report_verify
et_data4_sha256
et_data4_validate_utf8
SYMBOLS

"${data4_nm}" -g --defined-only -P "${archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${data4_mode}" |
  LC_ALL=C sort > "${data4_tmp}/actual-symbols"
diff -u "${data4_tmp}/expected-symbols" "${data4_tmp}/actual-symbols"

mv -f -- "${archive}" "${data4_output}/libeshkol_transformer_data4_pipeline.a"
printf 'DATA4 pipeline %s C11 build, C++17 header, and symbol audit: PASS (%s)\n' \
  "${data4_mode}" "${data4_output}/libeshkol_transformer_data4_pipeline.a"
