#!/usr/bin/env bash
set -euo pipefail

e2e_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${e2e_root}"

e2e_output="${1:-${e2e_root}/build/e2e_lifecycle}"
e2e_mode="${2:-normal}"

export CC="${CC:-clang-21}"
export CXX="${CXX:-clang++-21}"
export LC_ALL=C
e2e_ar="${AR:-ar}"
e2e_nm="${NM:-nm}"

e2e_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard
  "-ffile-prefix-map=${e2e_root}=."
  -I "${e2e_root}/include" -I "${e2e_root}/native"
)

case "${e2e_mode}" in
  normal)
    e2e_cflags+=(-O2)
    ;;
  sanitize)
    e2e_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    ;;
  *)
    printf 'unknown E2E lifecycle build mode: %s\n' "${e2e_mode}" >&2
    exit 2
    ;;
esac

mkdir -p -- "${e2e_output}"
e2e_tmp="$(mktemp -d "${e2e_output}/.e2e-build.XXXXXX")"
trap 'rm -rf -- "${e2e_tmp}"' EXIT

# C++17 header syntax and layout check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${e2e_root}/include" -I "${e2e_root}/native" \
  -fsyntax-only "${e2e_root}/tests/e2e_lifecycle/header_e2e_cpp.cpp"

# Build deterministic archive twice from fresh directories and verify byte-identity.
for trial in first second; do
  tdir="${e2e_tmp}/${trial}"
  mkdir -- "${tdir}"
  "${CC}" "${e2e_cflags[@]}" \
    -c "${e2e_root}/native/e2e_lifecycle_eshkol_bridge.c" \
    -o "${tdir}/e2e_lifecycle_eshkol_bridge.o"
  "${e2e_ar}" rcsD "${tdir}/libeshkol_transformer_e2e_lifecycle.a" \
    "${tdir}/e2e_lifecycle_eshkol_bridge.o"
done

archive="${e2e_tmp}/first/libeshkol_transformer_e2e_lifecycle.a"
cmp "${archive}" "${e2e_tmp}/second/libeshkol_transformer_e2e_lifecycle.a"
archive_sha="$(sha256sum "${archive}" | awk '{print $1}')"
printf 'E2E_LIFECYCLE %s deterministic archive SHA-256: %s\n' "${e2e_mode}" "${archive_sha}"

# Exact defined-symbol audit.
cat > "${e2e_tmp}/expected-symbols" <<'SYMBOLS'
et_e2e_aot_create
et_e2e_aot_destroy
et_e2e_aot_get_f64
et_e2e_aot_get_i64
et_e2e_aot_get_metrics
et_e2e_aot_get_sha256_hex
et_e2e_aot_read_byte
et_e2e_aot_run_stage1_data4
et_e2e_aot_run_stage2_reload_encode
et_e2e_aot_run_stage2_train_save
et_e2e_aot_run_stage3_pretrain_part1
et_e2e_aot_run_stage3_pretrain_resume
et_e2e_aot_run_stage4_lora_reload
et_e2e_aot_run_stage4_lora_train_save
et_e2e_aot_run_stage5_dpo
et_e2e_aot_run_stage6_eval5
et_e2e_aot_run_stage6_verify_fresh
et_e2e_aot_run_stage7_spec_decode
et_e2e_aot_run_stage8_io5_quant
et_e2e_run_full_in_process
et_e2e_stage3_check_fd_gradients
SYMBOLS

"${e2e_nm}" -g --defined-only -P "${archive}" |
  awk '$2 ~ /^[A-Za-z]$/ && ($1 !~ /^__/ || mode == "normal") {print $1}' mode="${e2e_mode}" |
  LC_ALL=C sort > "${e2e_tmp}/actual-symbols"
diff -u "${e2e_tmp}/expected-symbols" "${e2e_tmp}/actual-symbols"

# Audit undefined imports in normal mode: confirm linkage against ALL 9 native subsystems.
if [[ "${e2e_mode}" == normal ]]; then
  imports="$("${e2e_nm}" -u "${e2e_tmp}/first/e2e_lifecycle_eshkol_bridge.o" | awk 'NF {print $NF}')"
  for required_prefix in et_data4_ et_t2_bpe_ et_mod5_adamw_ et_mod5_rope_ \
                         et_lora_ et_dpo_ et_eval5_ et_spec_ et_io5_; do
    if ! grep -Fq "${required_prefix}" <<< "${imports}"; then
      printf 'missing required native subsystem linkage: %s*\n' "${required_prefix}" >&2
      exit 1
    fi
  done
fi

mv -f -- "${archive}" "${e2e_output}/libeshkol_transformer_e2e_lifecycle.a"
printf 'E2E_LIFECYCLE %s C11 build, C++17 header, and symbol audit: PASS (%s)\n' \
  "${e2e_mode}" "${e2e_output}/libeshkol_transformer_e2e_lifecycle.a"
