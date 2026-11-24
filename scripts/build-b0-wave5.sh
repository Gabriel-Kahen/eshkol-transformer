#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

out_dir="${root}/build/b0_wave5"
mkdir -p "${out_dir}"

# 1. C++17 cleanliness check across all 11 Wave 5 public ABI headers + K1 ABI.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/mod5_rope_abi.h"
#include "eshkol_transformer/mod5_moe_abi.h"
#include "eshkol_transformer/mod5_mla_abi.h"
#include "eshkol_transformer/mod5_adamw_abi.h"
#include "eshkol_transformer/o2_muon_abi.h"
#include "eshkol_transformer/lora_adapter_abi.h"
#include "eshkol_transformer/dpo_alignment_abi.h"
#include "eshkol_transformer/spec_decode_abi.h"
#include "eshkol_transformer/t2_bpe_trainer_abi.h"
#include "eshkol_transformer/data4_pipeline_abi.h"
#include "eshkol_transformer/io5_quant_abi.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_mod5_mla_dims>);
static_assert(std::is_standard_layout_v<et_mod5_adamw_config_v1>);
static_assert(std::is_standard_layout_v<et_wsd_config>);
static_assert(std::is_standard_layout_v<et_lora_config>);
static_assert(std::is_standard_layout_v<et_dpo_config>);
static_assert(std::is_standard_layout_v<et_spec_sampling_params_v1>);
static_assert(std::is_standard_layout_v<et_t2_bpe_train_config>);
static_assert(std::is_standard_layout_v<et_data4_pipeline_config>);
static_assert(ET_IO5_QK8_0 == 32u);
CPP

# 2. Build K1 runtime archive + all 11 real Wave 5 static archives via repo scripts.
if [[ -r "${ESHKOL_BUILD_DIR:-${root}/.deps/eshkol-build}/eshkol-transformer-provenance.tsv" ]]; then
  bash "${root}/scripts/build-k1.sh" "${root}/build/k1" normal
else
  mkdir -p "${root}/build/k1"
  "${CC}" -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -ffp-contract=off \
    -fPIC -I "${root}/include" -c "${root}/native/kernel_abi.c" \
    -o "${root}/build/k1/kernel_abi.o"
  ar rcsD "${root}/build/k1/libeshkol_transformer_k1.a" \
    "${root}/build/k1/kernel_abi.o"
fi

CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-mod5-rope.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-mod5-moe.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-mod5-mla.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-mod5-adamw.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-o2-muon.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-lora.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-dpo.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-spec-decode.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-t2-trainer.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-data4-pipeline.sh"
CC="${CC}" CXX="${CXX}" bash "${root}/scripts/build-io5-quant.sh"

archives=(
  "${root}/build/mod5_rope/libeshkol_transformer_mod5_rope.a"
  "${root}/build/mod5_moe/libeshkol_transformer_mod5_moe.a"
  "${root}/build/mod5_mla/libeshkol_transformer_mod5_mla.a"
  "${root}/build/mod5_adamw/libeshkol_transformer_mod5_adamw.a"
  "${root}/build/o2_muon/libeshkol_transformer_o2_muon.a"
  "${root}/build/lora/libeshkol_transformer_lora.a"
  "${root}/build/dpo/libeshkol_transformer_dpo.a"
  "${root}/build/spec_decode/libeshkol_transformer_spec_decode.a"
  "${root}/build/t2_trainer/libeshkol_transformer_t2_trainer.a"
  "${root}/build/data4_pipeline/libeshkol_transformer_data4_pipeline.a"
  "${root}/build/io5_quant/libeshkol_transformer_io5_quant.a"
  "${root}/build/k1/libeshkol_transformer_k1.a"
)

for a in "${archives[@]}"; do
  if [[ ! -r "${a}" ]]; then
    printf 'missing required Wave 5 archive: %s\n' "${a}" >&2
    exit 1
  fi
done

# 3. Compile normal -O2 runner twice from fresh scratch dirs to verify determinism.
scratch="$(mktemp -d "${out_dir}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

normal_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all
  "-ffile-prefix-map=${root}=."
  -I "${root}/include"
)

for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p "${tdir}"
  "${CC}" "${normal_cflags[@]}" -c "${root}/native/b0_wave5_bench.c" \
    -o "${tdir}/b0_wave5_bench.o"
  "${CC}" -Wl,--build-id=none "${tdir}/b0_wave5_bench.o" "${archives[@]}" \
    -lm -o "${tdir}/b0_wave5_bench"
done

cmp "${scratch}/first/b0_wave5_bench.o" "${scratch}/second/b0_wave5_bench.o"
cmp "${scratch}/first/b0_wave5_bench" "${scratch}/second/b0_wave5_bench"
mv -f "${scratch}/first/b0_wave5_bench" "${out_dir}/b0_wave5_bench"

# 4. Compile ASan/UBSan/LSan sanitized runner + sanitized static archives.
san_root="${out_dir}/san_archives"
rm -rf -- "${san_root}"
san_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -O1 -g -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
  "-ffile-prefix-map=${root}=."
  -I "${root}/include" -I "${root}/native"
)

build_san_archive() {
  local subdir="$1"
  local libname="$2"
  local src="$3"
  local obj="${scratch}/san_${subdir}.o"
  mkdir -p "${san_root}/${subdir}"
  "${CC}" "${san_cflags[@]}" -c "${root}/native/${src}" -o "${obj}"
  ar rcsD "${san_root}/${subdir}/${libname}" "${obj}"
}

build_san_archive k1 libeshkol_transformer_k1.a kernel_abi.c
build_san_archive mod5_rope libeshkol_transformer_mod5_rope.a mod5_rope_provider.c
build_san_archive mod5_moe libeshkol_transformer_mod5_moe.a mod5_moe_provider.c
build_san_archive mod5_mla libeshkol_transformer_mod5_mla.a mod5_mla_provider.c
build_san_archive mod5_adamw libeshkol_transformer_mod5_adamw.a mod5_adamw_state.c
build_san_archive o2_muon libeshkol_transformer_o2_muon.a o2_muon_optimizer.c
build_san_archive lora libeshkol_transformer_lora.a lora_adapter.c
build_san_archive dpo libeshkol_transformer_dpo.a dpo_alignment.c
build_san_archive spec_decode libeshkol_transformer_spec_decode.a spec_decode.c
build_san_archive t2_trainer libeshkol_transformer_t2_trainer.a t2_bpe_trainer.c
build_san_archive data4_pipeline libeshkol_transformer_data4_pipeline.a data4_pipeline.c
build_san_archive io5_quant libeshkol_transformer_io5_quant.a io5_quant_codec.c

san_archives=(
  "${san_root}/mod5_rope/libeshkol_transformer_mod5_rope.a"
  "${san_root}/mod5_moe/libeshkol_transformer_mod5_moe.a"
  "${san_root}/mod5_mla/libeshkol_transformer_mod5_mla.a"
  "${san_root}/mod5_adamw/libeshkol_transformer_mod5_adamw.a"
  "${san_root}/o2_muon/libeshkol_transformer_o2_muon.a"
  "${san_root}/lora/libeshkol_transformer_lora.a"
  "${san_root}/dpo/libeshkol_transformer_dpo.a"
  "${san_root}/spec_decode/libeshkol_transformer_spec_decode.a"
  "${san_root}/t2_trainer/libeshkol_transformer_t2_trainer.a"
  "${san_root}/data4_pipeline/libeshkol_transformer_data4_pipeline.a"
  "${san_root}/io5_quant/libeshkol_transformer_io5_quant.a"
  "${san_root}/k1/libeshkol_transformer_k1.a"
)

"${CC}" "${san_cflags[@]}" "${root}/native/b0_wave5_bench.c" "${san_archives[@]}" \
  -lm -o "${out_dir}/b0_wave5_bench_san"

bin_sha="$(sha256sum "${out_dir}/b0_wave5_bench" | awk '{print $1}')"
printf 'B0 Wave 5 deterministic build (normal + sanitize + C++17 headers): PASS (sha256=%s)\n' "${bin_sha}"
