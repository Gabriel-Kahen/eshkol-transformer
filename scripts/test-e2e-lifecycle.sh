#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp diff nm readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E2E_LIFECYCLE_COMPILER_TIMEOUT_SECONDS:-180}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

e2e_dir="${PROJECT_ROOT}/build/e2e_lifecycle"
mkdir -p "${e2e_dir}"
scratch="$(mktemp -d "${e2e_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

# 1. Build all 9 existing Wave 4/5 native archives + K1 ABI + E2E lifecycle bridge.
bash scripts/build-data4-pipeline.sh "${PROJECT_ROOT}/build/data4_pipeline" normal
bash scripts/build-t2-trainer.sh "${PROJECT_ROOT}/build/t2_trainer" normal
bash scripts/build-mod5-adamw.sh
bash scripts/build-mod5-rope.sh
bash scripts/build-lora.sh "${PROJECT_ROOT}/build/lora"
bash scripts/build-dpo.sh "${PROJECT_ROOT}/build/dpo"
bash scripts/build-eval5.sh "${PROJECT_ROOT}/build/eval5" normal
bash scripts/build-spec-decode.sh "${PROJECT_ROOT}/build/spec_decode"
bash scripts/build-io5-quant.sh
bash scripts/build-k1.sh "${scratch}/k1" normal
bash scripts/build-e2e-lifecycle.sh "${e2e_dir}" normal

archives=(
  "${e2e_dir}/libeshkol_transformer_e2e_lifecycle.a"
  "${PROJECT_ROOT}/build/data4_pipeline/libeshkol_transformer_data4_pipeline.a"
  "${PROJECT_ROOT}/build/t2_trainer/libeshkol_transformer_t2_trainer.a"
  "${PROJECT_ROOT}/build/mod5_adamw/libeshkol_transformer_mod5_adamw.a"
  "${PROJECT_ROOT}/build/mod5_rope/libeshkol_transformer_mod5_rope.a"
  "${scratch}/k1/libeshkol_transformer_k1.a"
  "${PROJECT_ROOT}/build/lora/libeshkol_transformer_lora.a"
  "${PROJECT_ROOT}/build/dpo/libeshkol_transformer_dpo.a"
  "${PROJECT_ROOT}/build/eval5/libeshkol_eval5.a"
  "${PROJECT_ROOT}/build/spec_decode/libeshkol_transformer_spec_decode.a"
  "${PROJECT_ROOT}/build/io5_quant/libeshkol_transformer_io5_quant.a"
)

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard -frounding-math)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

# 2. C++17 header compilation and full archive linkage check.
mkdir -p "${scratch}/cpp-work"
"${CXX}" "${cppflags[@]}" -O2 tests/e2e_lifecycle/header_e2e_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_e2e_cpp"
"${scratch}/header_e2e_cpp" "${scratch}/cpp-work"
printf 'E2E_LIFECYCLE C++17 header and full 10-archive linkage: PASS\n'

# 3. Native C11 full pipeline + FD gradient + negative tests (normal -O2).
mkdir -p "${scratch}/nat-a" "${scratch}/nat-b" "${scratch}/nat-neg"
"${CC}" "${cflags[@]}" -O2 tests/e2e_lifecycle/test_e2e_lifecycle_native.c \
  "${archives[@]}" -lm -o "${scratch}/test_e2e_lifecycle_native"
"${scratch}/test_e2e_lifecycle_native" \
  "${scratch}/nat-a" "${scratch}/nat-b" "${scratch}/nat-neg"
printf 'E2E_LIFECYCLE native C11 normal (-O2) tests: PASS\n'

# 4. Native C11 + C++17 full pipeline & negative tests under ASan/UBSan/LSan.
san_sources=(
  native/e2e_lifecycle_eshkol_bridge.c
  native/data4_pipeline.c
  native/t2_bpe_trainer.c
  native/mod5_adamw_state.c
  native/mod5_rope_provider.c
  native/kernel_abi.c
  native/lora_adapter.c
  native/dpo_alignment.c
  native/eval5_benchmark.c
  native/spec_decode.c
  native/io5_quant_codec.c
)
san_objs=()
mkdir -p "${scratch}/san-objs" "${scratch}/san-a" "${scratch}/san-b" \
         "${scratch}/san-neg" "${scratch}/san-cpp"
for src in "${san_sources[@]}"; do
  base="$(basename -- "${src}" .c)"
  "${CC}" "${cflags[@]}" "${san[@]}" -c "${src}" \
    -o "${scratch}/san-objs/${base}.san.o"
  san_objs+=("${scratch}/san-objs/${base}.san.o")
done

"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/e2e_lifecycle/test_e2e_lifecycle_native.c \
  "${san_objs[@]}" -lm -o "${scratch}/test_e2e_lifecycle_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" \
  tests/e2e_lifecycle/header_e2e_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header_e2e_cpp.san"

export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${scratch}/test_e2e_lifecycle_native.san" \
  "${scratch}/san-a" "${scratch}/san-b" "${scratch}/san-neg"
"${scratch}/header_e2e_cpp.san" "${scratch}/san-cpp"
unset ASAN_OPTIONS UBSAN_OPTIONS
printf 'E2E_LIFECYCLE ASan/UBSan/LSan (detect_leaks=1) C11 + C++17 tests: PASS\n'

# 5. Dual fresh Eshkol AOT compilations (--strict-types --optimize 0 --no-stdlib)
#    verifying byte-identical ELFs.
for run in 1 2; do
  out_dir="${scratch}/aot-${run}"
  mkdir -p "${out_dir}/cache" "${out_dir}/lib"
  bash scripts/build-e2e-lifecycle.sh "${out_dir}/lib" normal >/dev/null
  # Stage static libeshkol_transformer_io5_quant.a without its sibling .so so
  # the compiled AOT ELF links statically with zero runtime .so dependency.
  cp "${PROJECT_ROOT}/build/io5_quant/libeshkol_transformer_io5_quant.a" \
     "${out_dir}/lib/"
  diagnostic="${out_dir}/compile.log"
  if ! (
    cd "${out_dir}"
    timeout --foreground --kill-after=5s "${compiler_timeout}s" \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${out_dir}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -L "${out_dir}/lib" \
        -L "${PROJECT_ROOT}/build/data4_pipeline" \
        -L "${PROJECT_ROOT}/build/t2_trainer" \
        -L "${PROJECT_ROOT}/build/mod5_adamw" \
        -L "${PROJECT_ROOT}/build/mod5_rope" \
        -L "${scratch}/k1" \
        -L "${PROJECT_ROOT}/build/lora" \
        -L "${PROJECT_ROOT}/build/dpo" \
        -L "${PROJECT_ROOT}/build/eval5" \
        -L "${PROJECT_ROOT}/build/spec_decode" \
        --lib eshkol_transformer_e2e_lifecycle \
        --lib eshkol_transformer_data4_pipeline \
        --lib eshkol_transformer_t2_trainer \
        --lib eshkol_transformer_mod5_adamw \
        --lib eshkol_transformer_mod5_rope \
        --lib eshkol_transformer_k1 \
        --lib eshkol_transformer_lora \
        --lib eshkol_transformer_dpo \
        --lib eshkol_eval5 \
        --lib eshkol_transformer_spec_decode \
        --lib eshkol_transformer_io5_quant \
        "${PROJECT_ROOT}/tests/e2e_lifecycle/test_e2e_llm_lifecycle.esk" \
        -o test_e2e_llm_lifecycle >"${diagnostic}" 2>&1
  ) || [[ ! -x "${out_dir}/test_e2e_llm_lifecycle" ]] || grep -Fq 'ERROR:' "${diagnostic}"; then
    die "E2E lifecycle Eshkol AOT compilation ${run} failed"
  fi
done
diagnostic=''

elf1="${scratch}/aot-1/test_e2e_llm_lifecycle"
elf2="${scratch}/aot-2/test_e2e_llm_lifecycle"
readelf -h "${elf1}" >/dev/null
if readelf -d "${elf1}" | grep -Fq 'libeshkol_transformer_'; then
  die "E2E AOT ELF unexpectedly links a project shared library dynamically"
fi
cmp "${elf1}" "${elf2}"
elf_sha="$(sha256sum "${elf1}" | awk '{print $1}')"
printf 'E2E_LIFECYCLE AOT dual fresh byte-identical ELF SHA-256: %s\n' "${elf_sha}"

# 6. Execute BOTH fresh compiled Eshkol AOT ELFs across 5 distinct OS processes
#    (multi-process save/reload boundaries for BPE, AdamW snapshot, LoRA ckpt,
#    and EVAL5 manifest) and verify byte-identical stdout!
for run in 1 2; do
  bin="${scratch}/aot-${run}/test_e2e_llm_lifecycle"
  work="${scratch}/run-${run}-work"
  mkdir -p "${work}"
  diagnostic="${scratch}/run-${run}.stderr"
  : >"${scratch}/run-${run}.stdout"
  : >"${diagnostic}"

  for phase in stage1-2 stage2-3a stage3b-4a stage4b-8 stage6-verify; do
    timeout --foreground --kill-after=5s 60s \
      "${bin}" --phase "${phase}" --work-dir "${work}" \
      >>"${scratch}/run-${run}.stdout" 2>>"${diagnostic}" || {
        cat "${scratch}/run-${run}.stdout" >&2
        die "E2E AOT binary failed on run ${run} phase ${phase}"
      }
  done
  [[ ! -s "${diagnostic}" ]] || die "E2E AOT run ${run} emitted stderr"

  # Verify fail-closed contaminated manifest was NOT written, and all genuine
  # artifacts exist and are non-executable.
  [[ ! -e "${work}/stage6_contaminated.manifest" ]] ||
    die "contaminated manifest unexpectedly exists on disk"
  for art in stage1_clean_corpus.bin stage2_tokenizer.etbpe stage3_train.ets \
             stage3_full.etsnap stage3_mid.etsnap stage3_resumed.etsnap \
             stage4_lora.etlr stage6_clean.manifest; do
    [[ -f "${work}/${art}" && ! -x "${work}/${art}" ]] ||
      die "missing or executable artifact: ${art}"
  done
  cmp "${work}/stage3_full.etsnap" "${work}/stage3_resumed.etsnap"
done
diagnostic=''

cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

# Validate every stage's emitted telemetry & cross-process SHA-256 consistency.
awk '
  function parse_kv(arr, start, nf,    i, p) {
    delete arr
    for (i = start; i <= nf; ++i) {
      p = index($i, "=")
      if (p <= 1) return 0
      arr[substr($i, 1, p - 1)] = substr($i, p + 1)
    }
    return 1
  }
  $1 == "E2E" && $2 == "STAGE1_DATA4" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["input"] != "11" || kv["kept"] != "6" ||
        kv["exact_dup"] != "1" || kv["near_dup"] != "1" ||
        kv["bad_utf8"] != "1" || kv["bad_len"] != "1" ||
        kv["bad_qual"] != "1") exit 1
    if (length(kv["report_sha256"]) != 64) exit 1
    s1++
    next
  }
  $1 == "E2E" && $2 == "STAGE2A_BPE_TRAIN" {
    if (!parse_kv(kv, 3, NF)) exit 1
    tok_sha_train = kv["tokenizer_sha256"]
    if (length(tok_sha_train) != 64 || kv["vocab"] + 0 <= 256 || kv["merges"] + 0 <= 0) exit 1
    s2a++
    next
  }
  $1 == "E2E" && $2 == "STAGE2B_BPE_RELOAD" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["tokenizer_sha256"] != tok_sha_train) exit 1
    if (kv["lossless"] != "1" || kv["compression"] + 0 <= 1.5) exit 1
    s2b++
    next
  }
  $1 == "E2E" && $2 == "STAGE3A_PRETRAIN" {
    if (!parse_kv(kv, 3, NF)) exit 1
    full_snap_sha = kv["full_snap_sha256"]
    if (length(full_snap_sha) != 64) exit 1
    if (!(kv["loss4"] + 0 < kv["loss1"] + 0 && kv["loss8"] + 0 < kv["loss4"] + 0)) exit 1
    s3a++
    next
  }
  $1 == "E2E" && $2 == "STAGE3B_RESUME" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["resumed_snap_sha256"] != full_snap_sha) exit 1
    if (kv["weights_bitwise_equal"] != "1" || kv["snapshot_sha_equal"] != "1" ||
        kv["corrupt_rejected"] != "1") exit 1
    s3b++
    next
  }
  $1 == "E2E" && $2 == "STAGE4A_LORA" {
    if (!parse_kv(kv, 3, NF)) exit 1
    lora_base_sha = kv["base_sha256"]
    lora_ckpt_sha = kv["ckpt_sha256"]
    if (kv["rank"] != "4" || kv["base_frozen"] != "1") exit 1
    if (!(kv["lossf"] + 0 < kv["loss0"] + 0)) exit 1
    if (!(kv["merge_err"] + 0 <= 1e-6 && kv["unmerge_err"] + 0 <= 1e-6)) exit 1
    s4a++
    next
  }
  $1 == "E2E" && $2 == "STAGE4B_LORA_RELOAD" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["reload_equal"] != "1" || kv["base_sha256"] != lora_base_sha ||
        kv["ckpt_sha256"] != lora_ckpt_sha) exit 1
    s4b++
    next
  }
  $1 == "E2E" && $2 == "STAGE5_DPO" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["ref_frozen"] != "1" || kv["acc"] + 0 != 1.0) exit 1
    if (!(kv["lossf"] + 0 < kv["loss0"] + 0 && kv["marginf"] + 0 > kv["margin0"] + 0)) exit 1
    s5++
    next
  }
  $1 == "E2E" && $2 == "STAGE6A_EVAL5" {
    if (!parse_kv(kv, 3, NF)) exit 1
    eval_manifest_sha = kv["manifest_sha256"]
    if (kv["contam_status"] != "-7" || kv["contam_manifest_absent"] != "1" ||
        kv["contam_docs"] != "1" || kv["clean_contam"] != "0" ||
        !(kv["lm_ppl"] + 0 > 1.0) || length(eval_manifest_sha) != 64) exit 1
    s6a++
    next
  }
  $1 == "E2E" && $2 == "STAGE7_SPEC" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["greedy_exact"] != "1" || kv["stoch_replay"] != "1" ||
        kv["eos_hit"] != "1" || kv["eos_emitted"] != "2") exit 1
    s7++
    next
  }
  $1 == "E2E" && $2 == "STAGE8_IO5" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (!(kv["f16_max_err"] + 0 <= 1e-3 && kv["bf16_max_err"] + 0 <= 1e-2 &&
          kv["q8_0_rel_l2"] + 0 <= 1e-2)) exit 1
    s8++
    next
  }
  $1 == "E2E" && $2 == "STAGE6B_EVAL5_FRESH" {
    if (!parse_kv(kv, 3, NF)) exit 1
    if (kv["verified"] != "1" || kv["clean_contam"] != "0" ||
        kv["manifest_sha256"] != eval_manifest_sha) exit 1
    s6b++
    next
  }
  { exit 1 }
  END {
    if (s1 != 1 || s2a != 1 || s2b != 1 || s3a != 1 || s3b != 1 ||
        s4a != 1 || s4b != 1 || s5 != 1 || s6a != 1 || s6b != 1 ||
        s7 != 1 || s8 != 1) exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid E2E lifecycle AOT output trace"
}

cat "${scratch}/run-1.stdout"
printf 'E2E_LIFECYCLE dual fresh AOT ELF + 5-process stdout byte-identical: PASS (ELF SHA-256 %s)\n' "${elf_sha}"
