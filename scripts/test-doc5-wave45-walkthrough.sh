#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

unset MAKEFLAGS MFLAGS GNUMAKEFLAGS
# Use the checkout selected by each Git command and isolate the scratch index.
git_local_env="$(git rev-parse --local-env-vars)"
while read -r name; do
  unset "${name}"
done <<< "${git_local_env}"

cd "${PROJECT_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
export ESHKOL_SOURCE_DIR ESHKOL_BUILD_DIR
ESHKOL_SOURCE_DIR="$(realpath -m -- "$(eshkol_source_dir)")"
ESHKOL_BUILD_DIR="$(realpath -m -- "$(eshkol_build_dir)")"
verify_toolchain
printf 'DOC5 Wave 4/5 toolchain: Eshkol %s, LLVM %s\n' \
  "$(lock_value eshkol_commit)" "$("$(llvm_config)" --version)"

scratch="$(mktemp -d /tmp/eshkol-doc5-wave45.XXXXXX)"
trap 'rm -rf -- "${scratch}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export TMPDIR="${scratch}/tmp" XDG_CACHE_HOME="${scratch}/cache"
mkdir -p "${scratch}/repo" "${TMPDIR}" "${XDG_CACHE_HOME}"

# Run all checks and live builds in a disposable copy of the repository so no
# generated build/, .tmp/, or cache files pollute the caller's checkout.
git ls-files -z --deduplicate | tar --null -T - -cf - | tar -xf - -C "${scratch}/repo"
cp docs/DOC5_{API_REFERENCE,CORPUS_TO_MODEL_TUTORIAL,WAVE45_GUIDE}.md \
  "${scratch}/repo/docs/"
cp scripts/test-doc5-wave45-walkthrough.sh "${scratch}/repo/scripts/"
chmod +x "${scratch}/repo/scripts/test-doc5-wave45-walkthrough.sh"
cd "${scratch}/repo"
git init --quiet --template=
git add -f -- .
export BUILD_DIR="${PWD}/build"
unset ESHKOL_PATH ESHKOL_JIT_CACHE_DIR ESHKOL_AOT_MODULE_CACHE_DIR ESHKOL_LIB_DIR
export ESHKOL_JIT_CACHE=0

require_make_target() {
  local target="$1" context="$2"
  awk -F: -v target="${target}" '
    /^[^\t#][^:]*:/ && $0 !~ /^[^:]*:+=/ {
      count = split($1, names, /[[:blank:]]+/)
      for (i = 1; i <= count; i++)
        if (names[i] == target) found = 1
    }
    END { exit !found }
  ' Makefile || die "${context}: missing Makefile target: ${target}"
}

# 1. Verify every row in docs/DOC5_API_REFERENCE.md (source path, exact symbols,
#    and Makefile/script gates).
span='`([^`]+)`'
rows=0
while IFS='|' read -r left subsystem source_cell references gates scope right; do
  [[ -z "${left}" && "${source_cell}" =~ ${span} ]] || continue
  source_file="${BASH_REMATCH[1]}"
  [[ -f "${source_file}" ]] || die "DOC5 Wave 4/5 missing source: ${source_file}"
  while [[ "${references}" =~ ${span} ]]; do
    symbol="${BASH_REMATCH[1]}"
    references="${references#*"${BASH_REMATCH[0]}"}"
    awk -v needle="${symbol}" '
      {
        rest = $0
        while ((pos = index(rest, needle)) > 0) {
          before = pos > 1 ? substr(rest, pos - 1, 1) : ""
          after = substr(rest, pos + length(needle), 1)
          if (before !~ /[[:alnum:]_?!-]/ && after !~ /[[:alnum:]_?!-]/) {
            found = 1
            exit
          }
          rest = substr(rest, pos + 1)
        }
      }
      END { exit !found }
    ' "${source_file}" ||
      die "DOC5 Wave 4/5 missing reference '${symbol}' in ${source_file}"
  done
  while [[ "${gates}" =~ ${span} ]]; do
    gate="${BASH_REMATCH[1]}"
    gates="${gates#*"${BASH_REMATCH[0]}"}"
    case "${gate}" in
      'make '*)
        require_make_target "${gate#make }" 'DOC5 API reference'
        ;;
      'bash scripts/'*)
        [[ -f "${gate#bash }" ]] || die "DOC5 Wave 4/5 missing script: ${gate#bash }"
        ;;
      *) die "DOC5 Wave 4/5 unrecognized gate: ${gate}" ;;
    esac
  done
  rows=$((rows + 1))
done < docs/DOC5_API_REFERENCE.md
(( rows > 0 )) || die 'DOC5 API table is empty'
printf 'DOC5 API reference table: verified %s source rows\n' "${rows}"

# 2. Verify every header path, doc link, script path, and make target cited in
#    DOC5_API_REFERENCE.md, DOC5_CORPUS_TO_MODEL_TUTORIAL.md, and DOC5_WAVE45_GUIDE.md.
verify_doc_references() {
  local doc="$1"
  local ref_file="${scratch}/$(basename -- "${doc}").refs"
  local ref_count=0
  awk '
    function make_references(code,   target) {
      while (match(code, /(^|[[:space:];(&|])make[[:blank:]]+["\047]?[[:alnum:]_.][[:alnum:]_.-]*/)) {
        target = substr(code, RSTART, RLENGTH)
        code = substr(code, RSTART + RLENGTH)
        sub(/^.*make[[:blank:]]+["\047]?/, "", target)
        print FNR, "target", target
      }
    }
    /^[[:blank:]]*(```+|~~~+)/ {
      marker = substr($0, match($0, /[`~]/), 1)
      if (fence == marker) fence = ""
      else if (!fence) fence = marker
      next
    }
    {
      if (fence) make_references($0)
      else {
        count = split($0, spans, /`+/)
        for (i = 2; i < count; i += 2) make_references(spans[i])
      }
      line = $0
      while (match(line, /(^|[^[:alnum:]_./$-])(\.\.?\/)?(docs|include|native|scripts|tests|tools|src|internal|benchmarks|lib|toolchain)\/[[:alnum:]_./+-]+/)) {
        path = substr(line, RSTART, RLENGTH)
        line = substr(line, RSTART + RLENGTH)
        sub(/^[^[:alnum:]_.]/, "", path)
        sub(/^\.\.?\//, "", path)
        sub(/\.$/, "", path)
        print FNR, "path", path
      }
      if (!fence) {
        line = $0
        while (match(line, /\]\([[:alnum:]_][[:alnum:]_.-]*\.md[)#]/)) {
          print FNR, "path", "docs/" substr(line, RSTART + 2, RLENGTH - 3)
          line = substr(line, RSTART + RLENGTH)
        }
      }
    }
  ' "${doc}" > "${ref_file}"
  while read -r line kind reference; do
    local context="${doc}:${line}"
    case "${kind}" in
      target) require_make_target "${reference}" "${context}" ;;
      path) [[ -e "${reference}" ]] || die "${context}: missing path: ${reference}" ;;
    esac
    ref_count=$((ref_count + 1))
  done < "${ref_file}"
  (( ref_count > 0 )) || die "${doc}: no verified references found"
  printf 'DOC5 reference check (%s): verified %s paths and make targets\n' \
    "${doc}" "${ref_count}"
}

for doc in \
  docs/DOC5_API_REFERENCE.md \
  docs/DOC5_CORPUS_TO_MODEL_TUTORIAL.md \
  docs/DOC5_WAVE45_GUIDE.md; do
  verify_doc_references "${doc}"
done

# 3. Verify complete inventory coverage of all 57 public ABI headers in
#    include/eshkol_transformer/ (13 Wave 0–3 + 44 Wave 4/5), ensuring every
#    Wave 4/5 header appears in both DOC5_API_REFERENCE.md and
#    DOC5_WAVE45_GUIDE.md and every native/*_eshkol_bridge.h bridge appears in
#    DOC5_WAVE45_GUIDE.md.
wave03_headers=(
  include/eshkol_transformer/a2_attention_abi.h
  include/eshkol_transformer/a2_kv_cache.h
  include/eshkol_transformer/e3_evaluation_metrics_abi.h
  include/eshkol_transformer/f32_tensor.h
  include/eshkol_transformer/g3c4_primitives_abi.h
  include/eshkol_transformer/g3n_primitives_abi.h
  include/eshkol_transformer/g3s_sampling_abi.h
  include/eshkol_transformer/i64_tensor.h
  include/eshkol_transformer/indexed_cross_entropy.h
  include/eshkol_transformer/kernel_abi.h
  include/eshkol_transformer/l3s_masked_objective_abi.h
  include/eshkol_transformer/n2_primitives_abi.h
  include/eshkol_transformer/n3k_primitives_abi.h
)

wave45_headers=(
  include/eshkol_transformer/ac4_checkpointing_abi.h
  include/eshkol_transformer/amd4_hip_abi.h
  include/eshkol_transformer/beam_search_abi.h
  include/eshkol_transformer/constrained_decode_abi.h
  include/eshkol_transformer/continuous_batch_abi.h
  include/eshkol_transformer/data4_pipeline_abi.h
  include/eshkol_transformer/dist5_collectives_abi.h
  include/eshkol_transformer/dpo_alignment_abi.h
  include/eshkol_transformer/e2e_lifecycle_abi.h
  include/eshkol_transformer/eval5_benchmark_abi.h
  include/eshkol_transformer/flash_attn2_abi.h
  include/eshkol_transformer/fp8_gemm_abi.h
  include/eshkol_transformer/fsdp_zero_abi.h
  include/eshkol_transformer/grpo_abi.h
  include/eshkol_transformer/hybrid_mesh_abi.h
  include/eshkol_transformer/io5_quant_abi.h
  include/eshkol_transformer/kv_cache_quant_abi.h
  include/eshkol_transformer/lora_adapter_abi.h
  include/eshkol_transformer/mamba_ssm_abi.h
  include/eshkol_transformer/mod5_adamw_abi.h
  include/eshkol_transformer/mod5_generate_abi.h
  include/eshkol_transformer/mod5_mla_abi.h
  include/eshkol_transformer/mod5_moe_abi.h
  include/eshkol_transformer/mod5_primitives_abi.h
  include/eshkol_transformer/mod5_rope_abi.h
  include/eshkol_transformer/mp4_mixed_precision_abi.h
  include/eshkol_transformer/mtp_head_abi.h
  include/eshkol_transformer/multi_lora_sgmv_abi.h
  include/eshkol_transformer/o2_muon_abi.h
  include/eshkol_transformer/pa4_attention_candidate.h
  include/eshkol_transformer/pa4_wider_attention_abi.h
  include/eshkol_transformer/paged_kv_abi.h
  include/eshkol_transformer/pipeline_1f1b_abi.h
  include/eshkol_transformer/ppo_rlhf_abi.h
  include/eshkol_transformer/qlora_nf4_abi.h
  include/eshkol_transformer/radix_kv_abi.h
  include/eshkol_transformer/ring_attn_abi.h
  include/eshkol_transformer/rope_yarn_abi.h
  include/eshkol_transformer/spec_decode_abi.h
  include/eshkol_transformer/t2_bpe_trainer_abi.h
  include/eshkol_transformer/tensor_parallel_abi.h
  include/eshkol_transformer/unigram_tok_abi.h
  include/eshkol_transformer/w4a32_quant_abi.h
  include/eshkol_transformer/wave4_composed_trainer_abi.h
)

for hdr in "${wave03_headers[@]}"; do
  [[ -f "${hdr}" ]] || die "missing Wave 0-3 header on disk: ${hdr}"
done

for hdr in "${wave45_headers[@]}"; do
  [[ -f "${hdr}" ]] || die "missing Wave 4/5 header on disk: ${hdr}"
  grep -Fq "${hdr}" docs/DOC5_API_REFERENCE.md ||
    die "docs/DOC5_API_REFERENCE.md missing Wave 4/5 header: ${hdr}"
  grep -Fq "${hdr}" docs/DOC5_WAVE45_GUIDE.md ||
    die "docs/DOC5_WAVE45_GUIDE.md missing Wave 4/5 header: ${hdr}"
done

declared_raw="${scratch}/declared-headers.raw"
declared_sorted="${scratch}/declared-headers.sorted"
disk_sorted="${scratch}/disk-headers.sorted"
printf '%s\n' "${wave03_headers[@]}" "${wave45_headers[@]}" > "${declared_raw}"
sort -u "${declared_raw}" > "${declared_sorted}"
(( $(wc -l < "${declared_raw}") == $(wc -l < "${declared_sorted}") )) ||
  die 'duplicate or overlapping entry between wave03_headers and wave45_headers'
ls -1 include/eshkol_transformer/*.h | sort > "${disk_sorted}"
if ! diff -u "${declared_sorted}" "${disk_sorted}" >&2; then
  die 'include/eshkol_transformer/*.h inventory drift: every public ABI header must be classified in wave03_headers or wave45_headers'
fi

bridge_count=0
for bridge in native/*_eshkol_bridge.h; do
  [[ -f "${bridge}" ]] || die "missing bridge header: ${bridge}"
  grep -Fq "${bridge}" docs/DOC5_WAVE45_GUIDE.md ||
    die "docs/DOC5_WAVE45_GUIDE.md missing Eshkol AOT bridge: ${bridge}"
  bridge_count=$((bridge_count + 1))
done
(( bridge_count > 0 )) || die 'no native/*_eshkol_bridge.h files found'

total_headers=$(( ${#wave03_headers[@]} + ${#wave45_headers[@]} ))
printf 'DOC5 header & bridge inventory: verified %s/%s Wave 4/5 headers, %s/%s total public ABI headers (%s Wave 0-3 + %s Wave 4/5), and %s Eshkol AOT bridges\n' \
  "${#wave45_headers[@]}" "${#wave45_headers[@]}" \
  "${total_headers}" "${total_headers}" \
  "${#wave03_headers[@]}" "${#wave45_headers[@]}" \
  "${bridge_count}"

# 4. Live end-to-end verification pass on the documented walkthrough targets.
run_make_gate() {
  local target="$1" marker="$2" log="${scratch}/$1.log"
  make "${target}" 2>&1 | tee "${log}"
  grep -Eq -- "${marker}" "${log}" ||
    die "DOC5 Wave 4/5 ${target} completed without expected marker: ${marker}"
}

printf '\n== E2E 8-stage compiled Eshkol AOT LLM lifecycle ==\n'
run_make_gate test-e2e-lifecycle \
  '^E2E_LIFECYCLE dual fresh AOT ELF \+ 5-process stdout byte-identical: PASS \(ELF SHA-256 [0-9a-f]{64}\)$'

printf '\n== B0 Wave 5 11-engine native benchmark suite ==\n'
run_make_gate test-b0-wave5 \
  '^B0 Wave 5 extended benchmark suite: PASS$'

printf '\n== Q0 Python isolation boundary ==\n'
run_make_gate test-python-isolation '^OK$'

printf '\n== CI topology verification ==\n'
run_make_gate test-ci-topology '^CI TOPOLOGY PASS:'

printf '\nDOC5 Wave 4/5 walkthrough: PASS\n'
