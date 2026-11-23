#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
cc="${CC}"
cxx="${CXX}"
runner="$(eshkol_build_dir)/eshkol-run"

for foundation in wave4-composed ac4 mod5 mp4 dist5 eval5; do
  bash "scripts/build-${foundation}.sh"
done
bash scripts/build-wave4-composed-aot.sh

aot_dir="${PROJECT_ROOT}/build/wave4_composed_aot"
w4_dir="${PROJECT_ROOT}/build/wave4-composed"
ac4_dir="${PROJECT_ROOT}/build/ac4"
mod5_dir="${PROJECT_ROOT}/build/mod5"
mp4_dir="${PROJECT_ROOT}/build/mp4"
dist5_dir="${PROJECT_ROOT}/build/dist5"
eval5_dir="${PROJECT_ROOT}/build/eval5"

scratch="$(mktemp -d "${aot_dir}/.test.XXXXXX")"
sock_root="$(mktemp -d /tmp/eshkol-w4-aot.XXXXXX)"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}" "${sock_root}"
}
trap cleanup EXIT

bash scripts/build-k1.sh "${scratch}/k1"

archives=(
  "${aot_dir}/libeshkol_transformer_wave4_composed_aot.a"
  "${w4_dir}/libeshkol_wave4_composed.a"
  "${ac4_dir}/libeshkol_ac4.a"
  "${mod5_dir}/libeshkol_transformer_mod5.a"
  "${mp4_dir}/libeshkol_transformer_mp4.a"
  "${dist5_dir}/libeshkol_dist5.a"
  "${eval5_dir}/libeshkol_eval5.a"
  "${scratch}/k1/libeshkol_transformer_k1.a"
)

native_sources=(
  native/wave4_composed_eshkol_bridge.c
  native/wave4_composed_trainer.c
  native/ac4_checkpointing.c
  native/mod5_primitives_provider.c
  native/mp4_mixed_precision_provider.c
  native/dist5_collectives.c
  native/eval5_benchmark.c
  native/kernel_abi.c
)

# 1. C++17 header compilation and archive linkage check.
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I include -I native tests/wave4/header_bridge_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_bridge_cpp"
"${scratch}/header_bridge_cpp"
printf 'Wave 4/5 composed AOT bridge C++17 header and linkage: PASS\n'

# 2. Native C11 bridge tests in normal (-O2) AND ASan/UBSan/LSan modes.
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include -I native)

"${cc}" "${cflags[@]}" -O2 tests/wave4/test_wave4_composed_aot_bridge.c \
  "${archives[@]}" -lm -o "${scratch}/test_wave4_composed_aot_bridge"
"${scratch}/test_wave4_composed_aot_bridge"
printf 'Wave 4/5 composed AOT bridge normal (-O2) tests: PASS\n'

san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${cc}" "${cflags[@]}" "${san[@]}" tests/wave4/test_wave4_composed_aot_bridge.c \
  "${native_sources[@]}" -lm -o "${scratch}/test_wave4_composed_aot_bridge.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_wave4_composed_aot_bridge.san"
printf 'Wave 4/5 composed AOT bridge ASan/UBSan/LSan tests: PASS\n'

# 3. Compile tests/wave4/test_wave4_composed_aot.esk TWICE from fresh dirs
#    and verify byte-identical ELFs.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" "${scratch}/aot-${run}"
  bash scripts/build-wave4-composed-aot.sh "${scratch}/aot-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/aot-${run}" -L "${w4_dir}" -L "${ac4_dir}" \
      -L "${mod5_dir}" -L "${mp4_dir}" -L "${dist5_dir}" \
      -L "${eval5_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_wave4_composed_aot \
      --lib eshkol_wave4_composed \
      --lib eshkol_ac4 \
      --lib eshkol_transformer_mod5 \
      --lib eshkol_transformer_mp4 \
      --lib eshkol_dist5 \
      --lib eshkol_eval5 \
      --lib eshkol_transformer_k1 \
      tests/wave4/test_wave4_composed_aot.esk \
      -o "${scratch}/out-${run}/test_wave4_composed_aot" >"${diagnostic}" 2>&1
done
diagnostic=''

elf="${scratch}/out-1/test_wave4_composed_aot"
readelf -h "${elf}" >/dev/null
cmp "${elf}" "${scratch}/out-2/test_wave4_composed_aot"
elf_sha256="$(sha256sum "${elf}" | awk '{print $1}')"
printf 'Wave 4/5 composed AOT two fresh compilations, byte-identical ELF: PASS (SHA-256 %s)\n' \
  "${elf_sha256}"

# 4. Orchestrate live multi-process W=1, 2 runs across L=4, 8, f16 & bf16,
#    comparing NONE vs SQRT_SEGMENTS.
run_lane() {
  local layers="$1" world="$2" dtype="$3" policy="$4" lane_dir="$5"
  local sock_dir="${sock_root}/L${layers}-W${world}-${dtype}-${policy}"
  local eval_path="${lane_dir}/eval.manifest"
  mkdir -p "${lane_dir}" "${sock_dir}"
  local pids=()
  local r
  for (( r = 0; r < world; ++r )); do
    timeout --foreground --kill-after=5s 60s \
      "${elf}" \
        --sock-dir "${sock_dir}" \
        --rank "${r}" \
        --world "${world}" \
        --layers "${layers}" \
        --dtype "${dtype}" \
        --policy "${policy}" \
        --eval-path "${eval_path}" \
        >"${lane_dir}/rank-${r}.stdout" \
        2>"${lane_dir}/rank-${r}.stderr" &
    pids+=("$!")
  done
  local failed=0
  for (( r = 0; r < world; ++r )); do
    if ! wait "${pids[r]}"; then
      failed=1
    fi
  done
  for (( r = 0; r < world; ++r )); do
    if [[ -s "${lane_dir}/rank-${r}.stderr" ]]; then
      cat "${lane_dir}/rank-${r}.stderr" >&2
      failed=1
    fi
  done
  if [[ "${failed}" != 0 ]]; then
    for (( r = 0; r < world; ++r )); do
      if [[ -s "${lane_dir}/rank-${r}.stdout" ]]; then
        printf -- '--- L=%s W=%s %s %s rank=%s stdout ---\n' \
          "${layers}" "${world}" "${dtype}" "${policy}" "${r}" >&2
        cat "${lane_dir}/rank-${r}.stdout" >&2
      fi
    done
    die "compiled Eshkol AOT rank process failed (L=${layers} W=${world} ${dtype} ${policy})"
  fi
  [[ -s "${eval_path}" ]] || die "missing EVAL5 manifest at ${eval_path}"
}

for layers in 4 8; do
  for dtype in f16 bf16; do
    for world in 1 2; do
      none_dir="${scratch}/run-L${layers}-W${world}-${dtype}-none"
      sqrt_dir="${scratch}/run-L${layers}-W${world}-${dtype}-sqrt"
      run_lane "${layers}" "${world}" "${dtype}" none "${none_dir}"
      run_lane "${layers}" "${world}" "${dtype}" sqrt "${sqrt_dir}"

      # Validate bitwise parity across ranks and across NONE vs SQRT_SEGMENTS,
      # overflow backoff + underflow rejection, strict loss recovery, and
      # activation peak memory reduction.
      awk -v layers="${layers}" -v world="${world}" -v dtype="${dtype}" '
        function parse_kv(arr, start, nf,    i, k, v, p) {
          delete arr
          for (i = start; i <= nf; ++i) {
            p = index($i, "=")
            if (p <= 1) return 0
            k = substr($i, 1, p - 1)
            v = substr($i, p + 1)
            arr[k] = v
          }
          return 1
        }
        {
          policy_tag = (FILENAME ~ /-none\//) ? "NONE" : "SQRT"
          if ($1 == "WAVE4" && $2 == "AOT" && $3 == "STEP") {
            if (!parse_kv(kv, 4, NF)) exit 1
            r = kv["rank"] + 0
            s = kv["step"] + 0
            key = policy_tag ":" r ":" s
            if (key in step_seen) exit 1
            step_seen[key] = 1
            loss_str[key] = kv["loss"]
            loss_num[key] = kv["loss"] + 0
            if (s == 2) {
              if (kv["status"] != "OVERFLOW_BACKOFF") exit 1
              if (kv["scale"] + 0 != 32) exit 1
              if (kv["underflow_rejected"] != "PASS") exit 1
              if ((r == world - 1 && kv["local_overflow"] != "1") ||
                  (r != world - 1 && kv["local_overflow"] != "0")) exit 1
            } else {
              if (kv["status"] != "OK") exit 1
              if (kv["local_overflow"] != "0") exit 1
              if (s < 2 && kv["scale"] + 0 != 64) exit 1
              if (s > 2 && kv["scale"] + 0 != 32) exit 1
            }
            next
          }
          if ($1 == "WAVE4" && $2 == "AOT" && $3 == "RANK") {
            if (!parse_kv(kv, 4, NF)) exit 1
            r = kv["rank"] + 0
            if (kv["world"] + 0 != world || kv["layers"] + 0 != layers ||
                kv["dtype"] != dtype) exit 1
            if (kv["final_scale"] + 0 != 32) exit 1
            if (length(kv["weight_sha256"]) != 64) exit 1
            rkey = policy_tag ":" r
            if (rkey in rank_seen) exit 1
            rank_seen[rkey] = 1
            wsha[rkey] = kv["weight_sha256"]
            peak[rkey] = kv["peak_bytes"] + 0
            ws[rkey] = kv["workspace_bytes"] + 0
            narrow[rkey] = kv["narrow_bytes"] + 0
            next
          }
          if ($1 == "WAVE4" && $2 == "AOT" && $3 == "EVAL") {
            if (!parse_kv(kv, 4, NF)) exit 1
            if (kv["rank"] != "0" || kv["contaminated"] != "0") exit 1
            if (kv["perplexity"] + 0 <= 1.0) exit 1
            if (length(kv["manifest_sha256"]) != 64) exit 1
            if (policy_tag in eval_seen) exit 1
            eval_seen[policy_tag] = 1
            eval_nll[policy_tag] = kv["eval_nll"]
            eval_ppl[policy_tag] = kv["perplexity"]
            msha[policy_tag] = kv["manifest_sha256"]
            next
          }
          exit 1
        }
        END {
          if (!(("NONE" in eval_seen) && ("SQRT" in eval_seen))) exit 1
          if (eval_nll["NONE"] != eval_nll["SQRT"]) exit 1
          if (msha["NONE"] != msha["SQRT"]) exit 1

          for (r = 0; r < world; ++r) {
            if (!((("NONE:" r) in rank_seen) && (("SQRT:" r) in rank_seen))) exit 1
            if (wsha["NONE:" r] != wsha["NONE:0"]) exit 1
            if (wsha["SQRT:" r] != wsha["NONE:0"]) exit 1
            if (!(peak["SQRT:" r] < peak["NONE:" r])) exit 1
            if (!(ws["SQRT:" r] < ws["NONE:" r])) exit 1
            for (s = 0; s < 7; ++s) {
              kn = "NONE:" r ":" s
              ks = "SQRT:" r ":" s
              if (!(kn in step_seen) || !(ks in step_seen)) exit 1
              if (loss_str[kn] != loss_str[ks]) exit 1
              mean_loss[s] += loss_num[kn] / world
            }
          }

          prev = 1e300
          for (s = 0; s < 7; ++s) {
            if (s == 2) continue
            if (!(mean_loss[s] < prev)) exit 1
            prev = mean_loss[s]
          }

          saved_pct = 100.0 * (peak["NONE:0"] - peak["SQRT:0"]) / peak["NONE:0"]
          printf "WAVE4 AOT MATRIX L=%d W=%d %s peak=%d->%d (%.2f%%) ws=%d->%d weight_sha256=%s manifest_sha256=%s eval_nll=%s mean_losses=", \
            layers, world, dtype, peak["NONE:0"], peak["SQRT:0"], saved_pct, \
            ws["NONE:0"], ws["SQRT:0"], wsha["NONE:0"], msha["NONE"], eval_nll["NONE"]
          for (s = 0; s < 7; ++s) {
            printf "%s%.17g", (s == 0 ? "" : ","), mean_loss[s]
          }
          printf "\n"
        }
      ' "${none_dir}"/rank-*.stdout "${sqrt_dir}"/rank-*.stdout || {
        cat "${none_dir}"/rank-*.stdout "${sqrt_dir}"/rank-*.stdout >&2
        die "parity or invariant check failed for L=${layers} W=${world} ${dtype}"
      }
    done
  done
done

printf 'Wave 4/5 composed Eshkol AOT multi-process verification: PASS (ELF SHA-256 %s)\n' \
  "${elf_sha256}"
