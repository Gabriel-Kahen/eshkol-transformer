#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

ppo_dir="${PROJECT_ROOT}/build/ppo_rlhf"
bash scripts/build-ppo-rlhf.sh "${ppo_dir}"

scratch="$(mktemp -d "${ppo_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

archives=(
  "${ppo_dir}/libeshkol_transformer_ppo_rlhf_aot.a"
  "${ppo_dir}/libeshkol_transformer_ppo_rlhf.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" tests/ppo_rlhf/header_ppo_cpp.cpp "${archives[@]}" -lm \
  -o "${scratch}/header_ppo_cpp"
"${scratch}/header_ppo_cpp"
printf 'PPO-RLHF C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2) tests + Python binary64 reference oracle + f64 FD.
"${CC}" "${cflags[@]}" -O2 tests/ppo_rlhf/test_ppo_rlhf_native.c "${archives[@]}" \
  -lm -o "${scratch}/test_ppo_rlhf_native"
"${scratch}/test_ppo_rlhf_native"
python3 tests/ppo_rlhf/test_ppo_rlhf_oracle.py "${scratch}/test_ppo_rlhf_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" tests/ppo_rlhf/test_ppo_rlhf_native.c \
  native/ppo_rlhf_eshkol_bridge.c native/ppo_rlhf.c -lm \
  -o "${scratch}/test_ppo_rlhf_native.san"
"${scratch}/test_ppo_rlhf_native.san"

"${CC}" "${cflags[@]}" "${san[@]}" -c native/ppo_rlhf.c \
  -o "${scratch}/ppo_rlhf.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/ppo_rlhf_eshkol_bridge.c \
  -o "${scratch}/ppo_rlhf_eshkol_bridge.san.o"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/ppo_rlhf/header_ppo_cpp.cpp \
  "${scratch}/ppo_rlhf_eshkol_bridge.san.o" "${scratch}/ppo_rlhf.san.o" -lm \
  -o "${scratch}/header_ppo_cpp.san"
"${scratch}/header_ppo_cpp.san"
printf 'PPO-RLHF ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/ppo_rlhf/test_ppo_rlhf_aot.esk TWICE from fresh dirs and
#    verify byte-identical ELFs and byte-identical combined train+reload stdout.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" \
           "${scratch}/libs-${run}" "${scratch}/run-${run}"
  bash scripts/build-ppo-rlhf.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_ppo_rlhf_aot --lib eshkol_transformer_ppo_rlhf \
      tests/ppo_rlhf/test_ppo_rlhf_aot.esk \
      -o "${scratch}/out-${run}/test_ppo_rlhf_aot" >"${diagnostic}" 2>&1

  ckpt="${scratch}/run-${run}/model.etpp"
  tamper="${scratch}/run-${run}/tamper.etpp"
  trunc="${scratch}/run-${run}/trunc.etpp"
  sym="${scratch}/run-${run}/link.etpp"

  # Fresh process 1: FD checks, 16-iter PPO training, mid-run ETPP1000 save,
  # ref SHA-256 invariance, and tight target_kl early stopping.
  diagnostic="${scratch}/train-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env PPO_AOT_PHASE=train PPO_AOT_CKPT="${ckpt}" PPO_AOT_TAMPER="${tamper}" \
        PPO_AOT_TRUNC="${trunc}" PPO_AOT_SYMLINK="${sym}" \
    "${scratch}/out-${run}/test_ppo_rlhf_aot" \
    >"${scratch}/train-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/train-${run}.stdout" >&2
      die "compiled PPO-RLHF AOT train process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled PPO-RLHF AOT train emitted stderr"

  # Fresh process 2: reload ETPP1000 checkpoint at iter 8, resume iters 8..15,
  # verify bitwise SHA-256 parity + tamper/trunc/symlink/ref-mismatch + negatives.
  diagnostic="${scratch}/reload-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    env PPO_AOT_PHASE=reload PPO_AOT_CKPT="${ckpt}" PPO_AOT_TAMPER="${tamper}" \
        PPO_AOT_TRUNC="${trunc}" PPO_AOT_SYMLINK="${sym}" \
    "${scratch}/out-${run}/test_ppo_rlhf_aot" \
    >"${scratch}/reload-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/reload-${run}.stdout" >&2
      die "compiled PPO-RLHF AOT reload process failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled PPO-RLHF AOT reload emitted stderr"

  cat "${scratch}/train-${run}.stdout" "${scratch}/reload-${run}.stdout" \
    >"${scratch}/combined-${run}.stdout"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_ppo_rlhf_aot" >/dev/null
cmp "${scratch}/out-1/test_ppo_rlhf_aot" "${scratch}/out-2/test_ppo_rlhf_aot"
cmp "${scratch}/combined-1.stdout" "${scratch}/combined-2.stdout"

awk '
  BEGIN { num = "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $1 == "PPO" && $2 == "AOT" && $3 == "FD" {
    if (NF != 6 || $6 != "checks=30") exit 1
    split($4, ml, "="); split($5, mv, "=")
    if (ml[1] != "max_logit_err" || ml[2] !~ num || ml[2] + 0 >= 1e-5) exit 1
    if (mv[1] != "max_value_err" || mv[2] !~ num || mv[2] + 0 >= 1e-5) exit 1
    ++fd_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "ITER" {
    if ($4 !~ /^[0-9]+$/ || $4 + 0 != iter) exit 1
    split($5, er, "=")
    if (er[1] != "exp_reward" || er[2] !~ num) exit 1
    if (iter == 0) {
      if (NF != 6 || $6 != "step=0") exit 1
      init_r = er[2]
    } else {
      if (NF != 8 || er[2] + 0 <= prev_r) exit 1
    }
    final_r = er[2]; prev_r = er[2] + 0; ++iter; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "TRAIN" {
    if (NF != 8 || $4 != "iters=16" || $5 != "init_reward=" init_r ||
        $6 != "final_reward=" final_r || $8 != "ref_frozen=PASS") exit 1
    ++train_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "TRAIN_REF_SHA256" {
    if (NF != 7) exit 1
    tr_ref = $4 " " $5 " " $6 " " $7; ++tr_ref_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "TRAIN_PARAM_SHA256" {
    if (NF != 7) exit 1
    tr_par = $4 " " $5 " " $6 " " $7; ++tr_par_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "EARLY_STOP" {
    if (NF != 6 || $4 != "triggered=1") exit 1
    split($5, ep, "="); split($6, kl, "=")
    if (ep[1] != "epochs" || ep[2] + 0 >= 4 || kl[1] != "approx_kl" || kl[2] + 0 <= 1e-4) exit 1
    ++es_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "RELOAD" {
    if (NF != 5 || $4 != "final_reward=" final_r) exit 1
    ++reload_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "RELOAD_REF_SHA256" {
    if (NF != 7) exit 1
    rl_ref = $4 " " $5 " " $6 " " $7; ++rl_ref_seen; next
  }
  $1 == "PPO" && $2 == "AOT" && $3 == "RELOAD_PARAM_SHA256" {
    if (NF != 7) exit 1
    rl_par = $4 " " $5 " " $6 " " $7; ++rl_par_seen; next
  }
  $0 == "PPO AOT NEGATIVE PASS checks=17" { ++neg_seen; next }
  { exit 1 }
  END {
    if (fd_seen != 1 || iter != 17 || train_seen != 1 || tr_ref_seen != 1 ||
        tr_par_seen != 1 || es_seen != 1 || reload_seen != 1 ||
        rl_ref_seen != 1 || rl_par_seen != 1 || neg_seen != 1) exit 1
    if (tr_ref != rl_ref || tr_par != rl_par) exit 1
  }
' "${scratch}/combined-1.stdout" || {
  cat "${scratch}/combined-1.stdout" >&2
  die "invalid compiled PPO-RLHF AOT output"
}

cat "${scratch}/combined-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_ppo_rlhf_aot")"
printf 'PPO-RLHF AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'PPO-RLHF AOT two fresh compilations, byte-identical ELF, stdout & fresh-process resume: PASS\n'
