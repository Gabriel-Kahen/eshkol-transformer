#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C
export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"

runner="$(eshkol_build_dir)/eshkol-run"
mla_dir="${PROJECT_ROOT}/build/mod5_mla"

bash scripts/build-mod5-mla.sh "${mla_dir}"

scratch="$(mktemp -d "${mla_dir}/.test.XXXXXX")"
diagnostic=''
cleanup() {
  local status=$?
  if [[ "${status}" != 0 && -n "${diagnostic}" && -s "${diagnostic}" ]]; then
    cat "${diagnostic}" >&2
  fi
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

# Verify deterministic archive builds across two independent directories.
bash scripts/build-mod5-mla.sh "${scratch}/det-build"
cmp "${mla_dir}/libeshkol_transformer_mod5_mla.a" \
    "${scratch}/det-build/libeshkol_transformer_mod5_mla.a"
cmp "${mla_dir}/libeshkol_transformer_mod5_mla_bridge.a" \
    "${scratch}/det-build/libeshkol_transformer_mod5_mla_bridge.a"
archive_sha="$(sha256sum "${mla_dir}/libeshkol_transformer_mod5_mla.a")"
printf 'MOD5 MLA deterministic archive SHA-256: %s\n' "${archive_sha%% *}"

bash scripts/build-k1.sh "${scratch}/k1"

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
        -I include -I native)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

# 1. C++17 header and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/mod5_mla/header_mla_cpp.cpp \
  "${mla_dir}/mod5_mla_eshkol_bridge.o" \
  "${mla_dir}/libeshkol_transformer_mod5_mla.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" -lm -o "${scratch}/header-cpp"
"${scratch}/header-cpp"
printf 'MOD5 MLA C++17 header and linkage: PASS\n'

# 2. Native C11 suite (normal -O2, allocation-trapped dispatch).
"${CC}" "${cflags[@]}" -O2 tests/mod5_mla/test_mod5_mla_native.c \
  "${mla_dir}/mod5_mla_eshkol_bridge.o" \
  "${mla_dir}/libeshkol_transformer_mod5_mla.a" \
  "${scratch}/k1/libeshkol_transformer_k1.a" "${wrap[@]}" -lm \
  -o "${scratch}/test_mod5_mla_native"
"${scratch}/test_mod5_mla_native"

# 3. Independent binary64 Python oracle.
python3 tests/mod5_mla/test_mod5_mla_oracle.py "${scratch}/test_mod5_mla_native"

# 4. Native C11 + C++17 under ASan/UBSan/LSan.
for src in kernel_abi mod5_mla_provider mod5_mla_eshkol_bridge; do
  "${CC}" "${cflags[@]}" "${san[@]}" -c "native/${src}.c" \
    -o "${scratch}/${src}.san.o"
done
san_objs=("${scratch}/mod5_mla_eshkol_bridge.san.o"
          "${scratch}/mod5_mla_provider.san.o"
          "${scratch}/kernel_abi.san.o")
"${CC}" "${cflags[@]}" "${san[@]}" tests/mod5_mla/test_mod5_mla_native.c \
  "${san_objs[@]}" "${wrap[@]}" -lm -o "${scratch}/test_mod5_mla_native.san"
"${CXX}" "${cppflags[@]}" "${san[@]}" tests/mod5_mla/header_mla_cpp.cpp \
  "${san_objs[@]}" -lm -o "${scratch}/header-cpp.san"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/test_mod5_mla_native.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/header-cpp.san"
printf 'MOD5 MLA ASan/UBSan/LSan: PASS\n'

# 5. Compiled Eshkol AOT MLA training + KV-cache decode (compiled twice from fresh dirs).
for run in 1 2; do
  mkdir "${scratch}/cache-${run}" "${scratch}/link-${run}"
  # The pinned driver removes positional object inputs after linking.
  cp "${mla_dir}/mod5_mla_eshkol_bridge.o" "${scratch}/link-${run}/"
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${mla_dir}" -L "${scratch}/k1" \
      --lib eshkol_transformer_mod5_mla \
      --lib eshkol_transformer_k1 \
      tests/mod5_mla/test_mod5_mla_aot.esk \
      "${scratch}/link-${run}/mod5_mla_eshkol_bridge.o" \
      -o "${scratch}/mla-aot-${run}" >"${diagnostic}" 2>&1
  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s "${scratch}/mla-aot-${run}" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
    cat "${scratch}/run-${run}.stdout" >&2
    die "compiled MLA AOT executable failed on run ${run}"
  }
  [[ ! -s "${diagnostic}" ]] || die "compiled MLA AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/mla-aot-1" >/dev/null
cmp "${scratch}/mla-aot-1" "${scratch}/mla-aot-2"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

awk '
  BEGIN { number = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" }
  $0 == "MOD5 MLA AOT malformed statuses PASS" { ++negative; next }
  $1 == "MOD5" && $2 == "MLA" && $3 == "AOT" && $4 == "FD_GRADIENTS" {
    if (NF != 7 || $5 != "checked=18" || $6 !~ /^max_abs_err=/ || $7 != "PASS")
      exit 1
    ++fd_ok
    next
  }
  $1 == "MOD5" && $2 == "MLA" && $3 == "AOT" && $4 == "LOSS" {
    if (NF != 6 || $5 !~ /^[0-9]+$/ || $5 != steps || $6 !~ number ||
        $6 + 0 < 0 || (steps && $6 + 0 >= previous))
      exit 1
    if (!steps) initial = $6
    final = $6
    previous = $6 + 0
    ++steps
    next
  }
  $1 == "MOD5" && $2 == "MLA" && $3 == "AOT" && $4 == "KV_CACHE" {
    if (NF != 11 || $5 != "prefill=3" || $6 != "decode=5" ||
        $7 != "latent_bytes=256" || $8 != "full_mha_kv_bytes=2048" ||
        $9 !~ /^compression_ratio=8(\.0+)?$/ || $10 !~ /^max_abs_diff=/ ||
        $11 != "PASS")
      exit 1
    ++kv_ok
    next
  }
  $1 == "MOD5" && $2 == "MLA" && $3 == "AOT" && $4 == "RESULT" {
    if (steps != 13 || NF != 14 || $5 != "n=1" || $6 != "t=8" ||
        $7 != "d=16" || $8 != "h=4" || $9 != "dh=8" || $10 != "dc=8" ||
        $11 != "steps=12" || $12 != "initial=" initial ||
        $13 != "final=" final || $14 !~ /^checksum=/)
      exit 1
    ++result_ok
    next
  }
  { exit 1 }
  END {
    if (negative != 1 || fd_ok != 1 || kv_ok != 1 || result_ok != 1 || steps != 13)
      exit 1
  }
' "${scratch}/run-1.stdout" || {
  cat "${scratch}/run-1.stdout" >&2
  die "invalid MLA AOT output trace"
}

cat "${scratch}/run-1.stdout"
digest="$(sha256sum "${scratch}/mla-aot-1")"
printf 'MOD5 MLA AOT ELF SHA-256 %s\n' "${digest%% *}"
printf 'MOD5 MLA AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
