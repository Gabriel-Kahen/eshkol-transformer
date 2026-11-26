#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
verify_toolchain
export LC_ALL=C

export CC="${CC:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)}"
export CXX="${CXX:-$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)}"
runner="$(eshkol_build_dir)/eshkol-run"

cd_dir="${PROJECT_ROOT}/build/constrained_decode"
bash scripts/build-constrained-decode.sh "${cd_dir}"

scratch="$(mktemp -d "${cd_dir}/.test.XXXXXX")"
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
san=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
     -fno-sanitize-recover=all)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

archives=(
  "${cd_dir}/libeshkol_transformer_constrained_decode_aot.a"
  "${cd_dir}/libeshkol_transformer_constrained_decode.a"
)

# 1. C++17 header compile and linkage check.
"${CXX}" "${cppflags[@]}" -O2 tests/constrained_decode/header_cd_cpp.cpp \
  "${archives[@]}" -lm -o "${scratch}/header_cd_cpp"
"${scratch}/header_cd_cpp"
printf 'CONSTRAINED_DECODE C++17 header and linkage: PASS\n'

# 2. Native C11 normal (-O2, zero-alloc wrapped) tests + independent Python oracle.
"${CC}" "${cflags[@]}" -O2 tests/constrained_decode/test_constrained_decode_native.c \
  "${archives[@]}" "${wrap[@]}" -lm -o "${scratch}/test_cd_native"
"${scratch}/test_cd_native"
python3 tests/constrained_decode/test_constrained_decode_oracle.py "${scratch}/test_cd_native"

# 3. Native C11 and C++17 tests under ASan/UBSan/LSan.
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
"${CC}" "${cflags[@]}" "${san[@]}" -c native/constrained_decode.c \
  -o "${scratch}/constrained_decode.san.o"
"${CC}" "${cflags[@]}" "${san[@]}" -c native/constrained_decode_eshkol_bridge.c \
  -o "${scratch}/constrained_decode_eshkol_bridge.san.o"

"${CC}" "${cflags[@]}" "${san[@]}" \
  tests/constrained_decode/test_constrained_decode_native.c \
  "${scratch}/constrained_decode_eshkol_bridge.san.o" \
  "${scratch}/constrained_decode.san.o" "${wrap[@]}" -lm \
  -o "${scratch}/test_cd_native.san"
"${scratch}/test_cd_native.san"

"${CXX}" "${cppflags[@]}" "${san[@]}" \
  tests/constrained_decode/header_cd_cpp.cpp \
  "${scratch}/constrained_decode_eshkol_bridge.san.o" \
  "${scratch}/constrained_decode.san.o" -lm \
  -o "${scratch}/header_cd_cpp.san"
"${scratch}/header_cd_cpp.san"
printf 'CONSTRAINED_DECODE ASan/UBSan/LSan tests: PASS\n'

# 4. Compile tests/constrained_decode/test_constrained_decode_aot.esk TWICE from
#    fresh directories and verify byte-identical ELFs and deterministic output.
for run in 1 2; do
  mkdir -p "${scratch}/cache-${run}" "${scratch}/out-${run}" "${scratch}/libs-${run}"
  bash scripts/build-constrained-decode.sh "${scratch}/libs-${run}" >/dev/null
  diagnostic="${scratch}/compile-${run}.log"
  timeout --foreground --kill-after=5s 90s \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${scratch}/cache-${run}" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
      -L "${scratch}/libs-${run}" \
      --lib eshkol_transformer_constrained_decode_aot \
      --lib eshkol_transformer_constrained_decode \
      tests/constrained_decode/test_constrained_decode_aot.esk \
      -o "${scratch}/out-${run}/test_constrained_decode_aot" \
      >"${diagnostic}" 2>&1

  diagnostic="${scratch}/run-${run}.stderr"
  timeout --foreground --kill-after=5s 60s \
    "${scratch}/out-${run}/test_constrained_decode_aot" \
    >"${scratch}/run-${run}.stdout" 2>"${diagnostic}" || {
      cat "${scratch}/run-${run}.stdout" >&2
      die "compiled Constrained Decode AOT binary failed on run ${run}"
    }
  [[ ! -s "${diagnostic}" ]] || die "compiled Constrained Decode AOT emitted stderr"
done
diagnostic=''

readelf -h "${scratch}/out-1/test_constrained_decode_aot" >/dev/null
cmp "${scratch}/out-1/test_constrained_decode_aot" \
    "${scratch}/out-2/test_constrained_decode_aot"
cmp "${scratch}/run-1.stdout" "${scratch}/run-2.stdout"

# 5. Independently verify every emitted JSON payload from compiled Eshkol AOT
#    parses as valid JSON matching the exact schema {"answer":<enum>,"score":<int>,"ok":<bool>}.
python3 - "${scratch}/run-1.stdout" <<'PY'
import json
import sys

lines = [ln.rstrip("\n") for ln in open(sys.argv[1], encoding="utf-8")]
sha_seen = 0
json_seen = {}
enum_seen = 0
neg_seen = 0

for ln in lines:
    if ln.startswith("CD AOT ARTIFACT_SHA256 "):
        parts = ln.split()
        assert len(parts) == 7, ln
        sha_seen += 1
    elif ln.startswith("CD AOT JSON "):
        parts = ln.split(" ", 6)
        assert len(parts) == 7, ln
        label = parts[3]
        assert parts[4].startswith("steps="), ln
        assert parts[5].startswith("adv_blocked="), ln
        assert parts[6].startswith("payload="), ln
        payload = parts[6][len("payload="):]
        obj = json.loads(payload)
        assert isinstance(obj, dict), payload
        assert list(obj.keys()) == ["answer", "score", "ok"], payload
        assert obj["answer"] in ("yes", "no", "maybe"), payload
        assert type(obj["score"]) is int, payload
        assert type(obj["ok"]) is bool, payload
        json_seen[label] = obj
    elif ln.startswith("CD AOT ENUM "):
        parts = ln.split(" ", 4)
        assert len(parts) == 5 and parts[4].startswith("payload="), ln
        val = parts[4][len("payload="):]
        assert val in ("yes", "no", "maybe"), val
        enum_seen += 1
    elif ln == "CD AOT NEGATIVE PASS":
        neg_seen += 1
    else:
        raise AssertionError(f"unexpected AOT output line: {ln}")

assert sha_seen == 1
assert enum_seen == 1
assert neg_seen == 1
assert json_seen == {
    "greedy": {"answer": "yes", "score": 42, "ok": True},
    "adversarial_multi": {"answer": "maybe", "score": -42, "ok": False},
    "adversarial_byte": {"answer": "no", "score": 0, "ok": True},
}, json_seen
PY

cat "${scratch}/run-1.stdout"
elf_sha="$(sha256sum "${scratch}/out-1/test_constrained_decode_aot")"
printf 'CONSTRAINED_DECODE AOT ELF SHA-256 %s\n' "${elf_sha%% *}"
printf 'CONSTRAINED_DECODE AOT two fresh compilations, byte-identical ELF and stdout: PASS\n'
