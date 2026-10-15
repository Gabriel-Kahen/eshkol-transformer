#!/usr/bin/env bash
# Deferred pinned N5 gate. Do not use this as a source-only host check.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env git grep nm python3 realpath timeout; do
  require_command "${command}"
done
[[ -x /usr/bin/time ]] || die "GNU time is required"
[[ $# -eq 1 && "$1" == /* ]] || die "usage: $0 FRESH_ABSOLUTE_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "N5 output directory must be fresh"
[[ "${out}" != "${PROJECT_ROOT}" && "${out}" != "${PROJECT_ROOT}/"* ]] || \
  die "N5 evidence must remain outside the checkout"
git -C "${PROJECT_ROOT}" diff --quiet HEAD -- || die "source tree is dirty"
[[ -z "$(git -C "${PROJECT_ROOT}" ls-files --others --exclude-standard)" ]] || \
  die "source tree has untracked files"
git -C "${PROJECT_ROOT}" merge-base --is-ancestor \
  de9d9344a3a5662ba2d6fafc03cce72c47bb783c HEAD || \
  die "reviewed N5 source parent is absent"
mkdir -m 0700 -- "${out}"
trap 'status=$?; printf "%d\n" "$status" >"${out}/gate.exit"; if ((status)); then printf "N5 evidence retained: %s\n" "$out" >&2; fi' EXIT
printf 'head\t%s\ntree\t%s\n' \
  "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" \
  "$(git -C "${PROJECT_ROOT}" rev-parse HEAD^{tree})" >"${out}/source.tsv"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "${out}/toolchain.tsv"

for suffix in a b; do
  python3 -B -m tests.e3_private.n5_corpus \
    --output "${out}/one-over-${suffix}" \
    >"${out}/one-over-${suffix}.generation"
  python3 -B -m tests.e3_private.n5_corpus --boundary \
    --output "${out}/boundary-${suffix}" \
    >"${out}/boundary-${suffix}.generation"
done
diff -r "${out}/one-over-a" "${out}/one-over-b"
diff -r "${out}/boundary-a" "${out}/boundary-b"
python3 - "${out}" <<'PY'
from pathlib import Path
import hashlib
import sys
root = Path(sys.argv[1])
lines = []
for kind in ("one-over", "boundary"):
    corpus = root / f"{kind}-a"
    files = sorted(p for p in corpus.iterdir() if p.is_file())
    if len(files) != 10:
        raise SystemExit("N5 D1 payload inventory differs from nine shards and manifest")
    total = 0
    for path in files:
        data = path.read_bytes()
        total += len(data)
        lines.append(f"{kind}\t{path.name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    lines.append(f"{kind}\tTOTAL\t{total}\t-")
(root / "payload.tsv").write_text("\n".join(lines) + "\n")
PY

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
for profile in normal sanitized; do
  flags=(-std=c++17 -Wall -Wextra -Werror -fstack-protector-all
    -fPIC -fvisibility=hidden -fno-common)
  if [[ "${profile}" == sanitized ]]; then
    flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${cxx}" "${flags[@]}" \
    -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
    -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
    -o "${out}/retention_arena-${profile}.o" \
    >"${out}/arena-${profile}.stdout" 2>"${out}/arena-${profile}.stderr"
  nm -g --defined-only "${out}/retention_arena-${profile}.o" | \
    grep -E '[[:space:]]T et_e3_retention_region_used_v1$' >/dev/null || \
    die "N5 arena observer missing: ${profile}"
done
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"

for mode in normal gte; do
  for profile in normal sanitized; do
    artifact="${out}/private-${mode}-${profile}"
    build_profile=normal
    [[ "${profile}" != sanitized ]] || build_profile=sanitize
    /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-n5-cap-test.sh" \
      "${artifact}" "${mode}" "${build_profile}" \
      >"${out}/build-${mode}-${profile}.stdout" \
      2>"${out}/build-${mode}-${profile}.stderr"
    closure="${PROJECT_ROOT}/tests/e3_private/fault_package_source_closure.txt"
    [[ "${mode}" != gte ]] || \
      closure="${PROJECT_ROOT}/tests/e3_private/fault_n5_cap_mutant_source_closure.txt"
    cmp "${closure}" "${artifact}/e3_fault_package.o.evidence/source-closure.txt"
    if [[ "${profile}" == sanitized ]]; then
      e3_tuple_kind=fault
      source "${PROJECT_ROOT}/scripts/e3-retention-sanitizer-import-policy.sh"
      e3_retention_verify_sanitizer_imports \
        "${PROJECT_ROOT}/tests/e3_private/fault_package_undefined_symbols.txt" \
        "${artifact}/e3_fault_package.o.evidence/noninstrumentation-undefined.txt" \
        "${artifact}/e3_fault_package.o.evidence/sanitizer-undefined.txt"
    fi
    mkdir "${out}/link-${mode}-${profile}"
    ar rcsD "${out}/link-${mode}-${profile}/libeshkol_transformer_e3_n5.a" \
      "${artifact}/e3_fault_package.o" \
      "${out}/retention_arena-${profile}.o"
    compiler="${cxx}"
    [[ "${profile}" != sanitized ]] || compiler="${out}/cxx-sanitized"
    env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
      -u DEPENDENCIES_OUTPUT -u GCC_EXEC_PREFIX -u COMPILER_PATH \
      -u LIBRARY_PATH -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS \
      -u CCC_CC -u CCC_CXX -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${out}/cache-${mode}-${profile}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s \
        "${E3_COMPILER_TIMEOUT_SECONDS:-900}s" \
        "${runner}" --strict-types --no-stdlib -O 2 \
        -L "${out}/link-${mode}-${profile}" --lib eshkol_transformer_e3_n5 \
        "${PROJECT_ROOT}/tests/e3_private/n5_cap_runtime.esk" \
        -o "${out}/runtime-${mode}-${profile}" \
        >"${out}/compile-${mode}-${profile}.stdout" \
        2>"${out}/compile-${mode}-${profile}.stderr"
    if nm -u "${out}/runtime-${mode}-${profile}" | \
        grep -E '[[:space:]]U et_e3_fault_run_v1(@|$)' >/dev/null; then
      die "N5 private CABI remained unresolved: ${mode}/${profile}"
    fi
  done
done

run_profile() {
  local stem=$1 executable=$2 corpus=$3 gradient=$4 mode=$5 poison=$6
  ESHKOL_ARENA_POISON="${poison}" \
    E3_RETENTION_RSS_PATH="${out}/${stem}.rss" \
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
      -o "${out}/${stem}.time" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${E3_N5_RUNTIME_TIMEOUT_SECONDS:-7200}s" \
      "${executable}" "${corpus}" "${gradient}" "${mode}" \
      >"${out}/${stem}.stdout" 2>"${out}/${stem}.stderr"
  [[ ! -s "${out}/${stem}.stderr" ]] || die "N5 ${stem} wrote stderr"
  python3 - "${out}/${stem}.time" "${out}/${stem}.rss" <<'PY'
from pathlib import Path
import re
import sys
data = Path(sys.argv[1]).read_text()
if not re.fullmatch(r"elapsed_seconds=[0-9]+(?:\.[0-9]+)? peak_rss_kib=([0-9]+)\n", data):
    raise SystemExit("N5 resource record malformed")
rss = int(re.search(r"peak_rss_kib=([0-9]+)", data).group(1))
if not 0 < rss <= 524288:
    raise SystemExit(f"N5 peak RSS exceeded the 512 MiB gate: {rss}")
sample = Path(sys.argv[2]).read_text()
match = re.fullmatch(r"rss_before_kib=([0-9]+) rss_after_kib=([0-9]+)\n", sample)
if not match or any(int(word) <= 0 for word in match.groups()):
    raise SystemExit("N5 in-process RSS record malformed")
PY
}

for gradient in absent present; do
  oracle="oracle-${gradient}"
  run_profile "${oracle}" "${out}/runtime-normal-normal" \
    "${out}/boundary-a" "${gradient}" oracle 0
  grep -E '^E3-N5-METRICS loss_bits=[0-9]+ weight_bits=[0-9]+ ppl_bits=[0-9]+ accuracy_bits=[0-9]+ tokens=16384 batches=8192$' \
    "${out}/${oracle}.stdout" >/dev/null
  [[ "$(wc -l <"${out}/${oracle}.stdout")" == 1 ]] || \
    die "N5 oracle emitted an unexpected transcript"
  for mode in normal gte; do
    for profile in normal repeat poison sanitized; do
      executable="${out}/runtime-${mode}-normal"
      poison=0
      if [[ "${profile}" == poison || "${profile}" == sanitized ]]; then
        poison=1
      fi
      [[ "${profile}" != sanitized ]] || \
        executable="${out}/runtime-${mode}-sanitized"
      stem="${mode}-${gradient}-${profile}"
      run_profile "${stem}" "${executable}" "${out}/one-over-a" \
        "${gradient}" "${mode}" "${poison}"
      result=success
      [[ "${mode}" != gte ]] || result=unsupported
      printf 'E3-N5-CAP-PASS gradient=%s mode=%s one_over=8193 boundary=8192 first_error=29 boundary_result=%s\n' \
        "${gradient}" "${mode}" "${result}" >"${out}/${stem}.expected"
      if [[ "${mode}" == normal ]]; then
        cat "${out}/${oracle}.stdout" >>"${out}/${stem}.expected"
      fi
      cmp "${out}/${stem}.expected" "${out}/${stem}.stdout" || \
        die "N5 transcript/metric oracle mismatch: ${stem}"
      if [[ "${mode}" == gte ]]; then
        [[ "$(wc -l <"${out}/${stem}.stdout")" == 1 ]] || \
          die "N5 rejected mutant emitted a metric transcript"
      fi
    done
    for profile in repeat poison sanitized; do
      cmp "${out}/${mode}-${gradient}-normal.stdout" \
        "${out}/${mode}-${gradient}-${profile}.stdout"
    done
  done
done
printf 'E3-N5-COMPILED-PASS output=%s\n' "${out}"
