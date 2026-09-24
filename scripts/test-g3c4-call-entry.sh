#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp python3 timeout; do require_command "${command}"; done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-call-entry.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-call-entry-test}"
test_source="${G3C4_CALL_ENTRY_TEST_SOURCE:-${PROJECT_ROOT}/tests/g3c4/call_entry_test.esk}"
mkdir -p "${evidence}"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-call.XXXXXX")"
cleanup() {
  local result=$?
  if (( result != 0 )); then
    cat "${temporary_dir}/compile.log" >&2 || true
    cat "${temporary_dir}/runtime.stdout" >&2 || true
    cat "${temporary_dir}/allocation.stdout" >&2 || true
    cat "${temporary_dir}/publication.stdout" >&2 || true
    cat "${temporary_dir}"/retention-*.stdout >&2 || true
    cat "${temporary_dir}"/retention-*.stderr >&2 || true
    if [[ -f "${evidence}/native-affected.stdout" ]]; then
      cat "${evidence}/native-affected.stdout" >&2
    fi
  fi
  rm -rf -- "${temporary_dir}"
  return "${result}"
}
trap cleanup EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
  -I "$(eshkol_source_dir)/inc"
)
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_G3C4_I2_CONSTRUCTION_PRIVATE -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${temporary_dir}/i2_native_helpers.o"
"${cc}" "${cflags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
  -o "${temporary_dir}/m3t_f32_integration.o"
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/g3c4/call_entry_native.c" \
  -o "${temporary_dir}/g3c4_call_entry_native.o"
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/g3c4/call_entry_a2_stats.c" \
  -o "${temporary_dir}/g3c4_call_entry_a2_stats.o"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -fvisibility=hidden -fno-common -fstack-protector-all \
  -I "$(eshkol_source_dir)/inc" -I "$(eshkol_source_dir)/lib/core" \
  -c "${PROJECT_ROOT}/tests/g3c4/call_entry_allocation_shim.cpp" \
  -o "${temporary_dir}/g3c4_call_entry_allocation_shim.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${temporary_dir}/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell \
              n3k_primitives_provider n2_primitives_provider \
              a2_attention_provider; do
  "${cc}" "${cflags[@]}" -DET_A2_KV_CACHE_TESTING \
    -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
for source in m3_i64_integration m3_model; do
  "${cc}" "${cflags[@]}" \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
ar rcsD "${temporary_dir}/libg3c4_call_entry.a" \
  "${temporary_dir}"/*.o

cat >"${temporary_dir}/cxx-wrap" <<EOF
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") "\$@" \
  -Wl,--wrap=arena_allocate_vector_with_header \
  -Wl,--wrap=arena_allocate_cons_with_header \
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler
EOF
chmod +x "${temporary_dir}/cxx-wrap"

python3 - "${PROJECT_ROOT}/native/g3c4_call_entry_extension.esk" \
  "${temporary_dir}/g3c4_call_entry_fault_extension.esk" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
start = source.index("(define-syntax g3c4-with-call-internal")
head, body = source[:start], source[start:]
markers = [
    "(vector-set! canonical-ledger 0 #t)",
    "(vector-set! canonical-ledger 1 #t)",
    "(vector-set! canonical-ledger 2 #t)",
    "(vector-set! canonical-ledger 3 #t)",
    "(vector-set! m3-call-state 0 canonical)",
]
for number, marker in enumerate(markers, 1):
    if body.count(marker) != 1:
        raise SystemExit(f"publication marker drifted: {marker}")
    body = body.replace(marker, marker + f"\n             (g3c4-test-publication-cut {number})", 1)
Path(sys.argv[2]).write_text(head + body)
PY

compile_eshkol() {
  local source=$1 output=$2 log=$3 cache=$4
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${temporary_dir}/${cache}" \
    ESHKOL_CXX_COMPILER="${temporary_dir}/cxx-wrap" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${temporary_dir}" \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -L "${temporary_dir}" --lib g3c4_call_entry \
      "${source}" -o "${output}" >"${log}" 2>&1
}

compile_eshkol "${test_source}" "${temporary_dir}/call-entry" \
  "${temporary_dir}/compile.log" cache
compile_eshkol "${PROJECT_ROOT}/tests/g3c4/call_entry_failstop_test.esk" \
  "${temporary_dir}/call-entry-failstop" \
  "${temporary_dir}/failstop-compile.log" failstop-cache
compile_eshkol "${PROJECT_ROOT}/tests/g3c4/call_entry_allocation_test.esk" \
  "${temporary_dir}/call-entry-allocation" \
  "${temporary_dir}/allocation-compile.log" allocation-cache
compile_eshkol "${PROJECT_ROOT}/tests/g3c4/call_entry_publication_test.esk" \
  "${temporary_dir}/call-entry-publication" \
  "${temporary_dir}/publication-compile.log" publication-cache
compile_eshkol "${PROJECT_ROOT}/tests/g3c4/call_entry_retention.esk" \
  "${temporary_dir}/call-entry-retention" \
  "${temporary_dir}/retention-compile.log" retention-cache

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "${temporary_dir}/call-entry" \
  >"${temporary_dir}/runtime.stdout"
grep -E '^G3-C4 call entry PASS: checks=[1-9][0-9]*$' \
  "${temporary_dir}/runtime.stdout" >/dev/null
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "${temporary_dir}/call-entry" \
  >"${temporary_dir}/runtime-repeat.stdout"
cmp "${temporary_dir}/runtime.stdout" "${temporary_dir}/runtime-repeat.stdout"

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  300s "${temporary_dir}/call-entry-allocation" \
  >"${temporary_dir}/allocation.stdout"
grep -E '^G3-C4 allocation cuts PASS: .*handler=3 checks=[1-9][0-9]*$' \
  "${temporary_dir}/allocation.stdout" >/dev/null
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "${temporary_dir}/call-entry-publication" \
  >"${temporary_dir}/publication.stdout"
grep -E '^G3-C4 publication cuts PASS: cuts=5 checks=[1-9][0-9]*$' \
  "${temporary_dir}/publication.stdout" >/dev/null

for horizon in 1024 8192; do
  ESHKOL_ARENA_POISON=1 ESHKOL_ARENA_REPORT=1 \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
      "${temporary_dir}/call-entry-retention" "${horizon}" \
      >"${temporary_dir}/retention-${horizon}.stdout" \
      2>"${temporary_dir}/retention-${horizon}.stderr"
  grep -E "^G3-C4 Eshkol retention: horizon=${horizon} .*peak_native_bytes=1520 peak_live_caches=1 peak_cache_bytes=64$" \
    "${temporary_dir}/retention-${horizon}.stdout" >/dev/null
  grep -E '^\[eshkol-arena\] global_total_allocated_bytes=[1-9][0-9]*$' \
    "${temporary_dir}/retention-${horizon}.stderr" >/dev/null
  logical_bytes=$((horizon * 280))
  grep -F "horizon=${horizon} rng_entries=${horizon} rng_shells=${horizon} rng_registry_cells=${horizon} rng_logical_bytes=${logical_bytes} generator_entries=${horizon} generator_shells=${horizon} generator_registry_cells=${horizon} generator_logical_bytes=${logical_bytes} call_entries=${horizon} call_shells=${horizon} call_registry_cells=${horizon} call_logical_bytes=${logical_bytes}" \
    "${temporary_dir}/retention-${horizon}.stdout" >/dev/null
done
for mode in abort finish; do
  status=0
  timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${temporary_dir}/call-entry-failstop" "${mode}" \
    >"${temporary_dir}/failstop-${mode}.stdout" \
    2>"${temporary_dir}/failstop-${mode}.stderr" || status=$?
  [[ "${status}" == 134 ]] || \
    die "${mode} cleanup defect returned ${status}, expected fail-stop 134"
done

if [[ "${G3C4_SKIP_AFFECTED_GATE:-0}" == 1 ]]; then
  printf 'G3-C4 native affected gate: SKIPPED\n' \
    >"${evidence}/native-affected.stdout"
else
  "${PROJECT_ROOT}/scripts/test-g3c4-generator.sh" \
    "${evidence}/native-regression" \
    >"${evidence}/native-affected.stdout" 2>&1
fi
cp "${temporary_dir}/compile.log" "${evidence}/compile.log"
cp "${temporary_dir}/runtime.stdout" "${evidence}/runtime.stdout"
cp "${temporary_dir}/runtime-repeat.stdout" "${evidence}/runtime-repeat.stdout"
cp "${temporary_dir}/allocation-compile.log" "${evidence}/allocation-compile.log"
cp "${temporary_dir}/allocation.stdout" "${evidence}/allocation.stdout"
cp "${temporary_dir}/publication-compile.log" "${evidence}/publication-compile.log"
cp "${temporary_dir}/publication.stdout" "${evidence}/publication.stdout"
cp "${temporary_dir}/retention-compile.log" "${evidence}/retention-compile.log"
cp "${temporary_dir}"/retention-*.std{out,err} "${evidence}/"
cp "${temporary_dir}/failstop-compile.log" "${evidence}/failstop-compile.log"
cp "${temporary_dir}"/failstop-*.std{out,err} "${evidence}/"
cp "${PROJECT_ROOT}/native/g3c4_call_entry_source_closure.txt" \
  "${evidence}/source-closure.txt"
sha256sum "${temporary_dir}/g3c4_call_entry_fault_extension.esk" \
  "${temporary_dir}/cxx-wrap" >"${evidence}/generated-test-sha256.txt"
while IFS= read -r source; do
  printf '%s\t%s\n' "$(sha256sum "${PROJECT_ROOT}/${source}" | awk '{print $1}')" \
    "${source}"
done <"${PROJECT_ROOT}/native/g3c4_call_entry_source_closure.txt" \
  >"${evidence}/source-sha256.tsv"
for artifact in "${temporary_dir}"/*.o "${temporary_dir}"/*.a \
                "${temporary_dir}/call-entry" \
                "${temporary_dir}/call-entry-failstop" \
                "${temporary_dir}/call-entry-allocation" \
                "${temporary_dir}/call-entry-publication" \
                "${temporary_dir}/call-entry-retention"; do
  printf '%s\t%s\n' "$(sha256sum "${artifact}" | awk '{print $1}')" \
    "$(basename "${artifact}")"
done | LC_ALL=C sort -k2 >"${evidence}/object-sha256.tsv"
{
  source_commit="${G3C4_SOURCE_COMMIT:-$(git -C "${PROJECT_ROOT}" rev-parse HEAD)}"
  source_tree="${G3C4_SOURCE_TREE:-$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')}"
  [[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid source commit"
  [[ "${source_tree}" =~ ^[0-9a-f]{40}$ ]] || die "invalid source tree"
  printf 'commit\t%s\n' "${source_commit}"
  printf 'tree\t%s\n' "${source_tree}"
  printf 'container_image\t%s\n' "${G3C4_CONTAINER_IMAGE:-unreported}"
  printf 'cc\t%s\n' "$("${cc}" --version | head -n 1)"
  printf 'cxx\t%s\n' "$("${cxx}" --version | head -n 1)"
  printf 'eshkol\t%s\n' "$("${runner}" --version 2>&1)"
} >"${evidence}/environment.tsv"
if [[ -d "${evidence}/native-regression" ]]; then
  (cd "${evidence}" && find native-regression -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum) \
    >"${evidence}/native-regression-sha256.txt"
else
  printf 'SKIPPED\n' >"${evidence}/native-regression-sha256.txt"
fi
sha256sum "${PROJECT_ROOT}/scripts/test-g3c4-call-entry.sh" \
  "${PROJECT_ROOT}/scripts/check-g3c4-call-entry.py" \
  >"${evidence}/runner-sha256.txt"
(cd "${evidence}" && sha256sum \
  allocation-compile.log allocation.stdout compile.log environment.tsv \
  failstop-abort.stderr failstop-abort.stdout \
  failstop-compile.log failstop-finish.stderr failstop-finish.stdout \
  object-sha256.tsv publication-compile.log publication.stdout \
  retention-1024.stderr retention-1024.stdout retention-8192.stderr \
  generated-test-sha256.txt native-affected.stdout \
  native-regression-sha256.txt retention-8192.stdout \
  retention-compile.log runner-sha256.txt \
  runtime-repeat.stdout runtime.stdout source-closure.txt \
  source-sha256.tsv >evidence-manifest.tsv)
sha256sum "${evidence}/evidence-manifest.tsv" \
  >"${evidence}/evidence-manifest.sha256"
cat "${evidence}/runtime.stdout"
cat "${evidence}/allocation.stdout"
cat "${evidence}/publication.stdout"
cat "${evidence}/retention-1024.stdout"
cat "${evidence}/retention-8192.stdout"
printf 'G3-C4 call entry evidence: %s\n' "${evidence}"
