#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar flock mktemp objcopy python3 sha256sum timeout; do
  require_command "${command}"
done

evidence="${DATA4_Q_T2_D1_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/eshkol-data4-q-t2-d1.XXXXXX")}"
if [[ -d "${evidence}" && -n "$(ls -A -- "${evidence}")" ]]; then
  die "evidence directory must be empty: ${evidence}"
fi
mkdir -p -- "${evidence}/build"
evidence="$(realpath -- "${evidence}")"

# The compiler and DATA4 archive can each use substantial memory.
exec 9>/tmp/eshkol-transformer-heavy-build.lock
flock -x 9

runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
export CC="${cc}"
export CXX="$(tsv_value "${provenance}" cxx_path)"
bash "${PROJECT_ROOT}/scripts/build-data4-pipeline.sh" \
  "${evidence}/build/pipeline" normal >"${evidence}/archive.log" 2>&1

native_objects=()
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell p1_identity; do
  object="${evidence}/build/${source}.o"
  extra_flags=()
  if [[ "${source}" == p1_identity ]]; then
    extra_flags=(-DET_P1_TRUSTED_BUILD=1)
  fi
  "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
    "${extra_flags[@]}" \
    -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
    -I "${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/native/${source}.c" -o "${object}"
  native_objects+=("${object}")
done

# Interpose on a copy of the audited bridge object for fault and lifetime checks.
ar p "${evidence}/build/pipeline/libeshkol_transformer_data4_pipeline.a" \
  data4_pipeline_eshkol_bridge.o >"${evidence}/build/bridge.o"
objcopy \
  --redefine-sym et_data4_aot_session_create=et_data4_quality_test_real_create \
  --redefine-sym et_data4_aot_session_destroy=et_data4_quality_test_real_destroy \
  --redefine-sym et_data4_aot_copy_report=et_data4_quality_test_real_copy_report \
  --redefine-sym et_data4_aot_copy_report_sha256=et_data4_quality_test_real_copy_digest \
  "${evidence}/build/bridge.o"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/tests/data4/quality_corpus_faults.c" \
  -o "${evidence}/build/faults.o"
native_objects+=("${evidence}/build/bridge.o" "${evidence}/build/faults.o")

timeout --foreground --kill-after=5s 300s "${runner}" \
  --strict-types --optimize 0 --no-stdlib \
  -I "${PROJECT_ROOT}/internal/p1/lib" \
  -I "${PROJECT_ROOT}/internal/c1/lib" \
  -I "${PROJECT_ROOT}/internal/t2/lib" \
  -I "${PROJECT_ROOT}/internal/t1/lib" \
  -I "${PROJECT_ROOT}/internal/data4/lib" \
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
  -I "${PROJECT_ROOT}/native" \
  -L "${evidence}/build/pipeline" \
  --lib eshkol_transformer_data4_pipeline \
  "${PROJECT_ROOT}/tests/data4/quality_corpus_runtime.esk" \
  "${native_objects[@]}" -o "${evidence}/build/quality-corpus" \
  >"${evidence}/compile.stdout" 2>"${evidence}/compile.stderr"

for mode in mixed reordered all-rejected empty invalid; do
  mkdir -- "${evidence}/${mode}"
  "${evidence}/build/quality-corpus" "${mode}" "${evidence}/${mode}" \
    >"${evidence}/${mode}.stdout" 2>"${evidence}/${mode}.stderr"
  [[ ! -s "${evidence}/${mode}.stderr" ]] || die "unexpected stderr in ${mode}"
done
[[ -z "$(ls -A -- "${evidence}/invalid")" ]] || die "invalid input published D1 data"
[[ "$(cat "${evidence}/invalid.stdout")" == \
   "DATA4-Q invalid inputs and tampered staging rejected before publication" ]] || \
  die "wrong negative result"
for mode in mixed reordered all-rejected empty; do
  [[ -s "${evidence}/${mode}/manifest.etm" ]] || die "missing D1 manifest for ${mode}"
done
for mode in all-rejected empty; do
  [[ "$(ls -A -- "${evidence}/${mode}")" == manifest.etm ]] || \
    die "zero-token corpus published extra files"
done

python3 - "${evidence}" <<'PYTHON'
import hashlib
from pathlib import Path
import sys

evidence = Path(sys.argv[1])
expected = {
    "mixed": [
        ("invalid-utf8", bytes([237, 160, 128, 97, 98]), "rejected_utf8", ""),
        ("source.second", b"apple", "kept", ""),
        ("too-short", b"ab", "rejected_length", ""),
        ("same", b"apple", "duplicate_exact", "source.second"),
        ("near", b"pleap", "duplicate_near", "source.second"),
        ("low", b"!?!?", "rejected_quality", ""),
        ("source.last", bytes([99, 97, 102, 101, 204, 129, 13, 10]), "kept", ""),
    ],
    "reordered": [
        ("near", b"pleap", "kept", ""),
        ("same", b"pleap", "duplicate_exact", "near"),
        ("source.second", b"apple", "duplicate_near", "near"),
    ],
    "all-rejected": [
        ("too-short", b"a", "rejected_length", ""),
        ("low", b"????", "rejected_quality", ""),
    ],
    "empty": [],
}
for mode, documents in expected.items():
    report = (evidence / f"{mode}.stdout").read_bytes()
    assert report.startswith(b"format\teshkol-data4-corpus-pipeline-report\nversion\t1\t0\n"), mode
    prefix, checksum = report.rsplit(b"checksum\t", 1)
    assert checksum == hashlib.sha256(prefix).hexdigest().encode("ascii") + b"\n", mode
    rows = [line.split("\t") for line in report.decode("ascii").splitlines()
            if line.startswith("document\t")]
    assert len(rows) == len(documents), mode
    for ordinal, (row, (source, body, disposition, match)) in enumerate(zip(rows, documents)):
        assert row[:4] == ["document", str(ordinal), source, str(len(body))], row
        assert row[5:8] == [hashlib.sha256(body).hexdigest(), disposition, match], row
        assert row[8:] == (["4", "4"] if disposition == "duplicate_near" else ["0", "0"]), row
PYTHON

(
  cd -- "${evidence}"
  sha256sum build/quality-corpus mixed/* reordered/* all-rejected/* empty/* \
    mixed.stdout reordered.stdout all-rejected.stdout empty.stdout invalid.stdout > SHA256SUMS
)
"${evidence}/build/quality-corpus" existing "${evidence}/mixed" \
  >"${evidence}/existing.stdout" 2>"${evidence}/existing.stderr"
[[ ! -s "${evidence}/existing.stdout" && ! -s "${evidence}/existing.stderr" ]] || \
  die "unexpected output for existing D1 target"
(
  cd -- "${evidence}"
  sha256sum -c SHA256SUMS
) >"${evidence}/verify.log"
printf 'DATA4-Q corpus adapter passed; evidence: %s\n' "${evidence}"
