#!/usr/bin/env bash
# Run from the gate-only descendant against a separate, clean f666f7a checkout.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 2 ]] || { echo "usage: $0 EXACT_SOURCE_CHECKOUT FRESH_OUTPUT_DIRECTORY" >&2; exit 2; }
gate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_root="$(realpath -e -- "$1")"
out="$(realpath -m -- "$2")"
python3 -B - "$2" <<'PY'
import os, pathlib, sys
path = pathlib.Path(os.path.abspath(sys.argv[1]))
for ancestor in (path, *path.parents):
    if ancestor.is_symlink():
        raise SystemExit(f"output path traverses a symlink: {ancestor}")
PY
[[ "${out}" != "${source_root}" && "${out}" != "${source_root}/"* &&
   "${out}" != "${gate_root}" && "${out}" != "${gate_root}/"* ]] || {
  echo "output must be outside both checkouts" >&2; exit 1;
}
candidate=f666f7a6a29a96f63693126e130a5199337cfd8b
candidate_tree=af52d5fd6b30ff35361d9893c665c05667235556
review_dir=/home/gabe/.codex/evidence/eshkol-transformer/e3-m2-physical-capacity-independent-review-f666f7a6-20261011-a
review_sha=2295b9d6fc2162f39f9029e2150664e29b7e0e29eeb7b5e21a38a657c4acd7b2

[[ "${gate_root}" != "${source_root}" ]] || { echo "gate and source checkouts must differ" >&2; exit 1; }
[[ "$(git -C "${source_root}" rev-parse HEAD)" == "${candidate}" ]] || {
  echo "source checkout is not the reviewed candidate" >&2; exit 1;
}
[[ "$(git -C "${source_root}" rev-parse HEAD^{tree})" == "${candidate_tree}" ]] || {
  echo "candidate tree mismatch" >&2; exit 1;
}
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] || {
  echo "candidate checkout is dirty" >&2; exit 1;
}
[[ -z "$(git -C "${gate_root}" status --porcelain --untracked-files=all)" ]] || {
  echo "gate checkout is dirty" >&2; exit 1;
}
gate_commit="$(git -C "${gate_root}" rev-parse HEAD)"
gate_tree="$(git -C "${gate_root}" rev-parse HEAD^{tree})"
base_gate=313a38dbc2d2cfb8eba7c4357885798234291d45
[[ "$(git -C "${gate_root}" rev-parse HEAD^)" == "${base_gate}" &&
   "$(git -C "${gate_root}" rev-parse "${base_gate}^")" == "${candidate}" ]] || {
  echo "gate repair must descend directly from the pinned gate and candidate" >&2; exit 1;
}
expected_changes=$'docs/ROADMAP.md\ndocs/e3/E3_M2_PHYSICAL_CAPACITY_ONE_USE_GATE.md\nscripts/test-e3-m2-physical-capacity-one-use.sh\ntests/e3_private/numerical_physical_capacity_runtime.esk\ntests/e3_private/parse_numerical_observation.py\ntests/e3_private/test_numerical_tuple.py'
[[ "$(git -C "${gate_root}" diff --name-only "${candidate}" HEAD)" == "${expected_changes}" ]] || {
  echo "gate descendant changed files outside the reviewed test-only recipe" >&2; exit 1;
}
[[ "$(sha256sum "${review_dir}/SHA256SUMS-REVIEW" | cut -d' ' -f1)" == "${review_sha}" ]] || {
  echo "independent source review seal mismatch" >&2; exit 1;
}
(cd "${review_dir}" && sha256sum -c SHA256SUMS-REVIEW >/dev/null)
[[ ! -e "${out}" && ! -L "${out}" ]] || {
  echo "output must be a fresh path" >&2; exit 1;
}
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || {
  echo "Q0_PYTHON must be an absolute executable path" >&2; exit 1;
}
[[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" != 1 ]] || {
  echo "unsupported host override is forbidden for this proof" >&2; exit 1;
}
source "${source_root}/scripts/common.sh"
verify_toolchain
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || {
  echo "Q0 Python/PyTorch version mismatch: ${versions//$'\n'/ }" >&2; exit 1;
}

mkdir -p -- "${out}"
finish() {
  local status=$?
  trap - EXIT
  if [[ "${status}" == 0 ]]; then printf 'PASS: compiled private M2 value kill\n' >"${out}/GATE-STATUS"
  else printf 'HOLD: gate failed at exit %s\n' "${status}" >"${out}/GATE-STATUS"; fi
  if ! python3 -B - "${out}" <<'PY'
import hashlib, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
paths = []
for directory, subdirs, files in os.walk(root, followlinks=False):
    for name in subdirs + files:
        path = pathlib.Path(directory, name)
        if path.is_symlink():
            raise SystemExit(f"gate output contains symlink: {path}")
    paths += [pathlib.Path(directory, name) for name in files
              if name not in {"SHA256SUMS-GATE", "SHA256SUMS-GATE.partial"}]
with (root / "SHA256SUMS-GATE.partial").open("x", encoding="ascii") as seal:
    for path in sorted(paths):
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        seal.write(f"{digest}  {path.relative_to(root)}\n")
os.replace(root / "SHA256SUMS-GATE.partial", root / "SHA256SUMS-GATE")
PY
  then
    printf 'HOLD: evidence seal failed\n' >"${out}/GATE-STATUS"
    rm -f -- "${out}/SHA256SUMS-GATE" "${out}/SHA256SUMS-GATE.partial"
    status=1
  fi
  exit "${status}"
}
trap finish EXIT
cd "${source_root}"
{
  printf 'candidate_commit\t%s\ncandidate_tree\t%s\n' "${candidate}" "${candidate_tree}"
  printf 'gate_commit\t%s\ngate_tree\t%s\n' "${gate_commit}" "${gate_tree}"
  printf 'review_seal_sha256\t%s\nq0_python\t%s\n' "${review_sha}" "${q0_python}"
  printf 'q0_versions\t%s\n' "${versions//$'\n'/,}"
  printf 'eshkol_runner\t%s\n' "$(eshkol_build_dir)/eshkol-run"
} >"${out}/INPUTS.tsv"
cp -- "${review_dir}/SHA256SUMS-REVIEW" "${out}/SOURCE-REVIEW-SHA256SUMS"
cp -- "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "${out}/TOOLCHAIN-PROVENANCE.tsv"
cp -- "${source_root}/toolchain/eshkol.lock" "${out}/TOOLCHAIN-LOCK.tsv"
sha256sum "$(eshkol_build_dir)/eshkol-run" >"${out}/ESHKOL-RUN.sha256"

for variant in a b; do
  "${q0_python}" -B -m tests.e3_reference.corpus --output "${out}/corpus-${variant}" \
    >"${out}/corpus-${variant}.stdout" 2>"${out}/corpus-${variant}.stderr"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
    -B -m tests.e3_reference.reference \
    --corpus-reference "${out}/corpus-${variant}/corpus-reference.json" \
    --output "${out}/reference-${variant}.json" \
    >"${out}/reference-${variant}.stdout" 2>"${out}/reference-${variant}.stderr"
done
diff -r "${out}/corpus-a" "${out}/corpus-b" >"${out}/corpus-repeat.diff"
cmp "${out}/reference-a.json" "${out}/reference-b.json"

prefix="${source_root}/tests/e3_private/numerical_package"
for variant in a b physical-capacity; do
  mode="${variant}"
  if [[ "${variant}" == a || "${variant}" == b ]]; then mode=normal; fi
  "${source_root}/scripts/build-e3-numerical-test.sh" "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" 2>"${out}/package-${variant}.stderr"
  artifact="${out}/package-${variant}/e3_numerical_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects \
      undefined_symbols:undefined; do
    cmp "${prefix}_${pair%%:*}.txt" "${artifact}/${pair#*:}.txt"
  done
  cmp "${prefix}_archive_members.txt" \
    <(ar t "${out}/package-${variant}/libeshkol_transformer_e3_numerical.a")
  grep -F 'E3ROW' "${artifact}/strings.txt" >/dev/null
done
cmp "${out}/package-a/e3_numerical_package.o" "${out}/package-b/e3_numerical_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_numerical.a" \
    "${out}/package-b/libeshkol_transformer_e3_numerical.a"
cmp "${out}/package-a/e3_numerical_package.o.evidence/private.d" \
    "${out}/package-b/e3_numerical_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_numerical_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_numerical_package.o.evidence/native-depfiles" \
        >"${out}/native-depfiles-repeat.diff"

runner="$(eshkol_build_dir)/eshkol-run"
cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
for variant in a b physical-capacity; do
  runtime_source="${source_root}/tests/e3_private/numerical_runtime.esk"
  if [[ "${variant}" == physical-capacity ]]; then
    runtime_source="${gate_root}/tests/e3_private/numerical_physical_capacity_runtime.esk"
  fi
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${source_root}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_numerical \
      "${runtime_source}" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" 2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"
for variant in a b physical-capacity; do
  ESHKOL_ARENA_POISON=0 timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${out}/runtime-${variant}" packed-single "${out}/corpus-a/packed-single" \
    >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  grep -Fx 'E3-NUMERICAL-RUNTIME-PASS' "${out}/run-${variant}.stdout" >/dev/null
  [[ ! -s "${out}/run-${variant}.stderr" ]]
  parser=("${source_root}/tests/e3_private/parse_numerical_observation.py")
  if [[ "${variant}" == physical-capacity ]]; then
    parser=("${gate_root}/tests/e3_private/parse_numerical_observation.py" --physical-capacity)
  fi
  PYTHONPATH="${source_root}" python3 -B "${parser[0]}" \
    --stdout "${out}/run-${variant}.stdout" --batch-count 3 \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv" "${parser[@]:1}" \
    >"${out}/parse-${variant}.stdout" 2>"${out}/parse-${variant}.stderr"
  [[ ! -s "${out}/parse-${variant}.stderr" ]]
done
cmp "${out}/run-a.stdout" "${out}/run-b.stdout"
cmp "${out}/observed-a.json" "${out}/observed-b.json"
cmp "${out}/roles-a.tsv" "${out}/roles-b.tsv"
cmp "${out}/roles-a.tsv" "${out}/roles-physical-capacity.tsv"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
  -B -m tests.e3_native_parity.verify_observation \
  --reference "${out}/reference-a.json" --case packed-single \
  --observed "${out}/observed-a.json" --roles "${out}/roles-a.tsv" \
  --transcript "${out}/transcript-a.json" \
  >"${out}/verify-a.stdout" 2>"${out}/verify-a.stderr"
[[ ! -s "${out}/verify-a.stderr" ]]
python3 -B - "${out}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
normal = json.loads((root / "observed-a.json").read_bytes())
mutant = json.loads((root / "observed-physical-capacity.json").read_bytes())
assert len(normal["batches"]) == len(mutant["batches"]) == 3
assert normal["batches"][:2] == mutant["batches"][:2]
assert [row["mask"] for row in mutant["batches"]] == [[True, True], [True, True], [True, False]]
for left, right in zip(normal["batches"], mutant["batches"]):
    for field in ("inputs", "targets", "mask", "ce_f32", "logits_f32"):
        assert left[field] == right[field], field
assert normal["batches"][2]["reduction_f32"][1] == "3f800000"
assert mutant["batches"][2]["reduction_f32"][1] == "40000000"
assert normal["final"]["counters"] == [5, 3]
assert mutant["final"]["counters"] == [6, 3]
PY
if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
  -B -m tests.e3_native_parity.verify_observation \
  --reference "${out}/reference-a.json" --case packed-single \
  --observed "${out}/observed-physical-capacity.json" \
  --roles "${out}/roles-physical-capacity.tsv" \
  --transcript "${out}/transcript-physical-capacity.json" \
  >"${out}/verify-physical-capacity.stdout" 2>"${out}/verify-physical-capacity.stderr"; then
  echo "physical-capacity mutant survived the independent comparator" >&2; exit 1
fi
[[ ! -e "${out}/transcript-physical-capacity.json" ]]
[[ "$(tail -n 1 "${out}/verify-physical-capacity.stderr")" == \
  'tests.e3_reference.transcript.TranscriptError: batches[2].reduction_f32: exact rational reduction mismatch' ]]
[[ "$(git -C "${source_root}" rev-parse HEAD)" == "${candidate}" &&
   "$(git -C "${source_root}" rev-parse HEAD^{tree})" == "${candidate_tree}" &&
   -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" &&
   "$(git -C "${gate_root}" rev-parse HEAD)" == "${gate_commit}" &&
   "$(git -C "${gate_root}" rev-parse HEAD^{tree})" == "${gate_tree}" &&
   -z "$(git -C "${gate_root}" status --porcelain --untracked-files=all)" ]] || {
  echo "source or gate checkout drifted during proof" >&2; exit 1;
}
printf 'normal AOT and first physical-capacity value kill passed\n'
