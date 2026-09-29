#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar cmp docker git mv nm objcopy python3 readlink rg sha256sum; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"

fixed="${TR3_LINKED_COMPILER_EVIDENCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-build}"
compiler_source="${TR3_LINKED_COMPILER_SOURCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-src}"
runtime="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-${PROJECT_ROOT}/.deps}"
oracle="${TR3_TRAIN_ORACLE_PYTHON:-${M3_ORACLE_PYTHON:-}}"
[[ "${oracle}" == /* && -x "${oracle}" ]] || die "pinned PyTorch oracle path required"
pin="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
[[ -f "${fixed}/eshkol-run" && -e "${compiler_source}/.git" &&
   -f "${runtime}/eshkol-build/libeshkol-runtime.a" ]] ||
  die "fe9 toolchain unavailable; set TR3_LINKED_COMPILER_EVIDENCE_DIR, TR3_LINKED_COMPILER_SOURCE_DIR, and TR3_LEASE_RUNTIME_CANDIDATE_DIR"
case "${TR3_PUBLIC_INSTALLED_RUNTIME_PIN:-fe9}" in
  fe9)
    compiler_commit=fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9
    runner_name=eshkol-run
    compiler_tree=66c21f7ec19b1b4a42199fa30ed8e0e9727021bf
    runner_sha256=7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa
    runtime_build_dir="${runtime}/eshkol-build"
    runtime_sha256=32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e
    source_manifest=native/tr3_public_installed_fe9_source_closure.txt
    undefined_manifest=native/tr3_public_installed_fe9_undefined_symbols.txt
    [[ "$(git -C "${compiler_source}" rev-parse 'HEAD^{tree}')" == "${compiler_tree}" ]] || \
      die "fe9 compiler tree changed"
    [[ -z "$(git -C "${compiler_source}" status --porcelain --untracked-files=all)" ]] || \
      die "fe9 compiler source changed"
    [[ "$(tsv_value "${runtime_build_dir}/eshkol-transformer-provenance.tsv" eshkol_commit)" == \
       "${compiler_commit}" ]] || die "fe9 runtime provenance changed"
    ;;
  81298) die "public metrics-ref requires the fe9 true-f32 runtime" ;;
  *) die "unsupported TR3 public installed runtime pin" ;;
esac
image="$(tsv_value "${pin}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${pin}" container_digest)" ]] || die "pinned image changed"
[[ "$(sha256sum "${fixed}/${runner_name}" | awk '{print $1}')" == \
   "${runner_sha256}" ]] || \
  die "reviewed compiler changed"
[[ "$(git -C "${compiler_source}" rev-parse HEAD)" == \
   "${compiler_commit}" ]] || \
  die "reviewed compiler source changed"
git -C "${compiler_source}" diff --quiet HEAD -- \
  inc/eshkol/eshkol.h inc/eshkol/core/runtime.h lib/core/arena_memory.h || \
  die "reviewed compiler headers changed"
[[ "$(sha256sum "${runtime_build_dir}/libeshkol-runtime.a" | awk '{print $1}')" == \
   "${runtime_sha256}" ]] || \
  die "pinned runtime archive changed"

evidence="${TR3_PUBLIC_INSTALLED_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-public-installed.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence must be outside checkout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.tr3_public_installed.test_overfit_source_contract \
  tests.tr3_public_installed.test_overfit_checker \
  tests.tr3_public_installed.test_heldout_checker \
  >"${evidence}/learning-contract.stdout" \
  2>"${evidence}/learning-contract.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus/step-two"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.e3_reference.corpus \
  --output "${evidence}/corpus/e3"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus/step-three" --rows three
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus/overfit-one" --rows one
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3_public_installed.prepare_heldout \
  --output "${evidence}/corpus/heldout-pair"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  PYTHONDONTWRITEBYTECODE=1 "${oracle}" \
  -m tests.tr3_public_installed.check_unequal_reference \
  >"${evidence}/unequal-reference.stdout" \
  2>"${evidence}/unequal-reference.stderr"
rg -Fx 'TR3-TRAIN-UNEQUAL-REFERENCE-PASS weights=4,3 loss-bits=1085208078' \
  "${evidence}/unequal-reference.stdout" >/dev/null

docker run --rm --network none \
  -e TR3_COMPILER_RUNNER="/fixed/${runner_name}" \
  -v "${PROJECT_ROOT}:/workspace:ro" -v "${fixed}:/fixed:ro" \
  -v "${compiler_source}:/fixed-source:ro" \
  -v "${runtime_build_dir}:/candidate/eshkol-build-canonical:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/native /out/cache
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/source-compile.time \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I internal/e3/lib \
    -I src -I lib -L /candidate/eshkol-build-canonical \
    --shared-lib --dump-ir --emit-depfile /out/private.d \
    native/tr3_public_installed_root.esk -o /out/private \
    > /out/source-compile.stdout 2> /out/source-compile.stderr
test -s /out/private.ll
clang-21 -fPIC -c -x ir /out/private.ll -o /out/private.o
objcopy --redefine-syms=native/tr3_public_installed_private_renames.txt \
  /out/private.o

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
       -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
       -fexcess-precision=standard -frounding-math -fno-fast-math
       -I include -I native -I src)
compile() {
  local source=$1 object=$2
  shift 2
  clang-21 "${flags[@]}" "$@" -MMD -MF "/out/native/${object}.d" \
    -c "${source}" -o "/out/native/${object}"
}
compile native/data_io.c data_io.o
compile native/checkpoint_io.c checkpoint_io.o
compile native/kernel_abi.c kernel_abi.o
compile src/eshkol_transformer/m3_i64_integration.c m3_i64_integration.o
compile native/t1_i64_shell.c t1_i64_shell.o
compile native/e3_d2_native.c e3_d2_native.o -DET_TR3_C_D2_RESTORE
compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration.o \
  -DET_I2_PRIVATE_OWNED_CLONE_MATCH
compile src/eshkol_transformer/e3_frame.c e3_frame.o
compile native/e3_evaluation_metrics_provider.c e3_evaluation_metrics_provider.o
compile native/e3_diagnostic_destinations.c e3_diagnostic_destinations.o
compile native/n2_primitives_provider.c n2_primitives_provider.o
compile native/n3k_primitives_provider.c n3k_primitives_provider.o
compile native/a2_attention_provider.c a2_attention_provider.o
compile native/indexed_cross_entropy.c indexed_cross_entropy.o
compile native/l3s_masked_objective_provider.c l3s_masked_objective_provider.o
compile native/tr3b_objective_bridge.c tr3b_objective_bridge.o
compile native/o2_optimizer.c o2_optimizer.o \
  -DET_TR3_O2_STEP_CLEAR_NATIVE -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_O2_RESTORE_NATIVE
compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1
compile native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -DET_C2_I2_MODEL_COPY -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE
compile native/o2_wave2_package_bridge.c o2_bridge.o \
  -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -DET_TR3_O2_STEP_CLEAR_BRIDGE
compile native/c2_x1_canonical.c c2_x1_canonical.o
compile native/k2_capabilities.c k2_capabilities.o -DET_C2_CARRIER_FACTORIES
compile native/c2_checkpoint_codec.c c2_checkpoint_codec.o
compile native/c2_checkpoint_format.c c2_checkpoint_format.o
compile native/c2_checkpoint_save_bridge.c c2_checkpoint_save_bridge.o
compile native/c2_checkpoint_core.c c2_checkpoint_core.o
compile native/c2_checkpoint_reader.c c2_checkpoint_reader.o
compile native/c2_checkpoint_load_bridge.c c2_checkpoint_load_bridge.o
compile native/c2_checkpoint_inspect_bridge.c c2_checkpoint_inspect_bridge.o
compile native/tr3_c_restore_bindings.c tr3_c_restore_bindings.o \
  -DET_TR3_C_RESTORE_BINDINGS -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE -DET_TR3_C_O2_RESTORE_NATIVE
compile native/tr3_public_step_metrics.c tr3_public_step_metrics.o
# The E3 objects are exact supersets of these installed objects.  Compile
# comparison witnesses outside the link manifest, and reject either co-link.
mkdir -p /out/superset
clang-21 "${flags[@]}" -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -c src/eshkol_transformer/m3t_f32_integration.c \
  -o /out/superset/m3t_f32_integration.o
clang-21 "${flags[@]}" -c src/eshkol_transformer/m3_model.c \
  -o /out/superset/m3_model.o
for pair in "m3_model e3_frame" "m3t_f32_integration m3_call_f32_integration"; do
  read -r base replacement <<< "${pair}"
  nm -g --defined-only --format=posix "/out/superset/${base}.o" | \
    awk "{print \$1}" | LC_ALL=C sort -u > "/out/superset/${base}.symbols"
  nm -g --defined-only --format=posix "/out/native/${replacement}.o" | \
    awk "{print \$1}" | LC_ALL=C sort -u > "/out/superset/${replacement}.symbols"
  test ! -s <(comm -23 "/out/superset/${base}.symbols" \
    "/out/superset/${replacement}.symbols")
  test ! -e "/out/native/${base}.o"
done
test ! -e /out/native/d2_native.o
test -e /out/native/e3_d2_native.o
clang-21 "${flags[@]}" -I native \
  tests/tr3_public_installed/test_step_metrics.c \
  /out/native/tr3_public_step_metrics.o -lm -o /out/step_metrics_test
/out/step_metrics_test > /out/step-metrics.stdout 2> /out/step-metrics.stderr
grep -Fx TR3-PUBLIC-STEP-METRICS-KERNEL-PASS /out/step-metrics.stdout >/dev/null
test ! -s /out/step-metrics.stderr
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -I /fixed-source/inc -I native -MMD -MF /out/native/e1b_bridge.o.d \
  -c native/e1b_error_consumer_bridge.c -o /out/native/e1b_bridge.o
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -I /fixed-source/inc -I native -MMD -MF /out/native/tr3_public_installed_bridge.o.d \
  -c native/tr3_public_installed_bridge.c \
  -o /out/native/tr3_public_installed_bridge.o

clang++-21 -r /out/private.o /out/native/*.o -o /out/combined.raw.o
nm -g --defined-only --format=posix /out/combined.raw.o | \
  awk "{print \$1}" | LC_ALL=C sort -u > /out/raw-defined.txt
awk "NR == FNR {allowed[\$1] = 1; next} !(\$1 in allowed) {print \$1}" \
  native/tr3_public_installed_exports.txt /out/raw-defined.txt \
  > /out/localize-symbols.txt
cp /out/combined.raw.o /out/combined.o
objcopy --localize-symbols=/out/localize-symbols.txt /out/combined.o
nm -g --defined-only --format=posix /out/combined.o | \
  awk "{print \$1}" | LC_ALL=C sort -u > /out/global-defined.txt
cmp native/tr3_public_installed_exports.txt /out/global-defined.txt
nm -u --format=posix /out/combined.o | awk "{print \$1}" | \
  LC_ALL=C sort -u > /out/undefined-symbols.txt
ar rcsD /out/libeshkol_transformer_tr3_public_installed.a /out/combined.o
test "$(ar t /out/libeshkol_transformer_tr3_public_installed.a)" = combined.o

mkdir -p /out/corpus/installed-d1
mkdir -p /out/facades/transformer
while IFS= read -r facade; do
  test -f "lib/${facade}" && test ! -L "lib/${facade}"
  cp "lib/${facade}" "/out/facades/${facade}"
done < native/tr3_public_installed_facades.txt
export ESHKOL_LIB_DIR=/out/facades
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades --compile-only --emit-depfile /out/caller.d \
    tests/tr3_public_installed/runtime.esk -o /out/caller.o \
    > /out/caller-check.stdout 2> /out/caller-check.stderr
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_public_installed \
    tests/tr3_public_installed/runtime.esk \
    -o /out/caller > /out/caller-compile.stdout 2> /out/caller-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/caller /out/corpus/installed-d1 \
  > /out/caller.stdout 2> /out/caller.stderr
grep -Fx TR3-PUBLIC-OPERANDS-PASS /out/caller.stdout >/dev/null
test ! -s /out/caller.stderr
cmp /out/corpus/installed-d1/trainer-a.c2 \
    /out/corpus/installed-d1/trainer-b.c2

# Keep the legacy test-only producer for the metrics-ref bit boundary witness.
clang-21 "${flags[@]}" -I /fixed-source/inc \
  -c tests/tr3_metrics/linked_producer.c -o /out/linked_producer.o
ar rcsD /out/libeshkol_transformer_tr3_metrics_test.a \
  /out/combined.raw.o /out/linked_producer.o
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_metrics/linked_runtime.esk -o /out/linked_metrics_caller \
    > /out/linked-metrics-compile.stdout 2> /out/linked-metrics-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/linked_metrics_caller \
  > /out/linked-metrics.stdout 2> /out/linked-metrics.stderr
grep -Fx TR3-PUBLIC-METRICS-LINKED-PASS /out/linked-metrics.stdout >/dev/null
test ! -s /out/linked-metrics.stderr

# The test-only f32 inspector reads actual public step results from this same
# raw aggregate. The ordinary installed caller above links the localized one.
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_public_installed/step_runtime.esk -o /out/step_caller \
    > /out/step-compile.stdout 2> /out/step-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/step_caller /out/corpus/step-two /out/corpus/step-three \
  > /out/step.stdout 2> /out/step.stderr
grep -Fx "TR3-PUBLIC-STEP-AOT-PASS loss-bits=1085403699 weight-bits=1077936128" \
  /out/step.stdout >/dev/null
test ! -s /out/step.stderr

# A localized installed aggregate plus one read-only test bit inspector. The
# same caller runs in twelve fresh OS processes across A=1,2,3; no private
# trainer, C2, D2, or optimizer entry is linked into the caller.
clang-21 "${flags[@]}" -I /fixed-source/inc \
  -c tests/tr3_public_installed/resume_bits.c -o /out/resume_bits.o
ar rcsD /out/libeshkol_transformer_tr3_public_resume_test.a \
  /out/combined.o /out/resume_bits.o
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_public_resume_test \
    tests/tr3_public_installed/fresh_resume.esk -o /out/fresh_resume \
    > /out/fresh-resume-compile.stdout \
    2> /out/fresh-resume-compile.stderr
for accumulation in 1 2 3; do
  directory="/out/corpus/resume-${accumulation}"
  cp -a /out/corpus/step-two "${directory}"
  for mode in baseline train-baseline; do
    ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
      --kill-after=5s 120s \
      /out/fresh_resume "${directory}" "${mode}" "${accumulation}" \
      > "/out/fresh-${accumulation}-${mode}.stdout" \
      2> "/out/fresh-${accumulation}-${mode}.stderr"
    grep -Fx "TR3-PUBLIC-FRESH-RESUME-PASS ${mode} A=${accumulation}" \
      "/out/fresh-${accumulation}-${mode}.stdout" >/dev/null
    test ! -s "/out/fresh-${accumulation}-${mode}.stderr"
  done
  test ! -e "${directory}/interrupt-request"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s \
    /out/fresh_resume "${directory}" producer "${accumulation}" \
    > "/out/fresh-${accumulation}-producer.stdout" \
    2> "/out/fresh-${accumulation}-producer.stderr" &
  producer_pid=$!
  for ((attempt=0; attempt<1000; attempt++)); do
    if test -f "${directory}/interrupt-ready"; then break; fi
    if ! kill -0 "${producer_pid}" 2>/dev/null; then break; fi
    sleep 0.01
  done
  if ! test -f "${directory}/interrupt-ready"; then
    kill "${producer_pid}" 2>/dev/null || true
    wait "${producer_pid}" 2>/dev/null || true
    echo "producer did not publish ready after committed update" >&2
    exit 1
  fi
  test ! -e "${directory}/interrupt-ack"
  test ! -e "${directory}/interrupt-request.tmp"
  printf "1" > "${directory}/interrupt-request.tmp"
  mv -- "${directory}/interrupt-request.tmp" "${directory}/interrupt-request"
  wait "${producer_pid}"
  cmp -s <(printf "\001") "${directory}/interrupt-ready"
  cmp -s <(printf "\001") "${directory}/interrupt-ack"
  grep -Fx "TR3-PUBLIC-FRESH-RESUME-PASS producer A=${accumulation}" \
    "/out/fresh-${accumulation}-producer.stdout" >/dev/null
  test ! -s "/out/fresh-${accumulation}-producer.stderr"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM \
    --kill-after=5s 120s \
    /out/fresh_resume "${directory}" receiver "${accumulation}" \
    > "/out/fresh-${accumulation}-receiver.stdout" \
    2> "/out/fresh-${accumulation}-receiver.stderr"
  grep -Fx "TR3-PUBLIC-FRESH-RESUME-PASS receiver A=${accumulation}" \
    "/out/fresh-${accumulation}-receiver.stdout" >/dev/null
  test ! -s "/out/fresh-${accumulation}-receiver.stderr"
  grep -Fx "INTERRUPT-ACK 1" \
    "/out/fresh-${accumulation}-producer.stdout" >/dev/null
  grep "^METRIC " "/out/fresh-${accumulation}-baseline.stdout" \
    > "/out/fresh-${accumulation}-baseline.metrics"
  test "$(wc -l < "/out/fresh-${accumulation}-baseline.metrics")" = 4
  cmp "${directory}/resume-fresh-before-0.c2" \
      "${directory}/resume-fresh-after-0.c2"
  cmp "${directory}/resume-mismatch-before-0.c2" \
      "${directory}/resume-mismatch-after-0.c2"
  cmp "${directory}/resume-baseline-1.c2" "${directory}/resume-k-1.c2"
  cmp "${directory}/resume-baseline-4.c2" \
      "${directory}/resume-train-baseline-4.c2"
  cmp "${directory}/resume-baseline-1.c2" \
      "${directory}/resume-restored-1.c2"
  for ordinal in 2 3 4; do
    cmp "${directory}/resume-baseline-${ordinal}.c2" \
        "${directory}/resume-resumed-${ordinal}.c2"
    ! cmp -s "${directory}/resume-baseline-$((ordinal - 1)).c2" \
        "${directory}/resume-baseline-${ordinal}.c2"
  done
done
python3 - /out <<PY
from pathlib import Path
import sys

evidence = Path(sys.argv[1])
rows = ["accumulation\tupdate\tloss_f32_bits\tmask_weight_f32_bits"]
for accumulation in (1, 2, 3):
    records = [line.split() for line in
               (evidence / f"fresh-{accumulation}-baseline.metrics").read_text().splitlines()]
    assert len(records) == 4
    assert len({record[2] for record in records}) == 4, records
    producer = (evidence / f"fresh-{accumulation}-producer.stdout").read_text().splitlines()
    receiver = (evidence / f"fresh-{accumulation}-receiver.stdout").read_text().splitlines()
    trained = [line.split() for line in producer + receiver if line.startswith("TRAIN ")]
    assert len(trained) == 4, trained
    full = [line.split() for line in
            (evidence / f"fresh-{accumulation}-train-baseline.stdout").read_text().splitlines()
            if line.startswith("FULL ")]
    assert len(full) == 1 and int(full[0][4]) == 4, full
    assert int(full[0][3]) == sum(int(record[4]) for record in records), full
    for update, record in enumerate(records, 1):
        assert record[0] == "METRIC" and int(record[1]) == update
        train = trained[update - 1]
        assert train[0] == "TRAIN" and int(train[1]) == update, train
        assert train[2:5] == record[2:5] and int(train[5]) == 1, (train, record)
        rows.append(f"{accumulation}\t{update}\t{int(record[2]):08x}\t{int(record[3]):08x}")
(evidence / "fresh-loss-variation.tsv").write_text("\n".join(rows) + "\n")
PY
# Independently fixed E3 words, exact counters and EOS rollback through the
# installed public trainer and metrics-ref, using the raw-link f32 inspector.
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_public_installed/evaluate_runtime.esk -o /out/evaluate_caller \
    > /out/evaluate-compile.stdout 2> /out/evaluate-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/evaluate_caller /out/corpus/e3/packed-single \
  > /out/evaluate.stdout 2> /out/evaluate.stderr
grep -Fx TR3-PUBLIC-EVALUATE-AOT-PASS /out/evaluate.stdout >/dev/null
test ! -s /out/evaluate.stderr

# One genuine D2 batch must overfit through public trainer/evaluator APIs;
# public forward and state-dict restoration bind the change to parameters.
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_public_installed/one_batch_overfit.esk -o /out/overfit_caller \
    > /out/overfit-compile.stdout 2> /out/overfit-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/overfit_caller /out/corpus/overfit-one \
  > /out/overfit.stdout 2> /out/overfit.stderr
test ! -s /out/overfit.stderr

# The two training transitions and three held-out transitions are disjoint.
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_public_installed/heldout_runtime.esk -o /out/heldout_caller \
    > /out/heldout-compile.stdout 2> /out/heldout-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 600s \
  /out/heldout_caller /out/corpus/heldout-pair/train \
  /out/corpus/heldout-pair/heldout \
  > /out/heldout.stdout 2> /out/heldout.stderr
grep -Fx TR3-HELDOUT-AOT-PASS /out/heldout.stdout >/dev/null
test ! -s /out/heldout.stderr

# Focused bit and counter boundary proof for the actual installed C bridge.
clang-21 "${flags[@]}" -ffunction-sections -fdata-sections \
  -I /fixed-source/inc -c native/tr3_public_installed_bridge.c \
  -o /out/metrics_bridge_unit.o
clang-21 "${flags[@]}" -I /fixed-source/inc \
  -c tests/tr3_metrics/public_bridge.c -o /out/metrics_bridge_unit_test.o
clang++-21 -Wl,--gc-sections /out/metrics_bridge_unit.o \
  /out/metrics_bridge_unit_test.o \
  /candidate/eshkol-build-canonical/libeshkol-runtime.a \
  -pthread -ldl -lm -o /out/metrics_bridge_unit_test
/out/metrics_bridge_unit_test > /out/metrics-bridge.stdout \
  2> /out/metrics-bridge.stderr
grep -Fx TR3-PUBLIC-METRICS-BRIDGE-PASS /out/metrics-bridge.stdout >/dev/null
test ! -s /out/metrics-bridge.stderr
'

PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.tr3_public_installed.check_unequal_witness \
  "${evidence}/step.stdout" >"${evidence}/unequal-exact.stdout" \
  2>"${evidence}/unequal-exact.stderr"
rg -q '^TR3-TRAIN-UNEQUAL-EXACT-PASS loss-bits=' \
  "${evidence}/unequal-exact.stdout"
test ! -s "${evidence}/unequal-exact.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.tr3_public_installed.check_one_batch_overfit \
  "${evidence}/overfit.stdout" >"${evidence}/overfit-check.stdout" \
  2>"${evidence}/overfit-check.stderr"
rg -q '^TR3-ONE-BATCH-OVERFIT-PASS initial=' \
  "${evidence}/overfit-check.stdout"
test ! -s "${evidence}/overfit-check.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.tr3_public_installed.check_heldout \
  "${evidence}/heldout.stdout" >"${evidence}/heldout-check.stdout" \
  2>"${evidence}/heldout-check.stderr"
rg -q '^TR3-HELDOUT-CHECK-PASS before-bits=' \
  "${evidence}/heldout-check.stdout"
test ! -s "${evidence}/heldout-check.stderr"

python3 - "${evidence}/private.d" "${evidence}/source-closure.txt" <<'PY'
from pathlib import Path
import sys
dep = Path(sys.argv[1]).read_text().replace('\\\n', ' ')
paths = [path.removeprefix('/workspace/') for path in dep.split(':', 1)[1].split()]
assert len(paths) == len(set(paths))
Path(sys.argv[2]).write_text(''.join(path + '\n' for path in paths))
PY
cmp "${source_manifest}" "${evidence}/source-closure.txt"
python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import sys
root, evidence = map(Path, sys.argv[1:])
deps = sorted((evidence / 'native').glob('*.d'))
assert len(deps) == 33
paths = set()
for dep in deps:
    text = dep.read_text().replace('\\\n', ' ')
    for item in text.split(':', 1)[1].split():
        if item.startswith('/workspace/'):
            item = item.removeprefix('/workspace/')
        elif item.startswith('/'):
            continue
        path = (root / item).resolve(strict=True)
        assert path.is_relative_to(root), item
        paths.add(path.relative_to(root).as_posix())
(evidence / 'native-source-closure.txt').write_text(
    ''.join(path + '\n' for path in sorted(paths)))
(evidence / 'native-objects.txt').write_text(
    ''.join(dep.name.removesuffix('.d') + '\n' for dep in deps))
PY
cmp native/tr3_public_installed_native_source_closure.txt \
  "${evidence}/native-source-closure.txt"
cmp native/tr3_public_installed_native_objects.txt \
  "${evidence}/native-objects.txt"
while IFS= read -r facade; do
  cmp "lib/${facade}" "${evidence}/facades/${facade}"
done < native/tr3_public_installed_facades.txt
cmp "${undefined_manifest}" \
  "${evidence}/undefined-symbols.txt"
python3 - "${evidence}/caller.d" <<'PY'
from pathlib import Path
import sys
dep = Path(sys.argv[1]).read_text().replace('\\\n', ' ')
paths = [path.removeprefix('/workspace/') for path in dep.split(':', 1)[1].split()]
assert paths == ['tests/tr3_public_installed/runtime.esk',
                 '/out/facades/transformer/config.esk',
                 '/out/facades/transformer/error_consumer.esk',
                 '/out/facades/transformer/capabilities.esk',
                 '/out/facades/transformer/data.esk',
                 '/out/facades/transformer/diagnostic_transport.esk',
                 '/out/facades/transformer/module.esk',
                 '/out/facades/transformer/optim.esk',
                 '/out/facades/transformer/persistence.esk',
                 '/out/facades/transformer/tokenizer.esk',
                 '/out/facades/transformer/trainer.esk'], paths
PY
if rg '^et_e1b_private_|^tr3-lease-create-internal$|^tr3-lease-unenroll-internal!$|^trainer-create$|^trainer-release!$|^trainer-step!$|^trainer-train!$|^c2-public-trainer-state-release!$' \
    "${evidence}/global-defined.txt"; then
  die "candidate package leaked private trainer authority"
fi
{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'fixed_compiler_commit\t%s\n' "${compiler_commit}"
  printf 'fixed_runner_sha256\t%s\n' "${runner_sha256}"
  printf 'container_digest\t%s\n' "$(tsv_value "${pin}" container_digest)"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
  printf 'global_defined_count\t%s\n' "$(wc -l <"${evidence}/global-defined.txt")"
  printf 'undefined_count\t%s\n' "$(wc -l <"${evidence}/undefined-symbols.txt")"
} >"${evidence}/manifest.tsv"
find "${evidence}" -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | (cd "${evidence}" && xargs -d '\n' sha256sum) \
  >"${evidence}/SHA256SUMS"
printf 'TR3 public installed linked PASS: %s\n' "${evidence}"
