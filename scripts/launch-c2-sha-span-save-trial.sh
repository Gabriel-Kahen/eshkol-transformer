#!/usr/bin/env bash
# One opt-in, non-acceptance matched pair. Run only with an assigned build slot.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
(( $# == 3 )) || {
  echo 'usage: launch-c2-sha-span-save-trial.sh HEAD TREE FRESH_EVIDENCE_DIR' >&2
  exit 2
}
[[ "$1" =~ ^[0-9a-f]{40}$ && "$2" =~ ^[0-9a-f]{40}$ &&
   "$3" == /home/gabe/.codex/evidence/eshkol-transformer/* &&
   "$(realpath -m "$3")" == "$3" ]] || {
  echo 'usage: launch-c2-sha-span-save-trial.sh HEAD TREE FRESH_EVIDENCE_DIR' >&2
  exit 2
}
[[ "${C2_SHA_SPAN_SAVE_TRIAL_SLOT:-}" == assigned ]] || {
  echo 'an assigned C2 SHA span SAVE trial slot is required' >&2; exit 75;
}
head=$1 tree=$2 evidence=$3
toolchain=/home/gabe/.cache/eshkol-fe9dfd52-ubuntu22
image=sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6
exec 9>/home/gabe/.cache/eshkol-transformer-orchestration-build.lock
flock -n -x 9 || { echo 'orchestration build lock busy' >&2; exit 75; }
exec 8>/tmp/eshkol-transformer-heavy-build.lock
flock -n -x 8 || { echo 'heavy build lock busy' >&2; exit 75; }
exec 7>/home/gabe/.codex/eshkol-heavy-docker.lock
flock -n -x 7 || { echo 'pinned Docker lock busy' >&2; exit 75; }
[[ "$(git -C "$root" rev-parse HEAD)" == "$head" &&
   "$(git -C "$root" rev-parse HEAD^{tree})" == "$tree" &&
   -z "$(git -C "$root" status --porcelain --untracked-files=all)" ]] || {
  echo 'source differs from reviewed commit/tree or is dirty' >&2; exit 1;
}
[[ "$(sha256sum "$toolchain/build/eshkol-run" | awk '{print $1}')" == \
   55a552fbf25a2c26daf2f951ad4ff0a9259228ba41f88bed70083849444683b2 ]] || {
  echo 'pinned Eshkol runner changed' >&2; exit 1;
}
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image" ]] || {
  echo 'supported image digest changed' >&2; exit 1;
}
[[ ! -e "$evidence" ]] || { echo 'evidence directory already exists' >&2; exit 1; }
mkdir -p "$evidence"
snapshot() {
  local label=$1
  {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    free -k
    awk '/^(MemAvailable|SwapTotal|SwapFree|Committed_AS):/ {print}' /proc/meminfo
    awk '/^(pswpin|pswpout|pgmajfault|oom_kill) / {print}' /proc/vmstat
    cat /proc/pressure/memory
    df -Pk /tmp "$evidence"
    ps -eo pid,comm,rss --sort=-rss | head -16 || true
  } > "$evidence/host-$label.txt"
}
finish() {
  local trial_rc=$? receipt_rc=0
  trap - EXIT
  if ! printf 'end_utc=%s\nexit=%s\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$trial_rc" \
      >> "$evidence/run-status.txt"; then
    receipt_rc=1
  fi
  snapshot end || true
  if ! (cd "$evidence" &&
        find . -type f ! -name SHA256SUMS \
          ! -path './*/dev/cache-*/*' -print0 | sort -z | \
          xargs -0 -r sha256sum > SHA256SUMS &&
        [[ -s SHA256SUMS ]] && sha256sum -c SHA256SUMS --status); then
    echo 'C2 span SAVE trial evidence seal failed' >&2
    receipt_rc=1
  fi
  if (( trial_rc != 0 )); then exit "$trial_rc"; fi
  exit "$receipt_rc"
}
trap finish EXIT
printf 'start_utc=%s\norder=whole,spans\nsource_head=%s\nsource_tree=%s\n' \
  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$head" "$tree" \
  > "$evidence/run-status.txt"
snapshot start
preflight() {
  local arm=$1 available swap_free tmp_free evidence_free
  available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  swap_free=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
  tmp_free=$(df -Pk /tmp | awk 'NR==2 {print $4}')
  evidence_free=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
  printf '%s MemAvailableKiB=%s SwapFreeKiB=%s TmpFreeKiB=%s EvidenceFreeKiB=%s\n' \
    "$arm" "$available" "$swap_free" "$tmp_free" "$evidence_free" \
    >> "$evidence/resource-preflight.txt"
  (( available >= 8388608 && swap_free >= 4194304 &&
     tmp_free >= 4194304 && evidence_free >= 20971520 ))
}
run_arm() {
  local arm=$1 rc=0
  preflight "$arm" || { echo "$arm resource preflight failed" >&2; return 75; }
  mkdir -p "$evidence/$arm/dev"
  printf '%s start_utc=%s\n' "$arm" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    >> "$evidence/run-status.txt"
  docker run --rm --network none --pull never --cpus=3 \
    --memory=6g --memory-swap=7g \
    --mount type=bind,src="$toolchain",dst=/tc,readonly \
    --mount type=bind,src="$root",dst=/workspace,readonly \
    --mount type=bind,src="$evidence/$arm",dst=/evidence \
    --workdir /workspace \
    -e ESHKOL_SOURCE_DIR=/tc/src -e ESHKOL_BUILD_DIR=/tc/build \
    -e CC=/usr/bin/clang-21 -e CXX=/usr/bin/clang++-21 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e C2_OPERATIONAL_DEVELOPMENT_JOINT_ONLY=1 \
    -e C2_OPERATIONAL_DEVELOPMENT_MODE=joint \
    -e C2_DEV_ARTIFACT_DIR=/evidence/dev \
    -e C2_SPAN_TRIAL_RUNTIME=/evidence/staged/runtime.esk \
    "$image" bash -lc '
      set -euo pipefail
      python3 -B tests/c2/test_c2_sha_span_save_trial.py \
        --stage /evidence/staged --selector "'"$arm"'"
      /usr/bin/time -v /usr/bin/bash /evidence/staged/gate.sh \
        --operational-only > /evidence/outer.stdout 2> /evidence/outer.stderr
    ' > "$evidence/$arm/container.stdout" \
      2> "$evidence/$arm/container.stderr" || rc=$?
  printf '%s end_utc=%s exit=%s\n' "$arm" \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$rc" \
    >> "$evidence/run-status.txt"
  return "$rc"
}
run_arm whole
snapshot between
run_arm spans
for arm in whole spans; do
  base="$evidence/$arm/dev"
  fixture="$base/fixtures-a/joint-exact.c2"
  for output in joint-a joint-b; do
    cmp "$fixture" "$base/clang-a/output/joint/$output.c2"
  done
  rg -x 'C2 OPERATIONAL RUNTIME PASS: 26 checks mode=joint .*' \
    "$base/clang-a/joint.stdout" > /dev/null
  rg '^C2 OPERATIONAL (MEASURE|PHASE|RUNTIME PASS):' \
    "$base/clang-a/joint.stdout" > "$evidence/$arm/markers.txt"
  for hook in et_c2_private_model_sha256_v1 \
              et_c2_private_model_sha256_spans_experiment_v1; do
    [[ "$(nm -g --defined-only --format=posix \
      "$base/runtime/c2_checkpoint_format.o" | \
      awk -v hook="$hook" '$1 == hook {count++} END {print count+0}')" == 1 ]]
  done
  sha256sum "$fixture" "$base/clang-a/output/joint/joint-a.c2" \
    "$base/clang-a/output/joint/joint-b.c2" \
    > "$evidence/$arm/output-hashes.txt"
done
cmp "$evidence/whole/dev/fixtures-a/joint-exact.c2" \
    "$evidence/spans/dev/fixtures-a/joint-exact.c2"
cmp "$evidence/whole/staged/gate.sh" "$evidence/spans/staged/gate.sh"
cmp "$evidence/whole/dev/runtime/c2_checkpoint_format.o" \
    "$evidence/spans/dev/runtime/c2_checkpoint_format.o"
cmp "$evidence/whole/dev/clang-a/compile.stdout" \
    "$evidence/spans/dev/clang-a/compile.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/tests/c2/test_c2_sha_span_save_trial.py" --compare-stdout \
  "$evidence/whole/dev/clang-a/joint.stdout" \
  "$evidence/spans/dev/clang-a/joint.stdout" \
  > "$evidence/stdout-invariants.txt"
diff -u "$evidence/whole/staged/runtime.esk" \
  "$evidence/spans/staged/runtime.esk" > "$evidence/runtime-selector.diff" || \
  [[ $? == 1 ]]
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/tests/c2/test_c2_sha_span_save_trial.py" \
  --fixture "$evidence/whole/dev/fixtures-a/joint-exact.c2" \
  > "$evidence/preimage-budget.txt"
printf 'cross_arm_fixture_equal=true\nall_four_outputs_equal_fixture=true\n' \
  >> "$evidence/run-status.txt"
