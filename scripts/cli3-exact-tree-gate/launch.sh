#!/usr/bin/env bash
set -euo pipefail

bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
pins="${bundle}/pins.tsv"
result="${bundle}.result"
admission="${bundle}.admission.tsv"

value() {
  awk -F '\t' -v key="$2" '$1 == key { value=$2; count++ }
       END { if (count == 1 && value != "") print value; else exit 1 }' "$1"
}
pin() { value "${pins}" "$1"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
hash() { sha256sum "$1" | awk '{print $1}'; }

verify_bundle() {
  (cd "${bundle}" && sha256sum -c SHA256SUMS-PREPARED >/dev/null)
  [[ "$(git bundle list-heads "${bundle}/source.bundle")" == \
     "$(pin target_commit) refs/heads/codex/cli3-installed-corpus-validate" ]] || \
    die "source bundle does not name the exact accepted branch"
  [[ "$(hash "$(pin deps_parent_manifest)")" == \
     "$(pin deps_parent_manifest_sha256)" ]] || \
    die "frozen dependency inventory changed"
  local deps
  deps="$(pin deps_root)"
  [[ "$(git -C "${deps}/eshkol-src" rev-parse HEAD)" == "$(pin eshkol_commit)" &&
     "$(git -C "${deps}/eshkol-src" rev-parse 'HEAD^{tree}')" == \
       "$(pin eshkol_tree)" &&
     -z "$(git -C "${deps}/eshkol-src" status --porcelain --untracked-files=all)" ]] || \
    die "frozen Eshkol source identity or cleanliness changed"
  [[ "$(hash "${deps}/eshkol-build/eshkol-run")" == \
     "$(pin eshkol_runner_sha256)" &&
     "$(hash "${deps}/eshkol-build/libeshkol-runtime.a")" == \
       "$(pin eshkol_runtime_sha256)" &&
     "$(hash "${deps}/eshkol-build/eshkol-transformer-provenance.tsv")" == \
       "$(pin eshkol_provenance_sha256)" ]] || \
    die "frozen fe9 compiler, runtime, or provenance changed"
}

if [[ "${1:-}" == --verify-prepared && "$#" == 1 ]]; then
  verify_bundle
  printf 'CLI3 EXACT-TREE PREPARED VERIFIED: %s\n' "$(pin target_commit)"
  exit 0
fi
[[ "$#" == 0 ]] || die "usage: $0 [--verify-prepared]"
mkdir -- "${result}" || die "one-use result directory already exists: ${result}"

finalize() {
  local status=$? name first=complete
  trap - EXIT
  set +e
  if [[ -s "${result}/container.cid" ]]; then
    docker rm -f "$(cat "${result}/container.cid")" >/dev/null 2>&1 || true
  fi
  if [[ -d "${result}/source/.git" ]]; then
    git -C "${result}/source" rev-parse HEAD >"${result}/final-head.txt" 2>&1 || status=1
    git -C "${result}/source" rev-parse 'HEAD^{tree}' \
      >"${result}/final-tree.txt" 2>&1 || status=1
    git -C "${result}/source" status --porcelain --untracked-files=all \
      >"${result}/final-status.txt" 2>&1 || status=1
    [[ "$(cat "${result}/final-head.txt")" == "$(pin target_commit)" &&
       "$(cat "${result}/final-tree.txt")" == "$(pin target_tree)" &&
       ! -s "${result}/final-status.txt" ]] || status=1
  else
    status=1
  fi
  for name in initial-pins standalone-d1 installed-cli3 archive-and-app-symbols \
              cross-command final-pins; do
    if [[ ! -f "${result}/steps/${name}.status" ]]; then
      [[ "${first}" == complete ]] && first="${name}:missing"
      status=1
    elif [[ "$(cat "${result}/steps/${name}.status")" != 0 ]]; then
      [[ "${first}" == complete ]] && first="${name}:failed"
      status=1
    fi
  done
  if [[ "${first}" == complete && "${status}" != 0 ]]; then
    first=host-or-container
  fi
  printf '%s\n' "${first}" >"${result}/failure-stage.txt"
  printf '%s\n' "${status}" >"${result}/overall.status"
  printf '%s\n' "${status}" >"${result}/host-final-status.txt"
  if ! (cd "${result}" && find . -type f ! -name SHA256SUMS-RUN -print0 | \
      LC_ALL=C sort -z | xargs -0 -r sha256sum >SHA256SUMS-RUN); then
    status=1
    printf '%s\n' "${status}" >"${result}/overall.status"
    printf '%s\n' "${status}" >"${result}/host-final-status.txt"
    rm -f -- "${result}/SHA256SUMS-RUN"
  fi
  printf 'CLI3 exact-tree result: %s (status %s)\n' "${result}" "${status}" >&2
  exit "${status}"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify_bundle >"${result}/bundle-verify.stdout" 2>"${result}/bundle-verify.stderr"
[[ -r "${admission}" ]] || die "ROOT resource-admission record is missing: ${admission}"
[[ "$(value "${admission}" pr201_run)" == "$(pin pr201_run)" &&
   "$(value "${admission}" pr201_status)" == completed &&
   "$(value "${admission}" sole_heavy_slot)" == confirmed ]] || \
  die "PR201 completion and sole heavy slot have not been admitted"
cp -- "${admission}" "${result}/admission.tsv"

for command in docker flock git timeout /usr/bin/time; do
  command -v "${command}" >/dev/null || die "required host command missing: ${command}"
done
exec 9>"$(dirname -- "${bundle}")/.cli3-heavy-gate.lock"
flock -n 9 || die "another admitted CLI3 heavy gate holds the slot"
available_mem="$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)"
available_disk="$(df -PB1 "${result}" | awk 'NR == 2 { print $4 }')"
filesystem="$(df -PT "${result}" | awk 'NR == 2 { print $2 }')"
[[ "${available_mem}" =~ ^[0-9]+$ &&
   "${available_disk}" =~ ^[0-9]+$ &&
   "${available_mem}" -ge "$(pin minimum_mem_available_kib)" &&
   "${available_disk}" -ge "$(pin minimum_disk_available_bytes)" &&
   "$(nproc)" -ge 2 && "${filesystem}" != tmpfs ]] || \
  die "resource admission failed: memory=${available_mem} KiB disk=${available_disk} bytes fs=${filesystem}"
printf 'mem_available_kib\t%s\ndisk_available_bytes\t%s\nfilesystem\t%s\ncpus\t%s\n' \
  "${available_mem}" "${available_disk}" "${filesystem}" "$(nproc)" \
  >"${result}/resource-admission.tsv"
image="$(pin image_id)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == "${image}" ]] || \
  die "supported f31 image identity changed"
printf '%s\n' "${image}" >"${result}/image-id.txt"

git clone --no-checkout "${bundle}/source.bundle" "${result}/source" \
  >"${result}/source-clone.stdout" 2>"${result}/source-clone.stderr"
git -C "${result}/source" checkout --detach "$(pin target_commit)" \
  >"${result}/source-checkout.stdout" 2>"${result}/source-checkout.stderr"
[[ "$(git -C "${result}/source" rev-parse 'HEAD^{tree}')" == \
   "$(pin target_tree)" &&
   -z "$(git -C "${result}/source" status --porcelain --untracked-files=all)" ]] || \
  die "cloned source does not match the accepted exact tree"
mkdir "${result}/source/.deps"
chmod -R a-w "${result}/source"
mkdir -p "${result}/steps" "${result}/tmp" "${result}/cache" "${result}/home"
deps="$(pin deps_root)"

timeout --foreground --signal=TERM --kill-after=30s \
  "$(pin outer_timeout_seconds)s" \
  docker run --rm --network none --cpus=2 --memory=12g --memory-swap=12g \
    --pids-limit=1024 --user "$(id -u):$(id -g)" \
    --cidfile "${result}/container.cid" \
    -v "${result}/source:/workspace:ro" \
    -v "${deps}:/workspace/.deps:ro" \
    -v "${bundle}:/gate:ro" -v "${result}:/out" -w /workspace \
    -e ESHKOL_SOURCE_DIR=/workspace/.deps/eshkol-src \
    -e ESHKOL_BUILD_DIR=/workspace/.deps/eshkol-build \
    -e BUILD_DIR=/out/build -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache \
    -e HOME=/out/home -e CC=clang-21 -e CXX=clang++-21 \
    -e LLVM_CONFIG_EXECUTABLE=llvm-config-21 -e ESHKOL_JOBS=2 \
    "${image}" /usr/bin/bash /gate/inside.sh run \
    >"${result}/container.stdout" 2>"${result}/container.stderr"
