#!/usr/bin/env bash
set -euo pipefail

bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
pins="${bundle}/pins.tsv"
result="${bundle}.result"
admission="${bundle}.admission.tsv"
cidfile="${result}.container.cid"
bundle_manifest_sha=''
container_name=''

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
  (cd "$(dirname -- "$(pin deps_parent_manifest)")" && \
    sha256sum -c --quiet "$(basename -- "$(pin deps_parent_manifest)")") || \
    die "frozen dependency inventory entries changed"
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
  local status=$? name first=complete cid image cid_bytes
  local -a fields=()
  trap - EXIT
  set +e
  printf '%s\n' "${status}" >"${result}/host-command-exit.status"
  if [[ -e "${cidfile}" || -L "${cidfile}" ]]; then
    cp -- "${cidfile}" "${result}/host-container.cid" || status=1
    cid="$(cat -- "${cidfile}" 2>/dev/null)"
    cid_bytes="$(wc -c <"${cidfile}" 2>/dev/null)"
    if [[ ! -f "${cidfile}" || -L "${cidfile}" ||
          ! "${cid}" =~ ^[0-9a-f]{64}$ ||
          ( "${cid_bytes}" != 64 && "${cid_bytes}" != 65 ) ]]; then
      printf 'invalid-host-cid\n' >"${result}/container-cleanup.status"
      status=1
    elif timeout 20s docker container inspect --format \
        '{{.Id}}
{{.Name}}
{{.Image}}
{{index .Config.Labels "org.eshkol.cli3.gate"}}
{{index .Config.Labels "org.eshkol.cli3.target"}}' "${cid}" \
        >"${result}/container-inspect.stdout" \
        2>"${result}/container-inspect.stderr"; then
      mapfile -t fields <"${result}/container-inspect.stdout"
      image="$(pin image_id)"
      if [[ "${#fields[@]}" == 5 && "${fields[0]:-}" == "${cid}" &&
            "${fields[1]:-}" == "/${container_name}" &&
            "${fields[2]:-}" == "${image}" &&
            "${fields[3]:-}" == "${bundle_manifest_sha}" &&
            "${fields[4]:-}" == "$(pin target_commit)" ]]; then
        if timeout 20s docker rm -f -- "${cid}" \
            >"${result}/container-remove.stdout" \
            2>"${result}/container-remove.stderr"; then
          printf 'removed-owned-container\n' >"${result}/container-cleanup.status"
        else
          printf 'owned-container-removal-failed\n' >"${result}/container-cleanup.status"
          status=1
        fi
      else
        printf 'ownership-mismatch\n' >"${result}/container-cleanup.status"
        status=1
      fi
    else
      printf 'ownership-inspection-failed\n' >"${result}/container-cleanup.status"
      status=1
    fi
  elif [[ -e "${result}/container.stdout" ]]; then
    printf 'host-cid-missing\n' >"${result}/container-cleanup.status"
    status=1
  else
    printf 'container-not-started\n' >"${result}/container-cleanup.status"
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
   "$(value "${admission}" pr201_head)" == "$(pin pr201_head)" &&
   "$(value "${admission}" pr201_status)" == completed &&
   "$(value "${admission}" pr201_conclusion)" == success &&
   "$(value "${admission}" sole_heavy_slot)" == confirmed ]] || \
  die "exact PR201 successful completion and sole heavy slot have not been admitted"
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
[[ ! -e "${cidfile}" && ! -L "${cidfile}" ]] || \
  die "host-private container CID path already exists: ${cidfile}"
bundle_manifest_sha="$(hash "${bundle}/SHA256SUMS-PREPARED")"
target_commit="$(pin target_commit)"
container_name="eshkol-cli3-${target_commit:0:12}-${bundle_manifest_sha:0:12}"
printf 'name\t%s\nimage\t%s\ngate_manifest_sha256\t%s\ntarget_commit\t%s\n' \
  "${container_name}" "${image}" "${bundle_manifest_sha}" "${target_commit}" \
  >"${result}/container-identity.tsv"

timeout --foreground --signal=TERM --kill-after=30s \
  "$(pin outer_timeout_seconds)s" \
  docker run --network none --cpus=2 --memory=12g --memory-swap=12g \
    --pids-limit=1024 --user "$(id -u):$(id -g)" \
    --name "${container_name}" \
    --label "org.eshkol.cli3.gate=${bundle_manifest_sha}" \
    --label "org.eshkol.cli3.target=${target_commit}" \
    --cidfile "${cidfile}" \
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
