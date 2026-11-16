#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ $# == 2 && "$1" == /* && "$2" == /* ]] || \
  die "usage: $0 /absolute/production-generate-evidence /absolute/new/artifact"
for command in cmp find git grep install ldd readelf realpath sha256sum sort tar uname xargs; do
  require_command "${command}"
done
[[ -d "$1" && ! -L "$1" && ! -e "$2" && ! -L "$2" ]] || \
  die "CLI3 artifact requires existing real evidence and a new output path"
evidence="$(realpath -e -- "$1")"
artifact="$(realpath -m -- "$2")"
archive="${artifact}.tar"
project="$(realpath -e -- "${PROJECT_ROOT}")"
[[ ! -e "${archive}" && ! -L "${archive}" &&
   ! -e "${archive}.sha256" && ! -L "${archive}.sha256" ]] || \
  die "CLI3 artifact archive path already exists"
[[ "${evidence}" != "${project}" && "${evidence}" != "${project}/"* &&
   "${artifact}" != "${project}" && "${artifact}" != "${project}/"* &&
   "${artifact}" != "${evidence}" && "${artifact}" != "${evidence}/"* ]] || \
  die "CLI3 evidence and artifact must be separate and outside checkout"
[[ -f "${evidence}/scratch-path.txt" && ! -L "${evidence}/scratch-path.txt" ]] || \
  die "CLI3 production scratch path is missing"
mapfile -t recorded <"${evidence}/scratch-path.txt"
[[ "${#recorded[@]}" == 1 && "${recorded[0]}" == "${evidence}/.build."* ]] || \
  die "CLI3 production scratch path is not the exact evidence child"
scratch="$(realpath -e -- "${recorded[0]}")"
[[ "${scratch}" == "${recorded[0]}" && -d "${scratch}" && ! -L "${scratch}" ]] || \
  die "CLI3 production scratch path changed"
binary="${scratch}/eshkol-transformer"
[[ -f "${binary}" && -x "${binary}" && ! -L "${binary}" ]] || \
  die "CLI3 tested executable is unavailable"
[[ -f "${scratch}/config.json" && ! -L "${scratch}/config.json" &&
   -f "${scratch}/byte.t1" && ! -L "${scratch}/byte.t1" &&
   -f "${evidence}/tokenizer.json" && ! -L "${evidence}/tokenizer.json" ]] || \
  die "CLI3 production smoke inputs are unavailable"

verify_toolchain
[[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]] || \
  die "CLI3 artifact source checkout is dirty"
for pair in \
  source_closure:source-closure native_source_closure:native-source-closure \
  public_strings:public-strings public_exports:package-exports \
  undefined_symbols:undefined; do
  source_name=${pair%%:*}
  package_name=${pair#*:}
  cmp "${PROJECT_ROOT}/native/cli3_generate_${source_name}.txt" \
    "${scratch}/cli3.o.evidence/${package_name}.txt"
done

mkdir -m 0700 -- "${artifact}"
mkdir -m 0700 -- "${artifact}/metadata" "${artifact}/smoke"
install -m 0755 -- "${binary}" "${artifact}/eshkol-transformer"
install -m 0644 -- "${LOCK_FILE}" "${artifact}/metadata/eshkol.lock"
install -m 0644 -- "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "${artifact}/metadata/toolchain-provenance.tsv"
for pair in \
  source_closure:source-closure native_source_closure:native-source-closure \
  public_strings:public-strings public_exports:package-exports \
  undefined_symbols:undefined; do
  source_name=${pair%%:*}
  package_name=${pair#*:}
  install -m 0644 -- "${scratch}/cli3.o.evidence/${package_name}.txt" \
    "${artifact}/metadata/${source_name}.txt"
done
install -m 0644 -- "${scratch}/config.json" "${artifact}/smoke/config.json"
LC_ALL=C readelf -h -d "${artifact}/eshkol-transformer" \
  >"${artifact}/metadata/elf.txt"
grep -F 'Class:                             ELF64' "${artifact}/metadata/elf.txt" >/dev/null
grep -F 'Machine:                           Advanced Micro Devices X86-64' \
  "${artifact}/metadata/elf.txt" >/dev/null
LC_ALL=C ldd "${artifact}/eshkol-transformer" \
  >"${artifact}/metadata/dependencies.txt"
! grep -F 'not found' "${artifact}/metadata/dependencies.txt" >/dev/null

(cd "${artifact}/smoke" && ../eshkol-transformer --version \
  >version.stdout 2>version.stderr)
printf 'eshkol-transformer 0.0.1\n' >"${artifact}/smoke/version.expected"
cmp "${artifact}/smoke/version.expected" "${artifact}/smoke/version.stdout"
[[ ! -s "${artifact}/smoke/version.stderr" ]]
(cd "${artifact}/smoke" && ../eshkol-transformer tokenizer byte \
  --config config.json --output byte.t1 >tokenizer.json 2>tokenizer.stderr)
[[ ! -s "${artifact}/smoke/tokenizer.stderr" ]]
cmp "${evidence}/tokenizer.json" "${artifact}/smoke/tokenizer.json"
cmp "${scratch}/byte.t1" "${artifact}/smoke/byte.t1"

printf 'format\teshkol-cli3-binary-v1\nhead\t%s\ntree\t%s\nos\t%s\narch\t%s\n' \
  "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" \
  "$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')" \
  "$(uname -s)" "$(uname -m)" >"${artifact}/metadata/identity.tsv"
(cd "${artifact}" && find . -type l -print -quit | grep -q .) && \
  die "CLI3 artifact contains a symlink"
(cd "${artifact}" && find . -type f ! -name SHA256SUMS -print0 | \
  LC_ALL=C sort -z | xargs -0 -r sha256sum) >"${artifact}/SHA256SUMS"
(cd "${artifact}" && sha256sum -c SHA256SUMS >/dev/null)
[[ ! -e "${archive}" && ! -L "${archive}" &&
   ! -e "${archive}.sha256" && ! -L "${archive}.sha256" ]] || \
  die "CLI3 artifact archive path already exists"
tar --format=ustar --sort=name --mtime='@0' --owner=0 --group=0 \
  --numeric-owner -cf "${archive}" -C "$(dirname -- "${artifact}")" \
  "$(basename -- "${artifact}")"
(cd "$(dirname -- "${archive}")" && \
  sha256sum "$(basename -- "${archive}")" \
  >"$(basename -- "${archive}").sha256" && \
  sha256sum -c "$(basename -- "${archive}").sha256" >/dev/null)
printf 'CLI3-BINARY-ARTIFACT-PASS %s\n' "${archive}"
