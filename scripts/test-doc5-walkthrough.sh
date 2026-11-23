#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
export ESHKOL_SOURCE_DIR ESHKOL_BUILD_DIR
ESHKOL_SOURCE_DIR="$(realpath -m -- "$(eshkol_source_dir)")"
ESHKOL_BUILD_DIR="$(realpath -m -- "$(eshkol_build_dir)")"
verify_toolchain
printf 'DOC5 toolchain: Eshkol %s, LLVM %s\n' \
  "$(lock_value eshkol_commit)" "$("$(llvm_config)" --version)"

scratch="$(mktemp -d /tmp/eshkol-doc5.XXXXXX)"
trap 'rm -rf -- "${scratch}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export TMPDIR="${scratch}/tmp" XDG_CACHE_HOME="${scratch}/cache"
mkdir -p "${scratch}/repo" "${TMPDIR}" "${XDG_CACHE_HOME}"

# Component builders include checkout-relative build/ outputs. Run them all
# against the tracked source files in a disposable checkout, including the Git
# index needed by the Python-isolation gate.
git ls-files -z --deduplicate | tar --null -T - -cf - | tar -xf - -C "${scratch}/repo"
cp docs/DOC5_{API_REFERENCE,CORPUS_TO_MODEL_TUTORIAL}.md "${scratch}/repo/docs/"
cd "${scratch}/repo"
git init --quiet --template=
git add -f -- .
export BUILD_DIR="${PWD}/build" DATA4_EVIDENCE_DIR="${scratch}/data4"
unset ESHKOL_PATH ESHKOL_JIT_CACHE_DIR ESHKOL_AOT_MODULE_CACHE_DIR ESHKOL_LIB_DIR
export ESHKOL_JIT_CACHE=0

# The API table is the reference inventory. Read each code span from the
# documented source, symbol, and gate columns so drift fails the walkthrough.
span='`([^`]+)`'
rows=0
while IFS='|' read -r left subsystem source_cell references gates scope right; do
  [[ -z "${left}" && "${source_cell}" =~ ${span} ]] || continue
  source_file="${BASH_REMATCH[1]}"
  [[ -f "${source_file}" ]] || die "DOC5 missing source: ${source_file}"
  while [[ "${references}" =~ ${span} ]]; do
    symbol="${BASH_REMATCH[1]}"
    references="${references#*"${BASH_REMATCH[0]}"}"
    awk -v needle="${symbol}" '
      {
        rest = $0
        while ((pos = index(rest, needle)) > 0) {
          before = pos > 1 ? substr(rest, pos - 1, 1) : ""
          after = substr(rest, pos + length(needle), 1)
          if (before !~ /[[:alnum:]_?!-]/ && after !~ /[[:alnum:]_?!-]/) {
            found = 1
            exit
          }
          rest = substr(rest, pos + 1)
        }
      }
      END { exit !found }
    ' "${source_file}" ||
      die "DOC5 missing reference '${symbol}' in ${source_file}"
  done
  while [[ "${gates}" =~ ${span} ]]; do
    gate="${BASH_REMATCH[1]}"
    gates="${gates#*"${BASH_REMATCH[0]}"}"
    case "${gate}" in
      'make '*)
        target="${gate#make }"
        awk -F: -v target="${target}" '$1 == target { found=1 } END { exit !found }' Makefile ||
          die "DOC5 missing Makefile target: ${target}"
        ;;
      'bash scripts/'*)
        [[ -f "${gate#bash }" ]] || die "DOC5 missing script: ${gate#bash }"
        ;;
      *) die "DOC5 unrecognized gate: ${gate}" ;;
    esac
  done
  rows=$((rows + 1))
done < docs/DOC5_API_REFERENCE.md
(( rows > 0 )) || die 'DOC5 API table is empty'
printf 'DOC5 API reference: verified %s source rows\n' "${rows}"

printf '\n== DATA4 corpus selection, D1 shards and bound report ==\n'
bash scripts/test-data4-source.sh
if ! bash scripts/test-data4-native.sh; then
  for diagnostic in "${DATA4_EVIDENCE_DIR}"/*.stderr; do
    if [[ -s "${diagnostic}" ]]; then
      tail -n 80 -- "${diagnostic}" >&2
    fi
  done
  die 'DOC5 DATA4 native gate failed'
fi
cat "${DATA4_EVIDENCE_DIR}/report-a.tsv"

printf '\n== MOD5 Eshkol AOT decoder training ==\n'
make test-mod5-aot

printf '\n== IO5 formats, C1 checkpoint and compiled reload ==\n'
make test-io5
make test-io5-live-c1

printf '\n== MP4 storage conversion and loss-scale recovery ==\n'
MP4_ASAN_DETECT_LEAKS=1 make test-mp4

printf '\n== PA4-Wider reference and online-softmax conformance ==\n'
make test-pa4-wider

printf '\nDOC5 fresh-environment walkthrough: PASS\n'
