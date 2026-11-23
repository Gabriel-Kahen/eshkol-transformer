#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

unset MAKEFLAGS MFLAGS GNUMAKEFLAGS
# Use the checkout selected by each Git command and isolate the scratch index.
git_local_env="$(git rev-parse --local-env-vars)"
while read -r name; do
  unset "${name}"
done <<< "${git_local_env}"

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

require_make_target() {
  local target="$1" context="$2"
  awk -F: -v target="${target}" '
    /^[^\t#][^:]*:/ && $0 !~ /^[^:]*:+=/ {
      count = split($1, names, /[[:blank:]]+/)
      for (i = 1; i <= count; i++)
        if (names[i] == target) found = 1
    }
    END { exit !found }
  ' Makefile || die "${context}: missing Makefile target: ${target}"
}

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
        require_make_target "${gate#make }" 'DOC5 API reference'
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

tutorial=docs/DOC5_CORPUS_TO_MODEL_TUTORIAL.md
# Check Make commands in code, and paths in code, prose, and Markdown links.
awk '
  function make_references(code,   target) {
    while (match(code, /(^|[[:space:];(&|])make[[:blank:]]+["\047]?[[:alnum:]_.][[:alnum:]_.-]*/)) {
      target = substr(code, RSTART, RLENGTH)
      code = substr(code, RSTART + RLENGTH)
      sub(/^.*make[[:blank:]]+["\047]?/, "", target)
      print FNR, "target", target
    }
  }
  /^[[:blank:]]*(```+|~~~+)/ {
    marker = substr($0, match($0, /[`~]/), 1)
    if (fence == marker) fence = ""
    else if (!fence) fence = marker
    next
  }
  {
    if (fence) make_references($0)
    else {
      count = split($0, spans, /`+/)
      for (i = 2; i < count; i += 2) make_references(spans[i])
    }
    line = $0
    while (match(line, /(^|[^[:alnum:]_./$-])(\.\.?\/)?(docs|include|native|scripts|tests|tools|src|internal|benchmarks)\/[[:alnum:]_./+-]+/)) {
      path = substr(line, RSTART, RLENGTH)
      line = substr(line, RSTART + RLENGTH)
      sub(/^[^[:alnum:]_.]/, "", path)
      sub(/^\.\.?\//, "", path)
      sub(/\.$/, "", path)
      print FNR, "path", path
    }
    if (!fence) {
      line = $0
      while (match(line, /\]\([[:alnum:]_][[:alnum:]_.-]*\.md[)#]/)) {
        print FNR, "path", "docs/" substr(line, RSTART + 2, RLENGTH - 3)
        line = substr(line, RSTART + RLENGTH)
      }
    }
  }
' "${tutorial}" > "${scratch}/tutorial-references"
while read -r line kind reference; do
  context="${tutorial}:${line}"
  case "${kind}" in
    target) require_make_target "${reference}" "${context}" ;;
    path) [[ -e "${reference}" ]] || die "${context}: missing path: ${reference}" ;;
  esac
done < "${scratch}/tutorial-references"
printf 'DOC5 tutorial: verified Makefile targets and repository paths\n'

run_make_gate() {
  local target="$1" marker="$2" log="${scratch}/$1.log"
  make "${target}" 2>&1 | tee "${log}"
  rg --quiet --line-regexp -- "${marker}" "${log}" ||
    die "DOC5 ${target} completed without its expected output"
}

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
run_make_gate test-mod5-aot 'MOD5 AOT bridge ASan/UBSan/LSan and allocation-free dispatch: PASS'

printf '\n== IO5 formats, C1 checkpoint and compiled reload ==\n'
run_make_gate test-io5 'Ran [1-9][0-9]* tests? in [0-9]+[.][0-9]+s'
run_make_gate test-io5-live-c1 'IO5 LIVE C1 PASS: deterministic AOT and checkpoint, five exact live round trips, corruption rejections'

printf '\n== MP4 storage conversion and loss-scale recovery ==\n'
MP4_ASAN_DETECT_LEAKS=1 run_make_gate test-mp4 'MP4 ASan/UBSan: PASS \(LSan detect_leaks=1\)'

printf '\n== PA4-Wider reference and online-softmax conformance ==\n'
run_make_gate test-pa4-wider 'PA4-Wider benchmark: PASS \(432 pairs, 864 timed samples verified\)'

printf '\nDOC5 fresh-environment walkthrough: PASS\n'
