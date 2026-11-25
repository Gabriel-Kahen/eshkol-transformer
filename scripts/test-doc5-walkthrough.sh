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
[[ $# == 0 || ( $# == 1 && "$1" == --check-source ) ]] ||
  die "usage: $0 [--check-source]"
export ESHKOL_SOURCE_DIR ESHKOL_BUILD_DIR
ESHKOL_SOURCE_DIR="$(realpath -m -- "$(eshkol_source_dir)")"
ESHKOL_BUILD_DIR="$(realpath -m -- "$(eshkol_build_dir)")"

scratch="$(mktemp -d /tmp/eshkol-doc5.XXXXXX)"
trap 'rm -rf -- "${scratch}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export TMPDIR="${scratch}/tmp" XDG_CACHE_HOME="${scratch}/cache"
mkdir -p "${scratch}/repo" "${TMPDIR}" "${XDG_CACHE_HOME}"

# Component builders include checkout-relative build/ outputs. Run them all
# against the tracked source files in a disposable checkout.
git ls-files -z --deduplicate | tar --null -T - -cf - | tar -xf - -C "${scratch}/repo"
cp docs/DOC5_{API_REFERENCE,CORPUS_TO_MODEL_TUTORIAL}.md "${scratch}/repo/docs/"
cp scripts/test-doc5-walkthrough.sh "${scratch}/repo/scripts/"
cd "${scratch}/repo"
export BUILD_DIR="${PWD}/build"
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
source_span='^[[:blank:]]*`([^`]+)`[[:blank:]]*$'
rows=0
while IFS= read -r row; do
  [[ "${row}" =~ ^[[:space:]]*\| ]] || continue
  row="${row#"${row%%[![:space:]]*}"}"
  IFS='|' read -r _ component source_cell references gates _ <<< "${row}"
  case "${component}:${source_cell}" in
    ' Component : Source '|' --- : --- ') continue ;;
  esac
  [[ "${source_cell}" =~ ${source_span} ]] ||
    die "DOC5 API reference: malformed source cell in row: ${row}"
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
  gate_count=0
  gate_remainder="${gates}"
  while [[ "${gates}" =~ ${span} ]]; do
    gate="${BASH_REMATCH[1]}"
    gate_remainder="${gate_remainder/"${BASH_REMATCH[0]}"/}"
    gates="${gates#*"${BASH_REMATCH[0]}"}"
    gate_count=$((gate_count + 1))
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
  (( gate_count > 0 )) && [[ -z "${gate_remainder//[[:space:],]/}" ]] ||
    die "DOC5 API reference: malformed gate cell in row: ${row}"
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
    while (match(line, /(^|[^[:alnum:]_./$-])(\.\.?\/)?(docs|include|native|scripts|tests|tools|toolchain|src|internal|benchmarks)\/[[:alnum:]_./+-]+/)) {
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

if [[ ${1:-} == --check-source ]]; then
  printf 'DOC5 source check: PASS\n'
  exit 0
fi

verify_toolchain
printf 'DOC5 toolchain: Eshkol %s, LLVM %s\n' \
  "$(lock_value eshkol_commit)" "$("$(llvm_config)" --version)"

printf '\n== DATA4 source checks ==\n'
bash scripts/test-data4-source.sh >"${scratch}/data4-source.log" 2>&1 || {
  tail -n 80 "${scratch}/data4-source.log" >&2
  die 'DOC5 DATA4 source gate failed'
}
tail -n 3 "${scratch}/data4-source.log"

printf '\n== CLI3 cross-command corpus, training, evaluation, generation ==\n'
mkdir -p "${scratch}/evidence"
bash scripts/test-cli3-cross-command-chain.sh "${scratch}/evidence" \
  >"${scratch}/cli3-cross.log" 2>&1 || {
  tail -n 80 "${scratch}/cli3-cross.log" >&2
  die 'DOC5 CLI3 cross-command gate failed'
}
[[ $(cat "${scratch}/evidence/gate.pass") == CLI3-CROSS-COMMAND-CHAIN-PASS ]] ||
  die 'DOC5 CLI3 cross-command receipt missing'
printf 'CLI3-CROSS-COMMAND-CHAIN-PASS\n'
cli_root="$(cat "${scratch}/evidence/cross-command-scratch-path.txt")"
cli="${cli_root}/eshkol-transformer"
[[ -x "${cli}" && -f "${cli_root}/byte.t1" ]] ||
  die 'DOC5 CLI3 executable or byte tokenizer missing'

printf '\n== installed DATA4 command tests ==\n'
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared \
  tests/cli3/write_fault.c -ldl -o "${scratch}/data4-write-fault.so"
CLI3_DATA4_BINARY="${cli}" CLI3_DATA4_TOKENIZER="${cli_root}/byte.t1" \
  CLI3_DATA4_FAULT_SHIM="${scratch}/data4-write-fault.so" \
  python3 -m unittest -v tests.cli3.test_installed_data4 \
  >"${scratch}/data4-installed.log" 2>&1 || {
  tail -n 80 "${scratch}/data4-installed.log" >&2
  die 'DOC5 installed DATA4 command tests failed'
}
tail -n 3 "${scratch}/data4-installed.log"

printf '\n== integrated DATA4-to-CLI3 recipe ==\n'
mkdir "${cli_root}/prepared-train"
run_json() {
  local name=$1
  shift
  "$@" >"${scratch}/${name}.json" 2>"${scratch}/${name}.stderr"
  [[ -s "${scratch}/${name}.json" && ! -s "${scratch}/${name}.stderr" ]] ||
    die "DOC5 ${name}: expected JSON stdout and empty stderr"
}
run_json prepare "${cli}" corpus prepare \
  --tokenizer "${cli_root}/byte.t1" --source "train=${cli_root}/train.txt" \
  --minimum-bytes 6 --maximum-bytes 6 \
  --output-directory "${cli_root}/prepared-train" --shard-token-limit 4
run_json report-validate "${cli}" corpus report-validate \
  --input-directory "${cli_root}/prepared-train"
cmp "${scratch}/prepare.json" "${scratch}/report-validate.json"
run_json pretrain "${cli}" pretrain --config "${cli_root}/config.json" \
  --tokenizer "${cli_root}/byte.t1" --train-corpus "${cli_root}/prepared-train" \
  --checkpoint "${cli_root}/prepared-first.c2" --max-updates 1
run_json checkpoint "${cli}" checkpoint inspect \
  --input "${cli_root}/prepared-first.c2"
run_json evaluate "${cli}" evaluate --config "${cli_root}/config.json" \
  --tokenizer "${cli_root}/byte.t1" --train-corpus "${cli_root}/prepared-train" \
  --validation-corpus "${cli_root}/heldout" \
  --checkpoint "${cli_root}/prepared-first.c2"
run_json generate "${cli}" generate --config "${cli_root}/config.json" \
  --tokenizer "${cli_root}/byte.t1" --train-corpus "${cli_root}/prepared-train" \
  --checkpoint "${cli_root}/prepared-first.c2" \
  --prompt-hex 41 --sampling greedy --seed 1729
python3 - "${scratch}" <<'PYJSON'
from pathlib import Path
import json
import sys
root = Path(sys.argv[1])
def record(name):
    raw = (root / f'{name}.json').read_bytes()
    assert raw.endswith(b'\n') and raw.count(b'\n') == 1, name
    value = json.loads(raw)
    assert isinstance(value, dict), name
    return value
corpus = record('prepare')
assert corpus['artifact'] == 'token-corpus'
assert corpus['total_tokens'] == 6 and corpus['shard_count'] == 2
assert record('report-validate') == corpus
pretrain = record('pretrain')
checkpoint = record('checkpoint')
assert pretrain['artifact'] == checkpoint['artifact'] == 'checkpoint'
assert pretrain['updates'] == 1 and pretrain['tokens'] == 4
assert checkpoint['format_id'] == 'eshkol-training-state'
assert checkpoint['format_version'] == [1, 0]
assert record('evaluate')['artifact'] == 'evaluation'
assert record('generate') == {
    'artifact': 'generation', 'generated_hex': 'c5',
    'prompt_hex': '41', 'sampling': 'greedy', 'seed': 1729,
}
PYJSON
printf 'DOC5 integrated DATA4-to-CLI3 recipe: PASS\n'
printf '\nDOC5 fresh-environment walkthrough: PASS\n'
