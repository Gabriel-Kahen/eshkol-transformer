#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
usage: test-cli3-allocation-fallback.sh \
  --runtime-source DIR --runtime-build DIR --cli-artifact-dir DIR \
  [--evidence-dir DIR]
EOF
  exit 2
}

runtime_source=
runtime_build=
cli_artifact_dir=
evidence_dir=
while (( $# )); do
  case "$1" in
    --runtime-source) (( $# >= 2 )) || usage; runtime_source=$2; shift 2 ;;
    --runtime-build) (( $# >= 2 )) || usage; runtime_build=$2; shift 2 ;;
    --cli-artifact-dir) (( $# >= 2 )) || usage; cli_artifact_dir=$2; shift 2 ;;
    --evidence-dir) (( $# >= 2 )) || usage; evidence_dir=$2; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

for command in ar awk cmp find git nm readelf rg sha256sum sort timeout; do
  command -v "${command}" >/dev/null 2>&1 || die "required command missing: ${command}"
done
for directory in "${runtime_source}" "${runtime_build}" "${cli_artifact_dir}"; do
  [[ "${directory}" = /* && -d "${directory}" ]] || \
    die "required directory must be absolute and existing: ${directory}"
done
runtime_source="$(readlink -f -- "${runtime_source}")"
runtime_build="$(readlink -f -- "${runtime_build}")"
cli_artifact_dir="$(readlink -f -- "${cli_artifact_dir}")"

runner="${runtime_build}/eshkol-run"
runtime_archive="${runtime_build}/libeshkol-runtime.a"
cli_archive="${cli_artifact_dir}/libeshkol_transformer_cli3.a"
cli_object="${cli_artifact_dir}/cli3.o"
cli_evidence="${cli_artifact_dir}/cli3.o.evidence"
production_cli="${cli_artifact_dir}/eshkol-transformer"
provenance="${runtime_build}/final-provenance.tsv"
[[ -x "${runner}" && -r "${runtime_archive}" && -r "${cli_archive}" && \
   -r "${cli_object}" && -d "${cli_evidence}" && -x "${production_cli}" && \
   -r "${provenance}" ]] || \
  die "runtime or CLI production artifact is incomplete"

tsv_value() {
  local file=$1 key=$2
  awk -F '\t' -v key="${key}" '$1 == key { value=$2; count++ }
    END { if (count != 1 || value == "") exit 1; print value }' "${file}"
}
expected_commit="$(awk -F '\t' '$1 == "eshkol_commit" { print $2 }' \
  "${PROJECT_ROOT}/toolchain/eshkol.lock")"
[[ "${expected_commit}" == 81298b4a9608fb92eb6f351a2eabd8392da7d9ef ]] || \
  die "CLI3 allocation witness is pinned to final runtime 81298b4a"
[[ "$(tsv_value "${provenance}" source_commit)" == "${expected_commit}" ]] || \
  die "runtime build provenance commit differs from the project lock"
[[ "$(tsv_value "${provenance}" source_tree)" == \
   7669312845a9d8d372006af52271045e69505813 ]] || \
  die "runtime build provenance tree differs from the reviewed final tree"
[[ "$(tsv_value "${provenance}" source_status)" == clean && \
   "$(tsv_value "${provenance}" build_type)" == Release && \
   "$(tsv_value "${provenance}" promotion_testing)" == OFF && \
   "$(tsv_value "${provenance}" llvm_version)" == 21.1.8 ]] || \
  die "runtime build provenance is not the reviewed production configuration"
[[ "$(sha256sum "${runner}" | awk '{ print $1 }')" == \
   4a0e6303f7b85ed06fb753b52b62155235a3a77bca6c32aeb17241a28ed80be1 ]] || \
  die "final runtime runner hash differs"
[[ "$(sha256sum "${runtime_archive}" | awk '{ print $1 }')" == \
   c32bb593ac1f365f3cbeaefd581704c4be029a4aa8877db29463d0e12356c168 ]] || \
  die "final runtime archive hash differs"
[[ "$(git -C "${runtime_source}" rev-parse HEAD)" == "${expected_commit}" && \
   "$(git -C "${runtime_source}" rev-parse 'HEAD^{tree}')" == \
     7669312845a9d8d372006af52271045e69505813 && \
   -z "$(git -C "${runtime_source}" status --porcelain --untracked-files=all)" ]] || \
  die "runtime source is not the exact clean reviewed checkout"

cxx="$(tsv_value "${provenance}" cxx)"
[[ "${cxx}" = /* && -x "${cxx}" ]] || die "recorded C++ compiler is unavailable"
[[ "$(${cxx} --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')" == 21.1.8 ]] || \
  die "allocation witness requires Clang++ 21.1.8"

if [[ -z "${evidence_dir}" ]]; then
  evidence_dir="$(mktemp -d "${TMPDIR:-/tmp}/cli3-allocation-fallback.XXXXXX")"
else
  [[ "${evidence_dir}" = /* ]] || die "evidence directory must be absolute"
  mkdir -p -- "${evidence_dir}"
  [[ -z "$(find "${evidence_dir}" -mindepth 1 -print -quit)" ]] || \
    die "evidence directory must be empty"
fi
mkdir -p "${evidence_dir}/cache"

ar t "${cli_archive}" >"${evidence_dir}/archive-members.txt"
printf '%s\n' cli3.o >"${evidence_dir}/expected-archive-members.txt"
cmp "${evidence_dir}/expected-archive-members.txt" \
  "${evidence_dir}/archive-members.txt" || \
  die "CLI3 production archive member inventory drifted"
ar p "${cli_archive}" cli3.o >"${evidence_dir}/cli3-archive-member.o"
cmp "${cli_object}" "${evidence_dir}/cli3-archive-member.o" || \
  die "CLI3 production archive does not contain its measured object"
readelf -Ws "${cli_object}" >"${evidence_dir}/cli3-readelf-symbols.txt"
cmp "${cli_evidence}/readelf-symbols.txt" \
  "${evidence_dir}/cli3-readelf-symbols.txt" || \
  die "CLI3 production object differs from its measured symbol evidence"
for manifest in \
    defined_symbols:global-defined.txt \
    public_exports:package-exports.txt \
    public_strings:public-strings.txt \
    source_closure:source-closure.txt \
    native_source_closure:native-source-closure.txt \
    undefined_symbols:undefined.txt; do
  repository_name="native/cli3_${manifest%%:*}.txt"
  evidence_name="${manifest#*:}"
  cmp "${PROJECT_ROOT}/${repository_name}" "${cli_evidence}/${evidence_name}" || \
    die "CLI3 production evidence drifted: ${evidence_name}"
done

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${evidence_dir}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 900s "${runner}" \
    --strict-types --optimize 0 --no-stdlib --emit-object \
    --emit-depfile "${evidence_dir}/cli-main.d" \
    -I "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/src/eshkol_transformer/cli.esk" \
    -o "${evidence_dir}/cli-main.o" \
    >"${evidence_dir}/compile.stdout" 2>"${evidence_dir}/compile.stderr"
[[ ! -s "${evidence_dir}/compile.stderr" ]] || die "strict CLI main compile wrote stderr"

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -Wconversion \
  -Wsign-conversion -Wshadow -fPIC -I "${runtime_source}/inc" \
  -c "${PROJECT_ROOT}/tests/cli3/allocation_fallback_shim.cpp" \
  -o "${evidence_dir}/allocation-fallback-shim.o"

wrap_flags=(
  -Wl,--wrap=malloc
  -Wl,--wrap=arena_allocate_vector_with_header
  -Wl,--wrap=et_e1b_public_cli3_dispatch_fallback_v1
  -Wl,--wrap=eshkol_push_exception_handler
)
"${cxx}" -fPIE -fuse-ld=bfd "${evidence_dir}/cli-main.o" \
  "${evidence_dir}/allocation-fallback-shim.o" \
  -L "${cli_artifact_dir}" -leshkol_transformer_cli3 \
  "${runtime_archive}" "${wrap_flags[@]}" \
  -Wl,-Map,"${evidence_dir}/allocation-fallback.map" \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz -lopenblas \
  -o "${evidence_dir}/eshkol-transformer-allocation-fallback"

nm -u "${evidence_dir}/allocation-fallback-shim.o" | awk '{ print $2 }' | \
  rg '^__real_' | LC_ALL=C sort >"${evidence_dir}/actual-real-symbols.txt"
printf '%s\n' \
  __real_arena_allocate_vector_with_header \
  __real_eshkol_push_exception_handler \
  __real_et_e1b_public_cli3_dispatch_fallback_v1 \
  __real_malloc >"${evidence_dir}/expected-real-symbols.txt"
cmp "${evidence_dir}/expected-real-symbols.txt" \
  "${evidence_dir}/actual-real-symbols.txt" || die "shim GNU real-symbol set drifted"
nm -g --defined-only "${evidence_dir}/eshkol-transformer-allocation-fallback" | \
  awk '{ print $3 }' | rg '^__wrap_' | LC_ALL=C sort \
  >"${evidence_dir}/actual-wrap-symbols.txt"
sed 's/^__real_/__wrap_/' "${evidence_dir}/expected-real-symbols.txt" \
  >"${evidence_dir}/expected-wrap-symbols.txt"
cmp "${evidence_dir}/expected-wrap-symbols.txt" \
  "${evidence_dir}/actual-wrap-symbols.txt" || die "linked GNU wrapper set drifted"
map_symbol_from() {
  local origin=$1 symbol=$2
  awk -v origin="${origin}" -v symbol="${symbol}" '
    index($0, origin) { active=1; next }
    active && $NF == symbol { found=1; exit }
    active && $1 ~ /^\./ { active=0 }
    END { exit found ? 0 : 1 }
  ' "${evidence_dir}/allocation-fallback.map" || \
    die "link map does not bind ${symbol} from ${origin}"
}
cli_member="${cli_archive}(cli3.o)"
runtime_exception_member="${runtime_archive}(runtime_exceptions_hosted.cpp.o)"
runtime_object_member="${runtime_archive}(runtime_object_alloc.cpp.o)"
map_symbol_from "${cli_member}" et_e1b_public_cli3_dispatch_v1
map_symbol_from "${cli_member}" et_e1b_public_cli3_dispatch_fallback_v1
map_symbol_from "${runtime_exception_member}" eshkol_runtime_emergency_raise_v1
map_symbol_from "${runtime_exception_member}" eshkol_push_exception_handler
map_symbol_from "${runtime_object_member}" arena_allocate_vector_with_header

fault_cli="${evidence_dir}/eshkol-transformer-allocation-fallback"
run_fault() {
  local name=$1 marker=$2
  shift 2
  set +e
  "$@" >"${evidence_dir}/${name}.stdout" 2>"${evidence_dir}/${name}.stderr"
  local status=$?
  set -e
  [[ "${status}" == 70 ]] || die "${name} exited ${status}, expected 70"
  [[ ! -s "${evidence_dir}/${name}.stdout" ]] || die "${name} wrote stdout"
  printf '%s\n' 'eshkol-transformer: internal: diagnostic unavailable' \
    >"${evidence_dir}/${name}.expected.stderr"
  cmp "${evidence_dir}/${name}.expected.stderr" \
    "${evidence_dir}/${name}.stderr" || die "${name} static fallback drifted"
  [[ -s "${marker}" ]] || die "${name} did not produce its shim marker"
}

handler_marker="${evidence_dir}/handler.marker"
run_fault handler "${handler_marker}" env \
  CLI3_ALLOCATION_FALLBACK_MODE=handler \
  CLI3_ALLOCATION_FALLBACK_MARKER="${handler_marker}" \
  "${fault_cli}" --version
printf '%s\n' 'mode=handler trigger=1 fallback=1 depth=1 push=2 condition=5' \
  >"${evidence_dir}/handler.expected.marker"
cmp "${evidence_dir}/handler.expected.marker" "${handler_marker}" || \
  die "handler allocation marker drifted"

printf 'abcdefg' >"${evidence_dir}/document"
"${production_cli}" tokenizer byte \
  --config "${PROJECT_ROOT}/tests/x1/fixtures/minimal_config_v1.json" \
  --output "${evidence_dir}/byte.tsv" \
  >"${evidence_dir}/prepare.stdout" 2>"${evidence_dir}/prepare.stderr"
[[ ! -s "${evidence_dir}/prepare.stderr" ]] || die "tokenizer preparation wrote stderr"
mkdir "${evidence_dir}/corpus"
rollback_marker="${evidence_dir}/rollback.marker"
lock="${evidence_dir}/corpus/.d1-writer-lock"
shard0="${evidence_dir}/corpus/shard-0000000000000000.ets"
run_fault rollback "${rollback_marker}" env \
  CLI3_ALLOCATION_FALLBACK_MODE=rollback \
  CLI3_ALLOCATION_FALLBACK_MARKER="${rollback_marker}" \
  CLI3_ALLOCATION_FALLBACK_LOCK="${lock}" \
  CLI3_ALLOCATION_FALLBACK_SHARD0="${shard0}" \
  "${fault_cli}" corpus build --tokenizer "${evidence_dir}/byte.tsv" \
    --document "${evidence_dir}/document" \
    --output-directory "${evidence_dir}/corpus" --shard-token-limit 3
printf '%s\n' 'mode=rollback trigger=1 fallback=1 depth=3 condition=5' \
  >"${evidence_dir}/rollback.expected.marker"
cmp "${evidence_dir}/rollback.expected.marker" "${rollback_marker}" || \
  die "rollback allocation marker drifted"
[[ -z "$(find "${evidence_dir}/corpus" -mindepth 1 -print -quit)" ]] || \
  die "corpus rollback left a lock, temporary, shard, or manifest"

{
  printf 'runtime_commit\t%s\n' "${expected_commit}"
  printf 'runtime_tree\t%s\n' 7669312845a9d8d372006af52271045e69505813
  printf 'runtime_runner_sha256\t%s\n' "$(sha256sum "${runner}" | awk '{ print $1 }')"
  printf 'runtime_archive_sha256\t%s\n' "$(sha256sum "${runtime_archive}" | awk '{ print $1 }')"
  printf 'cli_object_sha256\t%s\n' "$(sha256sum "${cli_object}" | awk '{ print $1 }')"
  printf 'cli_archive_sha256\t%s\n' "$(sha256sum "${cli_archive}" | awk '{ print $1 }')"
  printf 'production_cli_sha256\t%s\n' "$(sha256sum "${production_cli}" | awk '{ print $1 }')"
  for item in cli-main.o cli3-archive-member.o allocation-fallback-shim.o \
      eshkol-transformer-allocation-fallback handler.stderr handler.marker \
      rollback.stderr rollback.marker allocation-fallback.map; do
    printf '%s_sha256\t%s\n' "${item//[^a-zA-Z0-9]/_}" \
      "$(sha256sum "${evidence_dir}/${item}" | awk '{ print $1 }')"
  done
} >"${evidence_dir}/sha256.tsv"
cat "${evidence_dir}/sha256.tsv"
printf 'CLI3 ALLOCATION FALLBACK PASS: real handler condition5, published-shard rollback, persistent constructor failure, static diagnostic\n'
