#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp mktemp python3 sha256sum timeout; do
  require_command "${command}"
done

evidence="${T2_NATIVE_CORPUS_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/eshkol-t2-native-corpus.XXXXXX")}"
if [[ -d "${evidence}" && -n "$(ls -A -- "${evidence}")" ]]; then
  die "evidence directory must be empty: ${evidence}"
fi
mkdir -p -- "${evidence}"/{build,fixture,first,second,rejected,blocked}
evidence="$(realpath -- "${evidence}")"
cd -- "${PROJECT_ROOT}"

runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
CC="${cc}" CXX="$(tsv_value "${provenance}" cxx_path)" \
  bash scripts/build-t2-trainer.sh "${evidence}/build" normal >"${evidence}/archive.log" 2>&1

trainer="${evidence}/build/libeshkol_transformer_t2_trainer.a"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
        -Wsign-conversion -Wshadow -O2 -I include -I native -I src)
"${cc}" "${cflags[@]}" tests/t2_trainer/native_corpus_fixture.c "${trainer}" \
  -o "${evidence}/build/fixture"
"${evidence}/build/fixture" "${evidence}/fixture"

native_objects=()
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell p1_identity t2_native_bpe_import; do
  object="${evidence}/build/${source}.o"
  extra_flags=()
  if [[ "${source}" == p1_identity ]]; then
    extra_flags=(-DET_P1_TRUSTED_BUILD=1)
  fi
  "${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    -fstack-protector-all -fPIC -fvisibility=hidden -fno-common \
    "${extra_flags[@]}" -I include -I native -I src \
    -c "native/${source}.c" -o "${object}"
  native_objects+=("${object}")
done

witness="${evidence}/build/witness"
if ! timeout --foreground --kill-after=5s 300s "${runner}" \
  --strict-types --optimize 0 --no-stdlib \
  -I internal/p1/lib -I internal/c1/lib -I internal/t2/lib -I internal/t1/lib \
  -I internal/data4/lib -I src -I lib -I native \
  -L "${evidence}/build" --lib eshkol_transformer_t2_trainer \
  tests/t2_trainer/native_corpus_witness.esk "${native_objects[@]}" \
  -o "${witness}" >"${evidence}/compile.log" 2>&1 || [[ ! -x "${witness}" ]]; then
  tail -80 "${evidence}/compile.log" >&2
  die "native BPE corpus witness compilation failed"
fi

python3 - "${evidence}/fixture" <<'PY'
from pathlib import Path
import sys

fixture = Path(sys.argv[1])
artifact = (fixture / "trained.etbpe").read_bytes()
(fixture / "bad-checksum.etbpe").write_bytes(artifact[:-1] + bytes([artifact[-1] ^ 1]))
(fixture / "truncated.etbpe").write_bytes(artifact[:-1])
(fixture / "wrong-version.etbpe").write_bytes(artifact[:8] + b"\x02" + artifact[9:])
(fixture / "link.etbpe").symlink_to("trained.etbpe")
PY

run_witness() {
  local mode="$1" directory="$2"
  if ! timeout --foreground --kill-after=5s 60s "${witness}" \
    "${mode}" "${evidence}/fixture" "${evidence}/${directory}" \
    >"${evidence}/${directory}.stdout" 2>"${evidence}/${directory}.stderr" ||
    [[ -s "${evidence}/${directory}.stderr" ]]; then
    cat "${evidence}/${directory}."{stdout,stderr} >&2
    die "native BPE corpus witness failed in ${mode} mode"
  fi
}

run_witness write first
run_witness write second
cmp -- "${evidence}"/{first,second}.stdout
diff -r -- "${evidence}"/{first,second}

run_witness negative rejected
[[ -z "$(ls -A -- "${evidence}/rejected")" ]] || die "rejected import published corpus files"

printf 'existing temporary\n' >"${evidence}/reserved"
cp -- "${evidence}/reserved" "${evidence}/blocked/manifest.etm.tmp"
run_witness blocked blocked
cmp -- "${evidence}/reserved" "${evidence}/blocked/manifest.etm.tmp"
[[ "$(ls -A -- "${evidence}/blocked")" == manifest.etm.tmp ]] ||
  die "blocked D1 write changed corpus directory"

cp -a -- "${evidence}"/{first,corrupt}
python3 - "${evidence}/corrupt/shard-0000000000000000.ets" <<'PY'
from pathlib import Path
import sys

shard = Path(sys.argv[1])
contents = shard.read_bytes()
shard.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
PY
run_witness corrupt corrupt

(
  cd -- "${evidence}"
  sha256sum archive.log compile.log build/witness fixture/trained.etbpe \
    fixture/native.i64 first/* second/* *.stdout > SHA256SUMS
  sha256sum --check --quiet SHA256SUMS
)
cat "${evidence}/"{first,rejected,blocked,corrupt}.stdout
printf 'T2 native corpus witness passed; evidence: %s\n' "${evidence}"
