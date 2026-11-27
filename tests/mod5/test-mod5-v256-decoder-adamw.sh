#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/common.sh"

[[ "${MOD5_V256_DECODER_ADAMW_SLOT:-0}" == 1 ]] ||
  die "MOD5 V256 decoder AdamW AOT requires an assigned build slot"
[[ $# == 1 && ! -e "$1" && ! -L "$1" ]] || die "expected a fresh evidence path"
[[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" == 0 ]] || die "expected the supported toolchain"
for command in ar cmp git python3 sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]] ||
  die "expected a clean source checkout"
mkdir -p -- "$1"
out="$(realpath -- "$1")"
cd "${PROJECT_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1

cc= cxx=
resolve_provenance_compilers cc cxx "$(lock_value supported_cc)" "$(lock_value supported_cxx)"
runner="$(eshkol_build_dir)/eshkol-run"
revision="$(git show -s --format='commit %H%ntree %T')"
{
  printf '%s\n' "$revision"
  sha256sum toolchain/eshkol.lock "$runner"
  "$cc" --version
} >"$out/provenance.txt"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "$out/toolchain.tsv"

python3 -B -m tests.mod5.mod5_v256_decoder_adamw build "$out"
mkdir "$out/objects" "$out/cache"
flags=(-std=c11 -O2 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
       -Wshadow -ffp-contract=off -fexcess-precision=standard -frounding-math
       -fno-fast-math -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
       -I include -I native -I tests/mod5)
compile() {
  local source=$1
  shift
  "$cc" "${flags[@]}" "$@" -c "$source" -o "$out/objects/$(basename "${source%.c}").o"
}
{
  compile native/i2_wave2_package_bridge.c -DET_I2_NATIVE_HELPERS_ONLY
  compile native/o2_wave2_package_bridge.c -DET_O2_NATIVE_HELPERS_ONLY
  compile native/p1_identity.c -DET_P1_TRUSTED_BUILD=1
  for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell f32_tensor \
                d2_native o2_optimizer mod5_eshkol_bridge mod5_primitives_provider; do
    compile "native/$source.c"
  done
  compile tests/mod5/mod5_v256_c2_transfer.c
  compile tests/mod5/mod5_v256_decoder_adamw_bridge.c
  ar rcsD "$out/libmod5_v256_decoder_adamw.a" "$out/objects/"*.o
  "$cc" "${flags[@]}" tests/mod5/test_mod5_v256_decoder_adamw_bridge.c \
    "$out/libmod5_v256_decoder_adamw.a" -lm -o "$out/native-adapters"
  "$out/native-adapters"
} >"$out/native.stdout" 2>"$out/native.stderr"
test ! -s "$out/native.stderr"

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="$out/cache" ESHKOL_LIB_DIR="$PROJECT_ROOT/lib" \
  ESHKOL_CXX_COMPILER="$cxx" timeout --foreground --kill-after=5s 900s "$runner" \
  --strict-types --no-stdlib -O 0 -I native -I internal/p1/lib -I internal/c1/lib \
  -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib -I src -I lib \
  -L "$out" --lib mod5_v256_decoder_adamw "$out/witness.esk" -o "$out/witness" \
  >"$out/compile.stdout" 2>"$out/compile.stderr"
test -x "$out/witness"
test ! -s "$out/compile.stderr"

for variant in untied tied; do
  model="$out/evaluation/model.etcp"
  if [[ "$variant" == tied ]]; then model="$out/tied.etcp"; fi
  for replay in a b; do
    prefix="$out/$variant-$replay"
    ESHKOL_ARENA_POISON=1 timeout --foreground --kill-after=5s 180s "$out/witness" \
      "$out/training" "$model" "$variant" "$prefix.bits" >"$prefix.stdout" 2>"$prefix.stderr"
    test ! -s "$prefix.stderr"
    python3 -B -m tests.mod5.mod5_v256_decoder_adamw verify "$out" "$variant" \
      "$prefix.stdout" "$prefix.bits" >"$prefix.oracle"
  done
  for extension in stdout bits oracle; do
    cmp "$out/$variant-a.$extension" "$out/$variant-b.$extension"
  done
  cat "$out/$variant-a.oracle"
done

[[ "$revision" == "$(git show -s --format='commit %H%ntree %T')" &&
   -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "source changed during the gate"
(cd "$out" && sha256sum provenance.txt toolchain.tsv training/fixture.json evaluation/fixture.json \
    tied.etcp witness.esk witness libmod5_v256_decoder_adamw.a native.std* compile.std* \
    {untied,tied}-{a,b}.{stdout,stderr,bits,oracle} >SHA256SUMS)
