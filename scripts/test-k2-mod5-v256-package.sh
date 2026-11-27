#!/usr/bin/env bash
# K2_DEFAULT_EVIDENCE_DIR selects a separately built canonical package receipt.
# K2_V256_SCHEME_DISCOVERY=1 also runs the strict AOT dependency discovery.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in awk cmp find grep nm python3 readelf realpath sha256sum sort strings wc xargs; do
  require_command "$command"
done
cc= cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
out="${1:-$(project_build_dir)/k2-mod5-v256-package}"
discover="${K2_V256_SCHEME_DISCOVERY:-0}"
[[ "$discover" == 0 || "$discover" == 1 ]] || die "set K2_V256_SCHEME_DISCOVERY to 0 or 1"
[[ ! -e "$out" && ! -L "$out" ]] || die "K2 package evidence directory must be fresh"
mkdir -p "$out"
out="$(realpath "$out")"
cd "$PROJECT_ROOT"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
git log -1 --format='%H %T' >"$out/revision.txt"
git status --porcelain >"$out/worktree-status.txt"
python3 -B -m unittest tests.k2.test_mod5_v256_package >"$out/source-tests.txt" 2>&1

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
  -D_POSIX_C_SOURCE=200809L -fPIC -fvisibility=hidden -fno-common
  -fstack-protector-all -I include -I native -I "$(eshkol_source_dir)/inc")
v256=(-DET_K2_PRIVATE_MOD5_V256 -DET_I2_PRIVATE_MOD5_WIDTH16_STORAGE_COPY
  -DET_I2_PRIVATE_MOD5_WIDTH16_V256_STORAGE_COPY)

inventory() {
  readelf --wide --syms "$1" | awk '
    NF == 8 && $5 ~ /^(GLOBAL|WEAK|UNIQUE)$/ {
      print ($7 == "UND" ? "U" : "D"), $4, $5, $6, $8
    }' | sort -u
}

for profile in default v256; do
  layout=()
  if [[ "$profile" == v256 ]]; then layout=("${v256[@]}"); fi
  dir="$out/$profile"
  mkdir "$dir"
  for source in k2_capabilities f32_tensor kernel_abi; do
    "$cc" "${flags[@]}" "${layout[@]}" -MMD -MF "$dir/$source.d" \
      -c "native/$source.c" -o "$dir/$source.o"
    inventory "$dir/$source.o" >"$dir/$source.symbols"
  done
  "$cc" -shared -Wl,--no-undefined "$dir"/*.o -lm -o "$dir/native.so"
  nm -D --defined-only --format=posix "$dir/native.so" |
    awk '{print $1}' | sort -u >"$dir/native.exports"
  nm -D --undefined-only --format=posix "$dir/native.so" |
    awk '{print $1}' | sort -u >"$dir/native.imports"
  strings -a -n 3 "$dir/k2_capabilities.o" "$dir/f32_tensor.o" |
    grep -E '^(I2:[[:alnum:].:-]+|eshkol-transformer-f32|storage\.copy|tensor\.f32|1\.[0-9]+)$' |
    sort -u >"$dir/native.interface-strings"
done

"$cc" "${flags[@]}" "${v256[@]}" tests/k2/mod5_v256_package_report.c \
  "$out/v256/"*.o -Wl,--wrap=et_kernel_runtime_discover -lm -o "$out/report"
for profile in default v256; do
  "$out/report" "$profile" >"$out/$profile/report.json"
  "$out/report" "$profile" >"$out/$profile/report-repeat.json"
  cmp "$out/$profile/report.json" "$out/$profile/report-repeat.json"
done
[[ "$(wc -c <"$out/v256/report.json")" == 3225 ]] ||
  die "K2 V256 report has incorrect length"
(cd "$out" && sha256sum -c <<'CHECKSUMS'
50c7078af4d3e495c4d668b6b8aaab29024cbab6a775fa6e98c9e9d10124389e  default/report.json
50e09d8e0bf757492e1a0c10aacaf58ba124d18f6e48b65cf3f21f41dd288ab6  v256/report.json
CHECKSUMS
)

for profile in default v256; do
  manifest=native/k2_mod5_v256
  if [[ "$profile" == default ]]; then manifest="${manifest}_default"; fi
  cmp "${manifest}_native_symbols.txt" "$out/$profile/k2_capabilities.symbols"
  cmp "${manifest}_interface_strings.txt" "$out/$profile/native.interface-strings"
done
cmp native/k2_mod5_v256_native_exports.txt "$out/default/native.exports"
cmp "$out/default/native.exports" "$out/v256/native.exports"
cmp "$out/default/native.imports" "$out/v256/native.imports"
cmp "$out/default/kernel_abi.symbols" "$out/v256/kernel_abi.symbols"
awk '$1 == "D" {print $5}' "$out/default/f32_tensor.symbols" |
  sort | cmp - native/f32_tensor_defined_symbols.txt
awk '$1 == "U" {print $5}' "$out/default/f32_tensor.symbols" |
  sort | cmp - native/f32_tensor_undefined_symbols.txt
{
  cat "$out/default/f32_tensor.symbols"
  printf 'D FUNC GLOBAL HIDDEN %s\n' \
    et_i2_private_mod5_width16_storage_provider_v1 \
    et_i2_private_mod5_width16_v256_storage_provider_v1
} | sort -u | cmp - "$out/v256/f32_tensor.symbols"

python3 -B -m tests.k2.check_mod5_v256_package native "$out"
if [[ -n "${K2_DEFAULT_EVIDENCE_DIR:-}" ]]; then
  python3 -B -m tests.k2.check_mod5_v256_package default "$K2_DEFAULT_EVIDENCE_DIR" \
    >"$out/default-canonical-check.sha256"
fi
if [[ "$discover" == 1 ]]; then
  bash scripts/test-k2-mod5-v256.sh "$out/scheme"
  python3 -B -m tests.k2.check_mod5_v256_package scheme "$out/scheme"
fi

git log -1 --format='%H %T' | cmp - "$out/revision.txt"
git status --porcelain | cmp - "$out/worktree-status.txt"
sha256sum "$(eshkol_build_dir)/eshkol-run" "$cc" "$cxx" >"$out/toolchain.sha256"
sha256sum native/k2_mod5_v256*.txt native/k2_wave2_*.txt \
  scripts/{common,test-k2-mod5-v256,test-k2-mod5-v256-package}.sh \
  tests/k2/{check_mod5_v256_package.py,test_mod5_v256_package.py,mod5_v256_package_report.c,mod5_v256_runtime.esk} \
  toolchain/eshkol.lock >"$out/package-inputs.sha256"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
  xargs -0 sha256sum >SHA256SUMS)
printf 'K2 MOD5 V256 native package boundary PASS\n'
