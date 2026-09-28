#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evidence="${1:-$project_root/build/g3t-manual-logits-materialize-native}"
compiler="${CC:-clang}"
mkdir -p "$evidence"
"$compiler" --version >"$evidence/compiler.txt"
python3 "$project_root/scripts/check-g3t-manual-logits-materialize.py" >"$evidence/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.q0.test_python_isolation \
  >"$evidence/q0.stdout" 2>"$evidence/q0.stderr"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffunction-sections
  -fdata-sections -I "$project_root" -I "$project_root/include"
  -I "$project_root/native" -I "$project_root/src"
  -I "$project_root/src/eshkol_transformer")
for mode in normal sanitize; do
  flags=(-O2)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$compiler" "${common[@]}" "${flags[@]}" -DET_F32_TENSOR_TESTING \
    -Wl,--gc-sections "$project_root/tests/g3t/test_manual_logits_materialize.c" \
    "$project_root/native/f32_tensor.c" "$project_root/native/kernel_abi.c" \
    -o "$evidence/$mode"
  if [[ "$mode" == sanitize ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      "$evidence/$mode" >"$evidence/$mode.stdout" 2>"$evidence/$mode.stderr"
  else
    "$evidence/$mode" >"$evidence/$mode.stdout" 2>"$evidence/$mode.stderr"
  fi
  test ! -s "$evidence/$mode.stderr"
done
"$evidence/normal" >"$evidence/repeat.stdout" 2>"$evidence/repeat.stderr"
test ! -s "$evidence/repeat.stderr"
cmp "$evidence/normal.stdout" "$evidence/repeat.stdout"
cmp "$evidence/normal.stdout" "$evidence/sanitize.stdout"
for state in on off; do
  flags=(-DET_G3T_PREFILL_SAMPLE_PRIVATE -DET_G3T_MANUAL_LOGITS_PRIVATE)
  if [[ "$state" == on ]]; then
    flags+=(-DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE)
  fi
  "$compiler" "${common[@]}" -O2 "${flags[@]}" \
    -c "$project_root/src/eshkol_transformer/g3t_transport.c" \
    -o "$evidence/feature-$state.o"
done
nm -g --defined-only "$evidence/feature-on.o" | \
  grep 'et_g3t_private_logits_copy_bits_v1' >"$evidence/feature-on.symbol"
if nm -g --defined-only "$evidence/feature-off.o" | \
   grep 'et_g3t_private_logits_copy_bits_v1' >"$evidence/feature-off.symbol"; then
  echo "feature-off snapshot symbol escaped" >&2
  exit 1
fi
for state in on off; do
  nm -g --defined-only "$evidence/feature-$state.o" | \
    awk '{print $NF}' | sort >"$evidence/feature-$state.defined"
done
diff -u "$evidence/feature-off.defined" \
  <(sed '/^et_g3t_private_logits_copy_bits_v1$/d' "$evidence/feature-on.defined")
sha256sum "$project_root/src/eshkol_transformer/g3t_transport.c" \
  "$project_root/src/eshkol_transformer/g3t_transport.h" \
  "$project_root/native/g3t_manual_logits_materialize_extension.esk" \
  "$project_root/native/g3t_manual_logits_materialize_local_symbols.txt" \
  "$project_root/scripts/check-g3t-manual-logits-materialize.py" \
  "$project_root/scripts/test-g3t-manual-logits-materialize-native.sh" \
  "$project_root/tests/g3t/test_manual_logits_materialize.c" \
  "$project_root/tests/g3t/p2_zero_budget_test.esk" \
  "$project_root/tests/q0/test_python_isolation.py" \
  "$project_root/native/g3t_manual_logits_materialize_source_closure.txt" \
  >"$evidence/source.sha256"
git -C "$project_root" rev-parse HEAD >"$evidence/head.txt"
git -C "$project_root" rev-parse HEAD^{tree} >"$evidence/tree.txt"
sha256sum "$evidence"/*.stdout "$evidence"/*.stderr \
  "$evidence"/*.symbol "$evidence"/*.defined "$evidence"/source.sha256 \
  "$evidence"/head.txt "$evidence"/tree.txt "$evidence"/compiler.txt \
  >"$evidence/SEAL.sha256"
cat "$evidence/normal.stdout"
printf 'Evidence: %s\n' "$evidence"
