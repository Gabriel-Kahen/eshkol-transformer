#!/usr/bin/env bash
# Explicit fe9 runtime witness, not a toolchain pin or production builder.
set -euo pipefail
m3h_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 3 ]] || { echo 'usage: test-m3cg-repaired-handler.sh RUNTIME_SOURCE RUNTIME_BUILD OUTPUT' >&2; exit 2; }
m3h_source="$(realpath "$1")"
m3h_build="$(realpath "$2")"
mkdir -p "$3"
m3h_out="$(realpath "$3")"
m3h_expected_runtime=32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e
m3h_expected_runner=7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa
m3h_expected_provenance=4b2e87315b91dd27f6453862a4c4ebcd9fa1f577619d3bfa6ceca92964a9824c
[[ "$(git -c safe.directory="${m3h_source}" -C "${m3h_source}" rev-parse HEAD)" == fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9 ]]
[[ "$(git -c safe.directory="${m3h_source}" -C "${m3h_source}" rev-parse HEAD^{tree})" == 66c21f7ec19b1b4a42199fa30ed8e0e9727021bf ]]
[[ -z "$(git -c safe.directory="${m3h_source}" -C "${m3h_source}" status --porcelain=v1)" ]]
[[ "$(sha256sum "${m3h_build}/libeshkol-runtime.a" | cut -d' ' -f1)" == "${m3h_expected_runtime}" ]]
[[ "$(sha256sum "${m3h_build}/eshkol-run" | cut -d' ' -f1)" == "${m3h_expected_runner}" ]]
[[ "$(sha256sum "${m3h_build}/eshkol-transformer-provenance.tsv" | cut -d' ' -f1)" == "${m3h_expected_provenance}" ]]
[[ "$(clang-21 -dumpversion)" == 21.1.8 ]]
m3h_env=(env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH
 -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX -u COMPILER_PATH
 -u LIBRARY_PATH -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX
 -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0
 ESHKOL_LIB_DIR="${m3h_root}/lib" ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
 XDG_CACHE_HOME="${m3h_out}/cache")
mkdir -p "${m3h_out}/cache"
cd "${m3h_out}"
"${m3h_env[@]}" timeout --foreground --signal=TERM --kill-after=5s 900s \
 "${m3h_build}/eshkol-run" --strict-types --no-stdlib \
 -I "${m3h_root}/internal/p1/lib" -I "${m3h_root}/internal/c1/lib" \
 -I "${m3h_root}/internal/t1/lib" -I "${m3h_root}/src" \
 -I "${m3h_root}/lib" -I "${m3h_root}/native" -L "${m3h_build}" \
 --shared-lib --dump-ir --emit-depfile "${m3h_out}/private.d" \
 "${m3h_root}/tests/m3cg/handler_failure_root.esk" -o private
clang-21 -c -x ir private.ll -o private.o
cat "${m3h_root}/native/"{e1b_private_renames,x1_config_private_renames,i2_wave2_p1_renames,d1_e1b_private_renames,t1_wave1_private_renames,i2_wave2_private_renames}.txt >renames.txt
sed 's/^m3t-public-/g3t-checked-m3t-/' "${m3h_root}/native/m3_package_private_renames.txt" >>renames.txt
printf 'm3cg-run et_m3cg_test_run_cabi_v1\nm3cg-handler-run et_m3cg_handler_run_cabi_v1\n' >>renames.txt
objcopy --redefine-syms=renames.txt private.o
m3h_sources=(native/e1b_error_consumer_bridge.c tests/m3cg/handler_failure_bridge.c
 native/data_io.c native/checkpoint_io.c native/kernel_abi.c
 src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
 src/eshkol_transformer/m3_call_f32_integration.c tests/m3cg/model_harness.c
 native/n2_primitives_provider.c native/n3k_primitives_provider.c native/a2_attention_provider.c)
m3h_objects=(private.o)
for m3h_i in "${!m3h_sources[@]}"; do
 "${m3h_env[@]}" clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 \
 -fPIC -fstack-protector-all -ffp-contract=off -fexcess-precision=standard \
 -fno-fast-math -frounding-math -DET_F32_TENSOR_TESTING -DET_M3T_PACKAGE_BUILD \
 -I "${m3h_root}/include" -I "${m3h_root}/native" -I "${m3h_root}/src" \
 -I "${m3h_source}/inc" -MMD -MF "native-${m3h_i}.d" \
 -c "${m3h_root}/${m3h_sources[m3h_i]}" -o "native-${m3h_i}.o"
 m3h_objects+=("native-${m3h_i}.o")
done
# Inspect actual raw dependencies: the guard is loaded from this worktree.
for m3h_input in native/m3_package_root.esk native/m3_model_extension.esk native/m3_call_adapters.esk tests/m3cg/witness.esk tests/m3cg/handler_failure.esk; do
 grep -F "${m3h_root}/${m3h_input}" private.d >/dev/null
done
# Bind every Eshkol input reported by the compiler, every source compiled by
# this script, and the complete reviewed delta. Paths are repository-relative,
# sorted under the C locale, and hashed from the read-only source mount.
m3h_reviewed=(
 docs/M3_CALL_GUARD_ORDER.md
 docs/ROADMAP.md
 native/m3t_transport_extension.esk
 scripts/test-m3cg-repaired-handler.sh
 tests/m3cg/predecessor_sources.sha256
 tests/m3cg/test_contract.py
 tests/m3cg/handler_failure.esk
 tests/m3cg/handler_failure_bridge.c
 tests/m3cg/handler_failure_caller.esk
 tests/m3cg/handler_failure_root.esk
 tests/m3cg/handler_failure_shim.cpp
)
{
 while IFS= read -r m3h_input; do
  [[ "${m3h_input}" == "${m3h_root}/"* ]]
  m3h_relative="${m3h_input#"${m3h_root}/"}"
  [[ -n "${m3h_relative}" && "${m3h_relative}" != *../* ]]
  printf '%s\n' "${m3h_relative}"
 done < <(sed -e '1s/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*\\$//' private.d |
          tr '[:space:]' '\n' | sed '/^$/d')
 printf '%s\n' "${m3h_sources[@]}"
 printf '%s\n' tests/m3cg/handler_failure_shim.cpp tests/m3cg/handler_failure_caller.esk
 printf '%s\n' "${m3h_reviewed[@]}"
} | LC_ALL=C sort -u >source-inputs.txt
(
 cd "${m3h_root}"
 while IFS= read -r m3h_input; do
  [[ -f "${m3h_input}" ]]
  sha256sum -- "${m3h_input}"
 done <"${m3h_out}/source-inputs.txt"
) >source-inputs.sha256
clang++-21 -r "${m3h_objects[@]}" -o aggregate.o
{ cat "${m3h_root}/native/m3_package_defined_symbols.txt"; printf 'et_m3cg_test_run_v1\net_m3cg_handler_run_v1\n'; } | LC_ALL=C sort >exports.txt
nm -g --defined-only --format=posix aggregate.o | awk 'NR==FNR {a[$1]=1;next} !($1 in a) {print $1}' exports.txt - >localize.txt
objcopy --localize-symbols=localize.txt aggregate.o
nm -g --defined-only --format=posix aggregate.o | awk '{print $1}' | LC_ALL=C sort >globals.txt
cmp exports.txt globals.txt
"${m3h_env[@]}" clang++-21 -std=c++17 -Wall -Wextra -Werror -Wpedantic -O2 \
 -I "${m3h_source}/inc" -c "${m3h_root}/tests/m3cg/handler_failure_shim.cpp" -o shim.o
"${m3h_env[@]}" timeout --foreground --signal=TERM --kill-after=5s 180s \
 "${m3h_build}/eshkol-run" --strict-types --no-stdlib --compile-only \
 "${m3h_root}/tests/m3cg/handler_failure_caller.esk" -o caller.o
clang++-21 caller.o aggregate.o shim.o "${m3h_build}/libeshkol-runtime.a" \
 -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler -Wl,--wrap=eshkol_get_raised_value \
 -Wl,-Map,witness.map -lpng -ljpeg -lwebp -lz -lopenblas -lcrypto -lpthread -ldl -lm -o witness
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 180s \
 ./witness >stdout 2>stderr
printf '%s\n' \
 'M3CG handler: case=prepublish pushes=2 condition5-reads=2 body=0 idle=1 retry=1' \
 'M3CG handler: case=body pushes=2 condition5-reads=3 body=1 idle=1 retry=1' \
 'M3CG-REPAIRED-HANDLER-AND-46-ENTRY-PASS' >expected
cmp expected stdout
[[ ! -s stderr ]]
sha256sum "${m3h_build}/libeshkol-runtime.a" "${m3h_build}/eshkol-run" \
 "${m3h_build}/eshkol-transformer-provenance.tsv" \
 "${m3h_root}/native/m3_model_extension.esk" "${m3h_root}/native/m3_call_adapters.esk" \
 "${m3h_root}/tests/m3cg/handler_failure_shim.cpp" source-inputs.sha256 \
 private.ll aggregate.o witness >hashes.txt
sha256sum stdout stderr expected hashes.txt source-inputs.txt source-inputs.sha256 \
 private.d witness.map globals.txt exports.txt >evidence-hashes.txt
sha256sum evidence-hashes.txt >evidence-hashes.sha256
cat stdout
cat hashes.txt
cat evidence-hashes.sha256
