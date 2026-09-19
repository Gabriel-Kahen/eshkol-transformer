#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  cat >&2 <<'EOF'
usage: ci-build-prerequisites.sh [--plan|--verify-only] SUITE [SUITE ...]

Build the canonical artifacts read by the named CI suites before those suites
write their own fresh or deterministic-rebuild artifacts. Multiple suites may
be supplied; every producer is run at most once, in a fixed order.
EOF
}

mode=build
case "${1:-}" in
  --plan|--verify-only)
    mode="${1#--}"
    shift
    ;;
esac
[[ $# -gt 0 ]] || { usage; exit 2; }

declare -A selected=()

select_producers() {
  local producer
  for producer in "$@"; do
    selected["${producer}"]=1
  done
}

for suite in "$@"; do
  case "${suite}" in
    native-numerics)
      # K2 proves collisions against the I2, T2, D2, and O2 archives, so the
      # core partition retains O2 even though test-o2.sh runs separately.
      select_producers k1 a2 l2 i1 i2 k2 n2 n3k t2 d2 o2
      ;;
    model-composition)
      select_producers i2 m3
      ;;
    diagnostic-transport)
      select_producers i2 m3t
      ;;
    native-optimizer)
      # O2 reads its own aggregate, K1, and the I2/T2/D2 collision archives.
      select_producers k1 i2 t2 d2 o2
      ;;
    contracts-data)
      # A0 and D1 read the canonical D1 archive; X1 builds into a temporary.
      select_producers d1
      ;;
    checkpoint-io|parameter-state|bpe-tokenizer|c2-format|c2-state|c2-save|c2-load|c2-operational)
      # These suites build test-local artifacts and only need configuration.
      ;;
    byte-tokenizer)
      # Retained conservatively: the accepted reverse-import path has required
      # the current D2 aggregate in CI even though T1 builds its own aggregate.
      select_producers d2
      ;;
    bpe-boundary)
      # test-t2-boundary positively links build/d2/libeshkol_transformer_wave2.a.
      select_producers d2
      ;;
    shard-loader)
      # D2 reads its canonical private/aggregate outputs plus canonical I1/K1.
      select_producers k1 i1 d2
      ;;
    c2-public|acceptance-c2)
      # Only the public C2 gate reads the canonical C2 aggregate.
      select_producers c2
      ;;
    smoke-benchmark)
      # smoke-after-build and benchmark-after-build consume the canonical smoke
      # executable. The core test itself performs separate fresh compilations.
      select_producers smoke
      ;;
    acceptance-predecessors)
      # Union of predecessor read-before-write artifacts. X1, P1, C1, T1,
      # and the smoke test itself perform intentional test-local fresh builds.
      # Add smoke-benchmark when those post-suite checks run in the same job.
      select_producers k1 a2 l2 i1 i2 k2 n2 n3k t2 d2 o2 d1 m3t m3
      ;;
    *)
      printf 'error: unknown CI suite: %s\n' "${suite}" >&2
      usage
      exit 2
      ;;
  esac
done

producer_order=(smoke k1 a2 l2 i1 i2 k2 n2 n3k t2 d2 o2 d1 c2 m3t m3)

producer_script() {
  case "$1" in
    smoke) printf '%s\n' 'compile-smoke.sh' ;;
    *) printf 'build-%s.sh\n' "$1" ;;
  esac
}

verify_file() {
  local path=$1
  [[ -s "${path}" ]] || die "CI prerequisite artifact is missing or empty: ${path}"
}

verify_producer() {
  local producer=$1 build_dir
  build_dir="$(project_build_dir)"
  case "${producer}" in
    smoke)
      verify_file "${build_dir}/eshkol-transformer-smoke"
      [[ -x "${build_dir}/eshkol-transformer-smoke" ]] || \
        die "CI smoke prerequisite is not executable: ${build_dir}/eshkol-transformer-smoke"
      verify_file "${build_dir}/eshkol-transformer-smoke.o"
      verify_file "${build_dir}/eshkol-transformer-smoke.d"
      ;;
    k1)
      verify_file "${build_dir}/k1/kernel_abi.o"
      verify_file "${build_dir}/k1/libeshkol_transformer_k1.a"
      ;;
    a2)
      verify_file "${build_dir}/a2/a2_attention_provider.o"
      verify_file "${build_dir}/a2/a2_kv_cache.o"
      verify_file "${build_dir}/a2/libeshkol_transformer_a2.a"
      ;;
    l2)
      verify_file "${build_dir}/l2/indexed_cross_entropy.o"
      verify_file "${build_dir}/l2/libeshkol_transformer_l2.a"
      ;;
    i1)
      verify_file "${build_dir}/i1/i64_tensor.o"
      verify_file "${build_dir}/i1/libeshkol_transformer_i64.a"
      ;;
    i2)
      verify_file "${build_dir}/i2/f32_tensor.o"
      verify_file "${build_dir}/i2/libeshkol_transformer_f32.a"
      verify_file "${build_dir}/i2/i2_wave2.o"
      verify_file "${build_dir}/i2/libeshkol_transformer_wave2.a"
      verify_file "${build_dir}/i2/i2_wave2.o.evidence/global-defined.txt"
      ;;
    k2)
      verify_file "${build_dir}/k2/k2_wave2.o"
      verify_file "${build_dir}/k2/libeshkol_transformer_wave2.a"
      verify_file "${build_dir}/k2/k2_wave2.o.evidence/global-defined.txt"
      ;;
    n2)
      verify_file "${build_dir}/n2/n2_primitives_provider.o"
      verify_file "${build_dir}/n2/libeshkol_transformer_n2.a"
      ;;
    n3k)
      verify_file "${build_dir}/n3k/n3k_primitives_provider.o"
      verify_file "${build_dir}/n3k/n3k_primitives_provider.d"
      verify_file "${build_dir}/n3k/libeshkol_transformer_n3k.a"
      ;;
    t2)
      verify_file "${build_dir}/t2/wave2.o"
      verify_file "${build_dir}/t2/libeshkol_transformer_wave2.a"
      verify_file "${build_dir}/t2/wave2.o.evidence/global-defined.txt"
      ;;
    d2)
      verify_file "${build_dir}/d2/d2_native.o"
      verify_file "${build_dir}/d2/libeshkol_transformer_d2_private.a"
      verify_file "${build_dir}/d2/d2_wave2.o"
      verify_file "${build_dir}/d2/libeshkol_transformer_wave2.a"
      verify_file "${build_dir}/d2/d2_wave2.o.evidence/global-defined.txt"
      ;;
    o2)
      verify_file "${build_dir}/o2/o2_wave2.o"
      verify_file "${build_dir}/o2/libeshkol_transformer_wave2.a"
      verify_file "${build_dir}/o2/o2_wave2.o.evidence/global-defined.txt"
      ;;
    d1)
      verify_file "${build_dir}/d1/libeshkol_transformer_d1.a"
      verify_file "${build_dir}/d1/libeshkol_transformer_d1.a.evidence/global-defined.txt"
      ;;
    m3)
      verify_file "${build_dir}/m3/m3_package.o"
      verify_file "${build_dir}/m3/libeshkol_transformer_m3.a"
      verify_file "${build_dir}/m3/m3_package.o.evidence/global-defined.txt"
      verify_file "${build_dir}/m3/facades/transformer/model.esk"
      ;;
    m3t)
      verify_file "${build_dir}/m3t/m3t_package.o"
      verify_file "${build_dir}/m3t/libeshkol_transformer_m3t.a"
      verify_file "${build_dir}/m3t/m3t_package.o.evidence/global-defined.txt"
      verify_file "${build_dir}/m3t/facades/transformer/diagnostic_transport.esk"
      ;;
    c2)
      verify_file "${build_dir}/c2/c2_wave2.o"
      verify_file "${build_dir}/c2/libeshkol_transformer_wave2.a"
      verify_file "${build_dir}/c2/c2_wave2.o.evidence/global-defined.txt"
      ;;
    *) die "unknown CI prerequisite producer: ${producer}" ;;
  esac
}

if [[ "${mode}" == plan ]]; then
  printf 'clean\t%s\n' 'scripts/clean.sh'
  printf 'configure\t%s\n' 'scripts/configure.sh'
  for producer in "${producer_order[@]}"; do
    [[ -v "selected[${producer}]" ]] || continue
    printf 'producer\t%s\t%s\n' "${producer}" "scripts/$(producer_script "${producer}")"
  done
  exit 0
fi

if [[ "${mode}" == build ]]; then
  /usr/bin/bash "${PROJECT_ROOT}/scripts/clean.sh"
  /usr/bin/bash "${PROJECT_ROOT}/scripts/configure.sh"
  for producer in "${producer_order[@]}"; do
    [[ -v "selected[${producer}]" ]] || continue
    if [[ "${producer}" == smoke ]]; then
      /usr/bin/bash "${PROJECT_ROOT}/scripts/compile-smoke.sh" "$(project_build_dir)"
    else
      /usr/bin/bash "${PROJECT_ROOT}/scripts/build-${producer}.sh"
    fi
  done
fi

verify_file "$(project_build_dir)/toolchain-manifest.tsv"
for producer in "${producer_order[@]}"; do
  [[ -v "selected[${producer}]" ]] || continue
  verify_producer "${producer}"
done

printf 'CI prerequisite artifacts verified:'
for producer in "${producer_order[@]}"; do
  [[ -v "selected[${producer}]" ]] || continue
  printf ' %s' "${producer}"
done
printf '\n'
