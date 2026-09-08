#!/usr/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
probe_directory=$(mktemp -d "${TMPDIR:-/tmp}/m3-numerical.XXXXXX")
trap 'rm -rf "$probe_directory"' EXIT
compiler=${CC:-clang}
flags=(-std=c11 -Wall -Wextra -Werror -O2 -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -Iinclude)

cat <<'NOTICE'
M3-PREFLIGHT DEVELOPMENT PROBES
Unsupported rows must reject explicitly; this does not execute a complete decoder.
Native probes do not establish public Eshkol reachability, supported-lane acceptance,
general autodiff, acceleration, or a broader shape/precision/device capability.
NOTICE
"$compiler" --version

"$compiler" "${flags[@]}" tests/probes/m3_preflight/numerical_rows.c \
  native/kernel_abi.c native/n2_primitives_provider.c \
  native/a2_attention_provider.c native/indexed_cross_entropy.c \
  -lm -o "$probe_directory/rows"
"$probe_directory/rows"

"$compiler" "${flags[@]}" tests/n2/test_primitives_provider.c \
  native/kernel_abi.c native/n2_primitives_provider.c \
  -lm -o "$probe_directory/n2"
"$probe_directory/n2"

"$compiler" "${flags[@]}" tests/a2/test_attention_provider.c \
  native/kernel_abi.c native/a2_attention_provider.c \
  -lm -o "$probe_directory/a2"
"$probe_directory/a2"

"$compiler" "${flags[@]}" tests/l2/test_indexed_cross_entropy.c \
  native/kernel_abi.c native/indexed_cross_entropy.c \
  -lm -o "$probe_directory/l2"
"$probe_directory/l2"
