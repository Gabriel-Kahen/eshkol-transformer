#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
artifact="${root}/build/dist5"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
archive="${scratch}/libeshkol_dist5.a"
warnings=(-Wall -Wextra -Werror -Wpedantic -I "${root}/include")

"${cc}" -std=c11 "${warnings[@]}" -O2 -ffp-contract=off -fno-fast-math \
  -fexcess-precision=standard -fstack-protector-all -fPIC \
  -c "${root}/native/dist5_collectives.c" -o "${scratch}/dist5_collectives.o"
ar rcsD "${archive}" "${scratch}/dist5_collectives.o"

expected=(
  et_dist5_allgather et_dist5_allreduce et_dist5_barrier et_dist5_broadcast
  et_dist5_comm_create et_dist5_comm_destroy et_dist5_reduce_scatter
  et_dist5_sync_gradients_f32
)
exports="$(nm -g --defined-only "${archive}" | awk 'NF == 3 {print $2, $3}' | LC_ALL=C sort)"
if [[ "${exports}" != "$(printf 'T %s\n' "${expected[@]}")" ]]; then
  printf 'unexpected DIST5 archive exports:\n%s\n' "${exports}" >&2
  exit 1
fi
imports="$(nm -u "${archive}" | awk '$1 == "U" {print $2}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    __errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|bcmp|calloc|\
    clock_gettime|close|fcntl|fegetenv|fesetenv|free|fstat|\
    getpeername|getsockname|getsockopt|malloc|memcmp|memcpy|memset|poll|\
    recv|send|shutdown) ;;
    *)
      printf 'unexpected DIST5 archive dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"
printf 'DIST5 archive audit: PASS (%s exports)\n' "${#expected[@]}"

cat > "${scratch}/header.cpp" <<'CPP'
#include "eshkol_transformer/dist5_collectives_abi.h"

int main() {
  et_dist5_comm_t *comm = nullptr;
  const auto init = et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR,
                                         0, 1, nullptr, 0, &comm);
  if (init != ET_DIST5_OK)
    return 1;
  const auto status = et_dist5_barrier(comm, 0, 1000);
  et_dist5_comm_destroy(comm);
  return status != ET_DIST5_OK;
}
CPP
"${cxx}" -std=c++17 "${warnings[@]}" "${scratch}/header.cpp" "${archive}" \
  -lm -o "${scratch}/header"
"${scratch}/header"
printf 'DIST5 C++17 header and linkage: PASS\n'
mv -f "${archive}" "${artifact}/"
