#!/usr/bin/env bash
# Version 1 of the noninstalled, length-1/3 I2 post-prepare wrapper tuple.
# The unwrapped zero-control package retains its independent import policy.
e3_i2_guard_import_manifest() {
  local role=${1:-} expected count manifest actual
  [[ $# == 1 && "${package_policy}" == e3-private-aggregate &&
     "${e3_tuple_kind}" == zero &&
     ( "${e3_i2_guard_length}" == 1 || "${e3_i2_guard_length}" == 3 ) &&
     ( -z "${e3_i2_mutant}" || "${e3_i2_mutant}" == omit-abort ) ]] ||
    die "I2 guard import policy requires the exact wrapped zero tuple"
  case "${role}" in
    normal)
      expected=1a26fdc3079cdb2751bfec5d94d04cd3b1629dfe658c0e880267d1eb07a4f61d
      count=174 ;;
    sanitized_noninstrumentation)
      [[ "${e3_zero_sanitize}" == 1 ]] || die "I2 guard sanitizer imports require sanitizer mode"
      expected=aa53d29e6f6c8f70156549df349812c0fdc616493ec3fc11732414e854fc882e
      count=172 ;;
    sanitized_instrumentation)
      [[ "${e3_zero_sanitize}" == 1 ]] || die "I2 guard sanitizer imports require sanitizer mode"
      expected=d9cac2726c87ad5c01f6c9f8443e51103be630a65a69150d7fe9ae38f638f3f6
      count=50 ;;
    *) die "unknown I2 guard import inventory" ;;
  esac
  manifest="${PROJECT_ROOT}/tests/e3_private/i2_guard_v1_${role}_undefined_symbols.txt"
  e3_check_repository_path "${manifest}"
  [[ -f "${manifest}" && -s "${manifest}" && ! -L "${manifest}" ]] ||
    die "I2 guard import inventory is missing or nonregular"
  actual="$(sha256sum "${manifest}" | cut -d ' ' -f 1)"
  [[ "${actual}" == "${expected}" && "$(wc -l <"${manifest}")" == "${count}" ]] ||
    die "I2 guard import inventory differs from version 1"
  LC_ALL=C sort -u "${manifest}" | cmp -s "${manifest}" - ||
    die "I2 guard import inventory is not sorted unique text"
  printf '%s\n' "${manifest}"
}
