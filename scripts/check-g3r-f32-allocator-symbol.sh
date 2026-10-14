#!/usr/bin/env bash
set -euo pipefail

object="${1:?pass the F32 object}"
symbols="${2:?pass an evidence path for defined symbols}"
LC_ALL=C nm -g --defined-only --format=posix "$object" > "$symbols"
grep -Eq '^et_f32_tensor_test_fail_alloc_after_v1 T [[:xdigit:]]+ [[:xdigit:]]+$' \
  "$symbols"
