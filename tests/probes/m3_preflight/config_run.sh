#!/usr/bin/env bash
# Development-only public X1 profile probe. No schema extension or model execution.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/common.sh"
verify_toolchain
cd "$PROJECT_ROOT"
config_tmp=$(mktemp -d "${TMPDIR:-/tmp}/m3-config.XXXXXX")
trap 'rm -rf -- "$config_tmp"' EXIT
for attempt in 1 2; do
  mkdir -p "$config_tmp/cache-$attempt"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$config_tmp/cache-$attempt" \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
    "$(eshkol_build_dir)/eshkol-run" --strict-types --no-stdlib \
    -I "$PROJECT_ROOT/lib" -L "$(project_build_dir)/i2" \
    --lib eshkol_transformer_wave2 \
    "$PROJECT_ROOT/tests/probes/m3_preflight/config_profile.esk" \
    -o "$config_tmp/profile-$attempt" >"$config_tmp/compile-$attempt.log" 2>&1 || {
      cat "$config_tmp/compile-$attempt.log" >&2; exit 1;
    }
  "$config_tmp/profile-$attempt" >"$config_tmp/stdout-$attempt"
done
cmp "$config_tmp/stdout-1" "$config_tmp/stdout-2"
cat "$config_tmp/stdout-1"
printf 'M3 X1 profile: two fresh public AOT executions agree; no model capability claim.\n'
