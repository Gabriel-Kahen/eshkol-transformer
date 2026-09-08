#!/usr/bin/env bash
# Diagnostic only. Requires the repository's compiler and built I2 aggregate.
set -euo pipefail
api_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$api_root"
source "$api_root/scripts/common.sh"
# Unsupported hosts require the caller's explicit override, as in normal gates.
verify_toolchain
for api_command in timeout mktemp nm awk sort cmp rg; do
  require_command "$api_command"
done
api_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
api_cc=$(tsv_value "$api_provenance" cc_path)
api_runner="$(eshkol_build_dir)/eshkol-run"
api_i2="$(project_build_dir)/i2"
api_timeout="${M3_API_COMPILER_TIMEOUT_SECONDS:-120}"
[[ "$api_timeout" =~ ^[1-9][0-9]*$ ]] || die 'M3_API_COMPILER_TIMEOUT_SECONDS must be positive'
mkdir -p "$(project_build_dir)/m3-preflight"
api_output=$(mktemp -d "$(project_build_dir)/m3-preflight/api.XXXXXX")
printf 'M3 API diagnostic: compiler=%s; native-cc=%s; AOT-cxx=%s\n' \
  "$api_runner" "$api_cc" "$ESHKOL_CXX_COMPILER"
printf 'M3 API diagnostic: logs=%s; no public model or supported-lane claim\n' "$api_output"
api_flags=(--strict-types --optimize 0 --no-stdlib -I "$api_root/lib")
for api_probe in api_missing_model api_private_constructor api_private_token_read; do
  for api_phase in object aot; do
    api_phase_flags=()
    [[ "$api_phase" != object ]] || api_phase_flags+=(--emit-object)
    api_artifact="$api_output/$api_probe-$api_phase"
    mkdir -p "$api_output/cache-$api_probe-$api_phase"
    if env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      ESHKOL_LIB_DIR="$api_root/lib" \
      XDG_CACHE_HOME="$api_output/cache-$api_probe-$api_phase" \
      timeout --foreground --signal=TERM --kill-after=5s "${api_timeout}s" \
      "$api_runner" "${api_flags[@]}" "${api_phase_flags[@]}" \
      "tests/probes/m3_preflight/$api_probe.esk" -o "$api_artifact" \
      >"$api_artifact.log" 2>&1; then
      printf 'Unexpected compile success: %s %s\n' "$api_probe" "$api_phase" >&2
      exit 1
    else
      api_status=$?
    fi
    [[ "$api_status" == 1 ]] || die "unexpected compiler exit $api_status: $api_probe $api_phase"
    [[ ! -e "$api_artifact" ]]
    case "$api_probe" in
      api_missing_model) rg -q 'Unresolved module dependency' "$api_artifact.log" ;;
      api_private_constructor) rg -q 'Unknown function: i2-module-create-internal' "$api_artifact.log" ;;
      api_private_token_read) rg -q 'Unknown function: t1-shell-read' "$api_artifact.log" ;;
    esac
    printf 'M3 API NEGATIVE PASS: %s %s\n' "$api_probe" "$api_phase"
  done
done
mkdir -p "$api_output/cache-positive"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  ESHKOL_LIB_DIR="$api_root/lib" \
  XDG_CACHE_HOME="$api_output/cache-positive" \
  timeout --foreground --signal=TERM --kill-after=5s "${api_timeout}s" \
  "$api_runner" "${api_flags[@]}" -L "$api_i2" \
  --lib eshkol_transformer_wave2 \
  tests/probes/m3_preflight/api_public_positive.esk \
  -o "$api_output/public-positive" >"$api_output/public-positive.log" 2>&1
timeout --foreground --signal=TERM --kill-after=5s 60s "$api_output/public-positive"
nm -g --defined-only --format=posix "$api_i2/i2_wave2.o" | \
  awk '{print $1}' | LC_ALL=C sort >"$api_output/aggregate-globals.txt"
cmp native/i2_wave2_defined_symbols.txt "$api_output/aggregate-globals.txt"
printf 'M3 API SYMBOL PASS: exact existing 47-global I2 aggregate\n'
api_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -frounding-math -DET_F32_TENSOR_TESTING
  -I include -I native)
timeout --foreground --signal=TERM --kill-after=5s "${api_timeout}s" \
  "$api_cc" "${api_cflags[@]}" \
  tests/probes/m3_preflight/api_parameter_lifetime.c \
  native/f32_tensor.c native/kernel_abi.c -o "$api_output/parameter-lifetime"
timeout --foreground --signal=TERM --kill-after=5s 60s "$api_output/parameter-lifetime"
timeout --foreground --signal=TERM --kill-after=5s "${api_timeout}s" \
  "$api_cc" "${api_cflags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer tests/probes/m3_preflight/api_parameter_lifetime.c \
  native/f32_tensor.c native/kernel_abi.c -o "$api_output/parameter-lifetime-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s "$api_output/parameter-lifetime-sanitized"
