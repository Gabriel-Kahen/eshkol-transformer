#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?source root required}
out=${2:?fresh output directory required}
eshkol_runner=${ESHKOL_RUNNER:?set ESHKOL_RUNNER to a local eshkol-run}
cxx=${CXX:-/usr/bin/clang++}
test ! -e "$out"
mkdir -p "$out"
"$eshkol_runner" --version >"$out/eshkol-version.txt"
"$cxx" --version >"$out/cxx-version.txt"

for mode in normal sanitize; do
  mkdir -p "$out/$mode/cache"
  compiler=$cxx
  if [[ $mode == sanitize ]]; then
    compiler="$out/$mode/cxx-wrap"
    cat >"$compiler" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "$cxx") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
    chmod 0500 "$compiler"
  fi
  for fixture in on off legacy; do
    source="$source_root/tests/g3r/c4_record_codec_test.esk"
    if [[ $fixture == off ]]; then
      source="$source_root/tests/g3r/record_codec_feature_off_test.esk"
    elif [[ $fixture == legacy ]]; then
      source="$source_root/tests/g3r/record_codec_test.esk"
    fi
    mkdir -p "$out/$mode/cache/$fixture"
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="$out/$mode/cache/$fixture" \
      ESHKOL_CXX_COMPILER="$compiler" \
      ESHKOL_LIB_DIR="$source_root/lib" \
      timeout --foreground --signal=TERM --kill-after=5s 180s \
      "$eshkol_runner" --strict-types --optimize 0 --no-stdlib \
      -I "$source_root/internal/c1/lib" -I "$source_root/native" \
      "$source" -o "$out/$mode/$fixture" \
      >"$out/$mode/$fixture-compile.stdout" \
      2>"$out/$mode/$fixture-compile.stderr"
    nm "$out/$mode/$fixture" >"$out/$mode/$fixture.symbols"
    for run in first repeat; do
      if [[ $mode == sanitize ]]; then
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON=1 \
          timeout --foreground --signal=TERM --kill-after=5s 120s \
          "$out/$mode/$fixture" \
          >"$out/$mode/$fixture-$run.stdout" \
          2>"$out/$mode/$fixture-$run.stderr"
      else
        ESHKOL_ARENA_POISON=1 \
          timeout --foreground --signal=TERM --kill-after=5s 120s \
          "$out/$mode/$fixture" \
          >"$out/$mode/$fixture-$run.stdout" \
          2>"$out/$mode/$fixture-$run.stderr"
      fi
      test ! -s "$out/$mode/$fixture-$run.stderr"
    done
    cmp "$out/$mode/$fixture-first.stdout" \
        "$out/$mode/$fixture-repeat.stdout"
    test ! -s "$out/$mode/$fixture-compile.stderr"
  done
  grep -F 'g3r-c4-record-encode!' "$out/$mode/on.symbols" >/dev/null
  grep -F 'g3r-c4-record-decode!' "$out/$mode/on.symbols" >/dev/null
  if grep -F 'g3r-c4-record-' "$out/$mode/off.symbols" >/dev/null; then
    echo 'feature-off C4 private symbol leaked' >&2
    exit 1
  fi
  if grep -F 'g3r-c4-record-' "$out/$mode/legacy.symbols" >/dev/null; then
    echo 'legacy C2 exports C4 private symbol' >&2
    exit 1
  fi
  python3 "$source_root/tests/g3r/check_c4_record_codec_oracle.py" \
    "$out/$mode/on-first.stdout" >"$out/$mode/oracle.stdout"
  python3 "$source_root/tests/g3r/check_record_codec_oracle.py" \
    "$out/$mode/legacy-first.stdout" >"$out/$mode/legacy-oracle.stdout"
  grep -Eq '^G3-R C4 codec PASS: checks=[1-9][0-9]*$' \
    "$out/$mode/on-first.stdout"
  grep -Fx 'G3-R record codec off PASS: checks=2' \
    "$out/$mode/off-first.stdout" >/dev/null
  grep -Fx 'G3-R record codec PASS: checks=49' \
    "$out/$mode/legacy-first.stdout" >/dev/null
done
for fixture in on off legacy; do
  cmp "$out/normal/$fixture-first.stdout" \
      "$out/sanitize/$fixture-first.stdout"
done
cmp "$out/normal/oracle.stdout" "$out/sanitize/oracle.stdout"
cmp "$out/normal/legacy-oracle.stdout" \
    "$out/sanitize/legacy-oracle.stdout"

(
  cd "$source_root"
  sha256sum internal/c1/lib/c1_sha256.esk \
    native/g3r_record_codec_extension.esk \
    native/g3r_c4_record_codec_extension.esk \
    tests/g3r/c4_record_codec_test.esk \
    tests/g3r/record_codec_test.esk \
    tests/g3r/record_codec_feature_off_test.esk \
    tests/g3r/check_c4_record_codec_oracle.py \
    tests/g3r/check_record_codec_oracle.py \
    scripts/test-g3r-c4-record-codec-local.sh \
    docs/g3/G3_R_C4_RECORD_CODEC_LEAF.md \
    docs/g3/G3_R_TRAINED_C4_UPSTREAM_CONTRACT.md docs/ROADMAP.md
) >"$out/source.sha256"
(
  cd "$out"
  find normal sanitize -type f ! -path '*/cache/*' \
    ! -name 'cxx-wrap' -print0 | sort -z | xargs -0 sha256sum
  sha256sum eshkol-version.txt cxx-version.txt source.sha256
) >"$out/SHA256SUMS"
echo 'PASS: local G3-R C4 codec normal/repeat/sanitizer/feature-off/legacy-C2'
