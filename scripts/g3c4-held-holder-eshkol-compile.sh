#!/usr/bin/env bash

# Compile one held-holder fixture and capture dependencies from the supported
# object mode. The pinned runner does not emit depfiles for executables.
compile_eshkol_fixture() {
  local fixture=$1 input=$2 directory=$3 evidence=$4 mode=$5 runner=$6
  local cxx=$7 project_root=$8
  local -a arguments=(--strict-types --optimize 0 --no-stdlib
    -I "$project_root/internal/p1/lib" -I "$project_root/internal/c1/lib"
    -I "$project_root/internal/t1/lib" -I "$project_root/src"
    -I "$project_root/lib" -I "$project_root/native"
    -L "$directory" --lib "g3c4_held_$fixture" "$input")

  mkdir -p "$directory/cache/$fixture"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache/$fixture" ESHKOL_CXX_COMPILER="$cxx" \
    ESHKOL_LIB_DIR="$project_root/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$runner" --compile-only --emit-depfile "$directory/$fixture.d" \
    "${arguments[@]}" \
    -o "$directory/deps-$fixture.o" \
    >"$evidence/$mode-$fixture-deps-compile.stdout" \
    2>"$evidence/$mode-$fixture-deps-compile.stderr"
  test ! -s "$evidence/$mode-$fixture-deps-compile.stderr"
  test -s "$directory/deps-$fixture.o"
  test -s "$directory/$fixture.d"

  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$directory/cache/$fixture" \
    ESHKOL_CXX_COMPILER="$directory/cxx-wrap" \
    ESHKOL_LIB_DIR="$project_root/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "$runner" "${arguments[@]}" -o "$directory/held-$fixture" \
    >"$evidence/$mode-$fixture-compile.stdout" \
    2>"$evidence/$mode-$fixture-compile.stderr"
  test ! -s "$evidence/$mode-$fixture-compile.stderr"
  test -s "$directory/held-$fixture"
}
