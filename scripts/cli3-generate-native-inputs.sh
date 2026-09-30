#!/usr/bin/env bash
# The CLI3 aggregate already owns N2/N3K/A2. Add only accepted G3N/G3S
# provider objects; a second M3/native registry would invalidate the witness.
[[ "${package_policy}" == cli3-tr3-aggregate &&
   ( "${cli3_generate_fixture}" == 1 ||
     "${cli3_production_generate}" == 1 ) ]] ||
  die "CLI3 generate native tuple mismatch"
for provider in g3n g3s; do
  "${e1b_clean_toolchain_env[@]}" /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-${provider}.sh" \
    "${e1b_tmp}/providers/${provider}" normal
done
for input in \
    g3n/g3n_primitives_provider \
    g3s/g3s_sampling_provider; do
  package_native_objects+=("${e1b_tmp}/providers/${input}.o")
  package_native_depfiles+=("${e1b_tmp}/providers/${input}.d")
done
