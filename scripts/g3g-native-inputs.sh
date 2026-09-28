#!/usr/bin/env bash
# E1B-only ordinary provider recipes for the fixed G3-G aggregate.
[[ "${package_policy}" == g3g-public-aggregate ]] || die "G3-G native input policy mismatch"
for g3g_provider in n2 n3k a2 g3n g3s; do
  "${e1b_clean_toolchain_env[@]}" /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-${g3g_provider}.sh" \
    "${e1b_tmp}/providers/${g3g_provider}" normal
done
g3g_reviewed_objects=(
  n2/n2_primitives_provider.o
  n3k/n3k_primitives_provider.o
  a2/a2_attention_provider.o
  g3n/g3n_primitives_provider.o
  g3s/g3s_sampling_provider.o
)
printf '%s\n' "${g3g_reviewed_objects[@]}" >"${e1b_tmp}/m3-native-objects.txt"
cmp "${PROJECT_ROOT}/native/g3g_package_native_objects.txt" \
  "${e1b_tmp}/m3-native-objects.txt" || die "G3-G native object tuple drifted"
for g3g_object in "${g3g_reviewed_objects[@]}"; do
  package_native_objects+=("${e1b_tmp}/providers/${g3g_object}")
  package_native_depfiles+=("${e1b_tmp}/providers/${g3g_object%.o}.d")
done
