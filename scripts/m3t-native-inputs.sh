#!/usr/bin/env bash
# E1B-only fixed native inputs: build with the provider-owned ordinary recipes.
[[ "${package_policy}" == m3t-diagnostic-aggregate ]] || die "M3T native input policy mismatch"
for m3t_provider in n2 n3k a2; do
  "${e1b_clean_toolchain_env[@]}" /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-${m3t_provider}.sh" \
    "${e1b_tmp}/providers/${m3t_provider}" normal
done
m3t_reviewed_objects=(
  n2/n2_primitives_provider.o
  n3k/n3k_primitives_provider.o
  a2/a2_attention_provider.o
)
printf '%s\n' "${m3t_reviewed_objects[@]}" >"${e1b_tmp}/m3t-native-objects.txt"
cmp "${PROJECT_ROOT}/native/m3t_package_native_objects.txt" \
  "${e1b_tmp}/m3t-native-objects.txt" || die "M3T native object tuple drifted"
for m3t_object in "${m3t_reviewed_objects[@]}"; do
  package_native_objects+=("${e1b_tmp}/providers/${m3t_object}")
  package_native_depfiles+=("${e1b_tmp}/providers/${m3t_object%.o}.d")
done
