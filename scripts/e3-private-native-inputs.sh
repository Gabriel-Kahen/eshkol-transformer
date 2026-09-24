#!/usr/bin/env bash
[[ "${package_policy}" == e3-private-aggregate ]] || \
  die "E3 native input policy mismatch"
for e3_provider in n2 n3k a2; do
  "${e1b_clean_toolchain_env[@]}" /usr/bin/bash \
    "${PROJECT_ROOT}/scripts/build-${e3_provider}.sh" \
    "${e1b_tmp}/providers/${e3_provider}" normal
done
e3_reviewed_objects=(
  n2/n2_primitives_provider.o
  n3k/n3k_primitives_provider.o
  a2/a2_attention_provider.o
)
printf '%s\n' "${e3_reviewed_objects[@]}" >"${e1b_tmp}/e3-native-objects.txt"
cmp "${e3_prefix}_native_objects.txt" \
  "${e1b_tmp}/e3-native-objects.txt" || die "E3 native object tuple drifted"
for e3_object in "${e3_reviewed_objects[@]}"; do
  package_native_objects+=("${e1b_tmp}/providers/${e3_object}")
  package_native_depfiles+=("${e1b_tmp}/providers/${e3_object%.o}.d")
done
