#!/usr/bin/env bash
# Sourced only by the exact zero-tuple discovery branch of build-e1b-consumer.sh.
e3_zero_discovery_capture() {
  local result=$1 status=$2 file source destination
  local dir="${e3_zero_discovery_dir}"
  [[ -d "${dir}" && ! -L "${dir}" && -O "${dir}" ]] || return 1
  printf 'result\t%s\nexit_status\t%s\nmode\t%s\nphase\t%s\n' \
    "${result}" "${status}" "${e3_private_test_sanitize}" \
    "${e3_zero_discovery_phase:-unknown}" >"${dir}/builder-status.tsv"
  for file in private.d private.ll private.asan.ll private.asan.json \
      private.asan-imports.txt private.asan-nm.stderr private.o bridge.o \
      package-bridge.o combined.raw.o combined.o combined.map \
      source-closure.txt native-source-closure.txt e3-native-objects.txt \
      package-exports.txt expected-undefined.txt expected-global-defined.txt \
      global-defined.txt undefined.txt unexpected-undefined.txt \
      noninstrumentation-undefined.txt sanitizer-undefined.txt \
      readelf-symbols.txt localize-symbols.txt d2-generation.json \
      zero-p1-generation.txt zero-p1-check.txt zero-p1-postcheck.txt; do
    source="${e1b_tmp}/${file}"
    [[ ! -f "${source}" || -L "${source}" ]] || cp -- "${source}" "${dir}/${file}"
  done
  if [[ -d "${output_object}.evidence.tmp.$$" &&
        ! -L "${output_object}.evidence.tmp.$$" ]]; then
    cp -a -- "${output_object}.evidence.tmp.$$/." "${dir}/"
  fi
  mkdir -p -- "${dir}/native-depfiles" "${dir}/generated" "${dir}/expected"
  for source in "${e1b_tmp}"/bridge.d "${e1b_tmp}"/package-bridge.d \
      "${e1b_tmp}"/package-native-*.d; do
    [[ ! -f "${source}" || -L "${source}" ]] || \
      cp -- "${source}" "${dir}/native-depfiles/$(basename -- "${source}")"
  done
  for file in e3_d2_dataset.esk e3_d2_source_provenance.json \
      transformer/module.esk; do
    source="${e3_generated_source_root:-/no-generated-source}/${file}"
    [[ ! -f "${source}" || -L "${source}" ]] || {
      destination="${dir}/generated/${file}"
      mkdir -p -- "$(dirname -- "${destination}")"
      cp -- "${source}" "${destination}"
    }
  done
  for source in "${package_source_closure:-/no-source-closure}" \
      "${package_native_source_closure:-/no-native-closure}" \
      "${undefined_symbols:-/no-undefined}" \
      "${e3_prefix:-/no-prefix}_sanitized_noninstrumentation_undefined_symbols.txt" \
      "${e3_prefix:-/no-prefix}_sanitized_instrumentation_undefined_symbols.txt" \
      "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"; do
    [[ ! -f "${source}" || -L "${source}" ]] || \
      cp -- "${source}" "${dir}/expected/$(basename -- "${source}")"
  done
  (cd -- "${dir}" && find . -type f ! -name SHA256SUMS -print0 | \
    LC_ALL=C sort -z | xargs -0 -r sha256sum) >"${dir}/SHA256SUMS"
}
