#!/usr/bin/env bash
set -euo pipefail

[[ $# == 1 && -f $1 ]] || {
  printf 'usage: %s READELF_SYMBOLS\n' "$0" >&2
  exit 2
}

# These probes belong only to the noninstalled retention/mode-graph tuples.
# Exact symbol-field matching avoids confusing a similarly named string with
# an actual definition or unresolved import in the canonical object.
for observer in \
    et_e3_test_retention_probe_v1 \
    et_e3_test_model_word_count_v1 \
    et_e3_test_model_word_v1 \
    et_e3_test_gradient_metadata_v1 \
    et_e3_test_seed_gradient_v1 \
    et_e3_test_first_eval_corrupt_v1 \
    et_e3_test_graph_event_v1 \
    et_d2_exact_read_test_fail_stage_v1 \
    et_f32_test_live_counts_snapshot_v1 \
    et_f32_test_retired_counts_snapshot_v1 \
    et_f32_test_borrow_event_counts_snapshot_v1; do
  if awk -v name="$observer" '$NF == name { found = 1 } END { exit !found }' "$1"; then
    printf 'E3 observer leaked into canonical private package: %s\n' "$observer" >&2
    exit 1
  fi
done
