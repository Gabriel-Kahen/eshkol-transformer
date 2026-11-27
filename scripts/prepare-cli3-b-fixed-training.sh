#!/usr/bin/env bash
set -euo pipefail

binary=$1
scratch=$2
cat >"${scratch}/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
# Three D2 rows have mask weights 2, 2, 1; update two crosses finite EOS.
printf '\003\305\051\007\013\023' >"${scratch}/train.txt"
"${binary}" tokenizer byte --config "${scratch}/config.json" \
  --output "${scratch}/byte.t1" >"${scratch}/tokenizer.json"
mkdir "${scratch}/train"
"${binary}" corpus build --tokenizer "${scratch}/byte.t1" \
  --document "${scratch}/train.txt" --output-directory "${scratch}/train" \
  --shard-token-limit 4 >"${scratch}/train.json"
