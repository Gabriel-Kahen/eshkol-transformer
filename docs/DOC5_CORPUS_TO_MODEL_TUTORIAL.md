# From a byte corpus to a fixed-profile model

This guide uses the installed CLI3 commands on the pinned CPU lane. It prepares
a DATA4 corpus, trains one update, inspects the C2 checkpoint, evaluates it,
and generates one byte. The [API reference](DOC5_API_REFERENCE.md) identifies
the public and bounded source entry points; the [DATA4 installed contract](DATA4_INSTALLED_REPORT.md)
describes report and publication limits.

## Pinned environment

Use the `eshkol-r0-debug:20260928` image with the Eshkol cache configured at
`/tc/src` and `/tc/build`. The lock in `toolchain/eshkol.lock` pins Ubuntu
22.04 x86-64, LLVM/Clang 21.1.8, and Eshkol
`fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`. From a transformer checkout:

```bash
repo=$(pwd -P)
repo_git=$(git rev-parse --path-format=absolute --git-common-dir)
toolchain_cache=/absolute/path/to/eshkol-fe9dfd52-ubuntu22
docker run --rm --network none \
  -v "$repo:$repo:ro" -v "$repo_git:$repo_git:ro" -w "$repo" \
  -v "$toolchain_cache:/tc:ro" \
  -e ESHKOL_SOURCE_DIR=/tc/src -e ESHKOL_BUILD_DIR=/tc/build \
  -e CC=clang-21 -e CXX=clang++-21 \
  eshkol-r0-debug:20260928 bash -ec '
    git config --global --add safe.directory "*"
    make test-doc5
  '
```

The image and `/tc` cache are prebuilt pinned inputs; the runner builds CLI3
from a disposable copy of the checkout. On a supported native Ubuntu 22.04
environment, follow [toolchain setup](../toolchain/README.md), run
`make toolchain`, then `make test-doc5` from a fresh checkout.

The runner checks every API row, tutorial repository path, and referenced Make
target. It copies tracked sources into a disposable directory, builds CLI3,
runs the existing cross-command gate and installed DATA4 command tests, then
executes the integrated recipe below. A source-only check is available as
`make test-doc5-source`. If you already have the sealed CI binary artifact,
follow the [downloaded artifact guide](CLI3_DOWNLOADED_ARTIFACT_RUNNER.md) to run
`make test-cli3-downloaded-artifact` on the same Ubuntu lane without rebuilding. The literal recipe and 18 installed D1/DATA4 tests passed in pinned Ubuntu using the verified [PR #259 CLI3 artifact](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45883398570).

## Prepare, train, evaluate, and generate

To build the CLI3 executable, use a disposable checkout with the same pinned
toolchain. The [cross-command gate](../scripts/test-cli3-cross-command-chain.sh)
uses the same config and bytes, and checks the output and rejection contracts.

```bash
cli_build=$(mktemp -d)
TMPDIR="$cli_build" CLI3_COMPILER_TIMEOUT_SECONDS=1200 \
  bash scripts/build-cli3.sh "$cli_build/cli3"
cli="$cli_build/cli3/eshkol-transformer"
```

With `cli` set to the absolute path of the built or verified installed
executable, run the recipe:

```bash
scratch=$(mktemp -d)
cat >"$scratch/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
printf '\003\305\051\007\013\023' >"$scratch/train.txt"
printf '\024\025\026\027\030\031' >"$scratch/heldout.txt"
"$cli" tokenizer byte --config "$scratch/config.json" --output "$scratch/byte.t1"
mkdir "$scratch/train" "$scratch/heldout"
"$cli" corpus prepare --tokenizer "$scratch/byte.t1" \
  --source "train=$scratch/train.txt" --minimum-bytes 6 --maximum-bytes 6 \
  --output-directory "$scratch/train" --shard-token-limit 4
"$cli" corpus report-validate --input-directory "$scratch/train"
"$cli" corpus validate "$scratch/train" --max-total-tokens 6
"$cli" corpus build --tokenizer "$scratch/byte.t1" \
  --document "$scratch/heldout.txt" --output-directory "$scratch/heldout" \
  --shard-token-limit 4
"$cli" pretrain --config "$scratch/config.json" --tokenizer "$scratch/byte.t1" \
  --train-corpus "$scratch/train" --checkpoint "$scratch/first.c2" --max-updates 1
"$cli" checkpoint inspect --input "$scratch/first.c2"
"$cli" evaluate --config "$scratch/config.json" --tokenizer "$scratch/byte.t1" \
  --train-corpus "$scratch/train" --validation-corpus "$scratch/heldout" \
  --checkpoint "$scratch/first.c2"
"$cli" generate --config "$scratch/config.json" --tokenizer "$scratch/byte.t1" \
  --train-corpus "$scratch/train" --checkpoint "$scratch/first.c2" \
  --prompt-hex 41 --sampling greedy --seed 1729
```

The prepared training corpus has six byte tokens in two D1 shards. The v2.0
`data4-report.tsv` records source selection, the SHA-256 of every byte in
`manifest.etm`, and a checksum over the report prefix. `report-validate`
checks the report against the validated D1 summary and exact manifest bytes;
even a report from another corpus with the same summary is rejected. Legacy
v1.0 reports are rejected with `version-mismatch`. DATA4 publishes the D1
corpus and report as separate units, so an interrupted publication can leave
a corpus without its sidecar. Preserve the original inputs and options to
regenerate a v2.0 report in a fresh directory if needed.

`corpus validate` checks the D1 manifest and shards with an explicit token
limit. Its plain-text summary begins `shards=2 tokens=6 vocab=256 limit=4`.
The `limit` field is the stored shard token limit, and `--max-total-tokens 6`
bounds the total tokens accepted by the command.

The C2 checkpoint holds fixed-profile CPU-f32 training state; it is distinct
from the C1 logical-parameter format used by IO5. This run establishes one
update, evaluation and one-byte generation for the context-two byte-tokenizer
profile. It does not establish a general model trainer, longer generation,
GPU execution, or CPU-v1 release admission. See the [roadmap](ROADMAP.md) for
their separate gates. Remove the scratch directory and any temporary build
directory when finished.
