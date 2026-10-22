# Fixed-profile language-model walkthrough

Status: command recipe derived from the candidate's compiled CLI acceptance script.
The exact candidate's hosted cross-command test passed. A fresh retained local
cross-command gate also passed at `574b98e42ee0d81491458fc5439c7efb64082014`
in 678.61 seconds. Its supported executable and actual artifacts are available;
this standalone recipe has not been executed separately. Full Wave 3 acceptance is tracked in
[WAVE3_COMPLETION.md](WAVE3_COMPLETION.md).

This exercises the existing CPU-f32 diagnostic model: context 2, hidden size 4,
one layer, two query heads and a 256-byte vocabulary. `pretrain` currently admits
1–4 updates per invocation; `generate` emits one raw byte. These limits are part
of the current command contract. This recipe demonstrates training and checkpoint
plumbing, not useful language quality or full generation continuation.

## Build on the supported lane

Use Ubuntu 22.04 x86-64 and Clang/LLVM 21.1.8 with the exact
[toolchain lock](../toolchain/eshkol.lock). Follow [toolchain setup](../toolchain/README.md).
From the checkout, with a verified pinned dependency build:

```bash
bash scripts/build-cli3.sh "$PWD/build/cli3"
export WAVE3_BIN="$PWD/build/cli3/eshkol-transformer"
"$WAVE3_BIN" --version
"$WAVE3_BIN" --help
```

The finish team serializes this build with its shared `heavy.lock` and at most
three compiler jobs. Build output contains the executable, aggregate, copied
facades and source/symbol evidence. Use the executable inside the supported
runtime environment in which it was built.

## Create artifacts, train, resume, evaluate and generate

Run the following in a new empty working directory. Python is unnecessary for the
training path; shell `printf` creates the exact raw-byte development fixtures.

```bash
set -euo pipefail
: "${WAVE3_BIN:?set WAVE3_BIN to the absolute supported executable path}"
cat > config.json <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
printf '\003\305\051\007\013\023' > train.txt
printf '\024\025\026\027\030\031' > heldout.txt
"$WAVE3_BIN" tokenizer byte --config config.json --output byte.t1
"$WAVE3_BIN" tokenizer inspect --input byte.t1
mkdir train heldout
"$WAVE3_BIN" corpus build --tokenizer byte.t1 --document train.txt \
  --output-directory train --shard-token-limit 4
"$WAVE3_BIN" corpus build --tokenizer byte.t1 --document heldout.txt \
  --output-directory heldout --shard-token-limit 4
"$WAVE3_BIN" corpus validate train
"$WAVE3_BIN" corpus validate heldout
"$WAVE3_BIN" pretrain --config config.json --tokenizer byte.t1 \
  --train-corpus train --checkpoint first.c2 --max-updates 1
"$WAVE3_BIN" pretrain --config config.json --tokenizer byte.t1 \
  --train-corpus train --checkpoint resumed.c2 --resume first.c2 --max-updates 1
"$WAVE3_BIN" pretrain --config config.json --tokenizer byte.t1 \
  --train-corpus train --checkpoint uninterrupted.c2 --max-updates 2
cmp resumed.c2 uninterrupted.c2
"$WAVE3_BIN" checkpoint inspect --input resumed.c2
for image in resumed uninterrupted; do
  "$WAVE3_BIN" evaluate --config config.json --tokenizer byte.t1 \
    --train-corpus train --validation-corpus heldout --checkpoint "$image.c2" \
    > "evaluate-$image.json"
  "$WAVE3_BIN" generate --config config.json --tokenizer byte.t1 \
    --train-corpus train --checkpoint "$image.c2" --prompt-hex 41 \
    --sampling greedy --seed 1729 > "generate-$image.json"
done
cmp evaluate-resumed.json evaluate-uninterrupted.json
cmp generate-resumed.json generate-uninterrupted.json
cat evaluate-resumed.json generate-resumed.json
```

A successful recipe requires identical resumed/uninterrupted checkpoint bytes and
identical evaluation/generation JSON. The acceptance script also checks input
checkpoint immutability, exact output schemas and rejected wrong tokenizer/config
or corrupted checkpoint images. Run that complete supported gate with a fresh
output directory:

```bash
bash scripts/test-cli3-cross-command-chain.sh /absolute/fresh/evidence-directory
```

The full gate builds its own binary; reuse an existing accepted result when only
inspecting the same source. Its `cross-command-scratch-path.txt` identifies the
retained binary/config/tokenizer/corpus/checkpoint fixtures. The exact output
JSON and sealed gate receipt, rather than an expected example invented here,
are the result to inspect.

## Retained bounded execution

Receipt: `/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/CLI3-LOCAL-RESULT-REVIEW.md`.
The binary is retained at
`/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/cli3-local/cross-command/.cli3-cross-command.budija/eshkol-transformer`,
SHA-256 `dce2be716ab56707e54b1106c138a25c925b365d73ec31f58060f4adf33056d1`.
The same directory contains config/tokenizer/corpora/checkpoints; JSON is in
`cli3-local/cross-command/` below the finish receipt root. Independently verified
resumed/uninterrupted checkpoint bytes and evaluation/generation JSON agree;
evaluation reports three batches/five tokens and greedy generation emits byte
`c5`. The 125-file gate seal verifies. This is the legacy context-two profile,
not the new context-four CPU v1 workflow or merged integration acceptance.
