# From corpus to model on the pinned CPU lane

Start with an empty scratch directory and run the delivered DATA4, MOD5, IO5,
MP4, and PA4-Wider components. The automated walkthrough uses separate
fixtures for corpus preparation, model training, and checkpoint exchange.
The [fixed-profile CLI3 recipe](#train-directly-from-a-d1-corpus-with-cli3)
shows the installed path that trains from D1 shards.

## 1. Check the environment and run the walkthrough

Use the `eshkol-r0-debug:20260928` image and a prepared toolchain cache with
`src/` and `build/` under its root. The build must have been configured at
`/tc/src` and `/tc/build` so the CMake and provenance paths match. The lock
pins Eshkol `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`, Ubuntu 22.04 x86-64,
and LLVM/Clang 21.1.8. The runner checks the clean Eshkol checkout, compiler
version and SHA-256, and build provenance before starting.

From the transformer checkout, set `toolchain_cache` to your prepared cache.
Mount the checkout at its original absolute path so a linked Git worktree can
still find its metadata. A read-only repository mount is sufficient.

```bash
set -o pipefail
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
    bash scripts/test-doc5-walkthrough.sh
  ' 2>&1 | tee /tmp/eshkol-doc5-walkthrough.log
```

The runner copies the tracked sources to a fresh directory under `/tmp`,
checks the [API reference](DOC5_API_REFERENCE.md) against that copy, and
removes the workspace and generated artifacts on exit. It prints
`DOC5 fresh-environment walkthrough: PASS` only after every stage succeeds.
To inspect artifacts yourself, run the corresponding stage below in a
disposable checkout in the same environment.

## 2. Prepare the corpus and verify the D1 report

```bash
bash scripts/test-data4-source.sh
DATA4_EVIDENCE_DIR=$(mktemp -d)
export DATA4_EVIDENCE_DIR
bash scripts/test-data4-native.sh
cat "$DATA4_EVIDENCE_DIR/report-a.tsv"
sha256sum "$DATA4_EVIDENCE_DIR/corpus-a/manifest.etm"
```

The native fixture starts with three explicit documents, `AB`, a second
`AB`, and `CD`. DATA4 filters by raw bytes and retains the first copy of
each passing document. The T2 tokenizer applies a beginning-of-sequence
token to each retained document. D1 writes the six resulting IDs to two
shards of three tokens. Compare the SHA-256 of `corpus-a/manifest.etm` to the
`manifest-sha256` field in the canonical `version<TAB>2<TAB>0` report.

The source checks reject a report swapped between corpora with equal
aggregate summaries but different manifests. The native gate checks two
byte-identical writes, rejects writing over the existing corpus, validates the
report and token stream, and deliberately corrupts a shard in `corpus-b` to
check D1 rejection. Use `corpus-a` for inspection.

With the installed CLI3 binary, `corpus prepare` takes repeatable
`--source SOURCE_ID=PATH` arguments and writes `data4-report.tsv` next to
`manifest.etm`. `corpus report-validate --input-directory DIRECTORY`
validates the shards and report together. For a v1.0 report, regenerate it
from the original inputs in a fresh directory. See the [installed report
contract](DATA4_INSTALLED_REPORT.md) for limits and publication behavior.

## 3. Train the MOD5 decoder block

```bash
make test-mod5-aot
```

The pinned compiler builds the Eshkol training program twice with fresh
caches and links it against the K1 and MOD5 archives. The program fits a
single pre-norm decoder block on deterministic synthetic inputs and targets.
It uses `N=1, T=8, D=16, F=32, Hq=4, Dh=4` with one KV head for MQA and two
for GQA. Eshkol performs the projections, residuals, loss, gradient
composition and SGD updates in CPU f64. The bridge marshals f32 arrays for
MOD5 RMSNorm, SwiGLU and attention forward/backward calls.

Expect identical ELF and stdout bytes across the two builds, 30 checked
finite differences per variant, and strictly decreasing loss across 16
updates. The pinned reference trace is approximately `0.008396 → 0.004525`
for MQA and `0.008880 → 0.004290` for GQA. The program reports a final
parameter checksum. Its inputs and outputs are internal to this fixture.
See the [MOD5 contract](MOD5_VARIANTS.md) for the 28 provider rows and bridge
limits.

## 4. Create a C1 checkpoint and round-trip it through IO5

```bash
make test-io5
make test-io5-live-c1
```

The format tests exercise the three mapping profiles, including the
bias-free `eshkol-mod5-v1` profile. Independent Python format readers and
the native GGML check require separately provisioned dependencies. The live
gate builds the Eshkol C1 bridge twice. The bridge creates an M3 logical parameter state
with I2 f32 storage, saves it to C1, and checks the saved state using the
compiled reader. The IO5 tools export and reimport the checkpoint through
five routes.

- Standalone safetensors and GGUF using IO5 synthetic names.
- Mapped safetensors with Hugging Face names and mapped GGUF with GGUF names,
  both using `eshkol-m3-v1`.
- Sharded Hugging Face safetensors using `eshkol-m3-v1` and
  `model.safetensors.index.json`.

All five routes must reproduce the original C1 bytes, then pass a live
compiled reload that checks tensor payloads, metadata and alias topology.
The harness also rejects corrupted exchange payloads, shards, and C1
fixtures. The bridge starts from its own generated state. The exchange
checks tensor data and metadata. See the
[live C1 guide](IO5_LIVE_C1_INTEROP.md) and [mapping contract](IO5_MODEL_MAPPING.md)
for import rules.

From a disposable checkout with a compatible M3 C1 checkpoint named
`original.etcp` in its root, try the sharded route.

```bash
mkdir bundle
python3 tools/io5/model_mapping.py c1-to-sharded-safetensors original.etcp \
  bundle/model.safetensors.index.json --profile eshkol-m3-v1 \
  --scheme huggingface --max-shard-bytes 4200
python3 tools/io5/model_mapping.py sharded-safetensors-to-c1 \
  bundle/model.safetensors.index.json restored.etcp
cmp original.etcp restored.etcp
```

## 5. Check storage precision and wider attention

```bash
MP4_ASAN_DETECT_LEAKS=1 make test-mp4
make test-pa4-wider
```

MP4 checks f32 conversion to and from f16 and bf16 storage, f32 gradient
unscaling, and dynamic loss scaling. The momentum fixture forces overflow,
skips that optimizer update, backs off from scale 64 to 32, and resumes on
clean gradients. Its reference and sanitizer checks are part of the gate.

PA4-Wider compares separately linked reference and online-softmax CPU-f32
providers for causal MHA, GQA and MQA. It covers query and key lengths up to
64 and head dimension up to 16, comparing forward and backward results and
checking finite differences and invalid inputs. The target also runs a
timing harness. Treat its output as a measurement of that run and machine.

## Train directly from a D1 corpus with CLI3

From a disposable checkout with the same toolchain, build CLI3 and train
the fixed CPU-f32 byte-tokenizer profile on a six-byte file. This recipe
uses the same config and bytes as the
[cross-command gate](../scripts/test-cli3-cross-command-chain.sh).

```bash
(
  set -e
  scratch=$(mktemp -d)
  trap 'rm -rf -- "$scratch"' EXIT
  TMPDIR="$scratch" CLI3_COMPILER_TIMEOUT_SECONDS=1200 \
    bash scripts/build-cli3.sh "$scratch/cli3"
  cli="$scratch/cli3/eshkol-transformer"

  cat >"$scratch/config.json" <<'JSON'
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":2,"model.hidden-size":4,"model.layer-count":1,"model.query-head-count":2,"model.vocabulary-size":256,"run.seed":1729,"training.accumulation-steps":2}
JSON
  printf '\003\305\051\007\013\023' >"$scratch/train.txt"
  mkdir "$scratch/train"
  "$cli" tokenizer byte --config "$scratch/config.json" --output "$scratch/byte.t1"
  "$cli" corpus build --tokenizer "$scratch/byte.t1" --document "$scratch/train.txt" \
    --output-directory "$scratch/train" --shard-token-limit 4
  "$cli" pretrain --config "$scratch/config.json" --tokenizer "$scratch/byte.t1" \
    --train-corpus "$scratch/train" --checkpoint "$scratch/first.c2" --max-updates 1
  "$cli" checkpoint inspect --input "$scratch/first.c2"
)
```

The result is a C2 training-state checkpoint from the fixed-profile CLI3
trainer. The IO5 commands in this guide take C1 logical parameter checkpoints.

## What this run establishes

The automated gate exercises the pinned CPU component paths, including real
D1 sharding, Eshkol AOT training, and compiled C1 reload. Training MOD5 on a
D1/D2 corpus, exporting its live weights, integrating MP4 into the trainer,
and selecting PA4 by default require separate integration. Full public TR3,
CPU-v1 release admission and GPU execution have separate gates tracked in
the [roadmap](ROADMAP.md).

## Next Steps: Full 8-Stage Eshkol AOT LLM Lifecycle & Wave 4/5 Suites

Beyond the baseline DOC5 walkthrough above, the repository integrates an
executable 8-stage compiled Eshkol AOT lifecycle and an 11-engine Wave 5
native benchmark suite on the pinned CPU lane:

```bash
make test-e2e-lifecycle
make test-b0-wave5
bash scripts/test-doc5-wave45-walkthrough.sh
```

- **`make test-e2e-lifecycle`** builds and runs the statically linked compiled
  Eshkol AOT binary across five OS process boundaries, chaining DATA4 corpus
  hygiene and near-deduplication, T2 BPE training (`ETBPE100`) and D1 `.ets`
  sharding, MOD5 RoPE + AdamW pretraining with bitwise fresh-process
  `ETM5TRN1` resume, LoRA fine-tuning (`ETLR1000`) and reversible merge, DPO
  preference alignment, contamination-gated EVAL5 scoring, lossless
  speculative decoding, and native IO5 `F16`/`BF16`/`Q8_0` quantization. See
  [E2E_LLM_LIFECYCLE.md](E2E_LLM_LIFECYCLE.md).
- **`make test-b0-wave5`** builds all 11 Wave 5 native static archives plus
  normal and ASan/UBSan/LSan benchmark runners (`wave5-bench-v1`) and checks
  two-run and sanitizer numerical checksum invariance. See
  [B0_WAVE5_BENCHMARKS.md](B0_WAVE5_BENCHMARKS.md).
- **`bash scripts/test-doc5-wave45-walkthrough.sh`** programmatically verifies
  every header path, doc link, script path, and `make` target referenced across
  [DOC5_API_REFERENCE.md](DOC5_API_REFERENCE.md), this tutorial, and the
  [Wave 4/5 operator guide](DOC5_WAVE45_GUIDE.md), then executes
  `make test-e2e-lifecycle`, `make test-b0-wave5`, `make test-python-isolation`,
  and `make test-ci-topology` in a fresh disposable checkout.

For the full category-indexed directory of Wave 4/5 subsystems (Data &
Tokenizer, Architecture, Optimizers, Alignment, Quantization, Distributed,
Serving, Benchmarks, and Lifecycle), see
[docs/DOC5_WAVE45_GUIDE.md](DOC5_WAVE45_GUIDE.md).
