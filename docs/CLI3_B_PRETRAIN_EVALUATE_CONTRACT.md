# CLI3-B fixed-profile pretrain and evaluate contract

Status: **proposed, dependency-gated**. CLI3-A's six commands, parser, checked
I/O, JSON/diagnostic rules and exit statuses in [CLI3_PROPOSAL.md](CLI3_PROPOSAL.md)
remain authoritative. This slice adds only `pretrain` and `evaluate` after the
public trainer, checkpoint, fresh-process resume, and evaluation candidates are
merged and tested together. It does not add `generate` or a public library API.
The contract is a fixed diagnostic training path, not general pretraining.
The source audit used remote-main `11894d5`, merged trainer state/train via
#149/#150, and draft heads #151 `ceb1d43`, #152 `cec0efb`, #153 `ae45da4`.
Those draft heads must be rechecked at composition; none is treated as merged.

## Grammar and fixed profile

```text
eshkol-transformer pretrain
  --config PATH --tokenizer PATH --train-corpus DIRECTORY
  --checkpoint PATH --max-updates N [--resume PATH] [--force]

eshkol-transformer evaluate
  --config PATH --tokenizer PATH --train-corpus DIRECTORY
  --validation-corpus DIRECTORY --checkpoint PATH
```

Each option is a singleton. CLI3-A's strict long-option grammar, path and
integer spelling, standalone help, bounded diagnostic and no-implicit-input
rules apply. `N` is an integer in 1..4 **per invocation**. Both commands use the
accepted M3T `eshkol-diagnostic-byte-decoder-v1` profile; there is no profile flag or
runtime-selected device, dtype, optimizer, schedule, shuffle, batch shape, or
checkpoint policy. The help text must identify the fixed profile and this
four-update invocation cap. The command-help responses are respectively:

```text
Usage: eshkol-transformer pretrain --config PATH --tokenizer PATH --train-corpus DIRECTORY --checkpoint PATH --max-updates N [--resume PATH] [--force]
Usage: eshkol-transformer evaluate --config PATH --tokenizer PATH --train-corpus DIRECTORY --validation-corpus DIRECTORY --checkpoint PATH
```

Each is its own one-line response plus LF. Top-level help retains CLI3-A's
existing bytes, then appends one blank line and these exact lines:

```text
Commands:
  pretrain    Train the fixed CPU f32 diagnostic profile (1..4 updates)
  evaluate    Evaluate a checkpoint with that fixed profile
```

`generate` remains a reserved usage error.

The caller reads an X1 source config at the CLI3-A 16,384-byte limit, runs
`config-parse`, `config-resolve` with no overrides and `config-validate`, and
requires the authentic M3T profile: context length 2, hidden size 4, one layer,
two query heads, vocabulary 256, CPU f32 and deterministic mode. The run seed
and positive `training.accumulation-steps` come from the resolved X1 value;
the initializer seed must equal `run.seed`. Reject any other resolved model or
precision policy as `unsupported` before constructing a dataset, model, or
output. Load the supplied canonical T1 byte tokenizer, require its fingerprint
and vocabulary to agree with the resolved configuration, and reject BPE or
other tokenizer families for this slice. A checkpoint's tokenizer and config
identities must match these exact operands at restore.

Open each finite D1 corpus through the public D2 ten-key option list with
`:batch-size 1`, `:sequence-length 2`, `:maximum-manifest-bytes 1048576`,
`:maximum-shard-bytes 1048576`, `:maximum-total-tokens 65536`,
`:maximum-batch-bytes 34`, `:shuffle-seed #f`, `:shuffle-window-rows 1`, and
`:packing? #t`. Each directory must pass D1/D2's complete checks. Training
must have a positive usable mask weight; evaluation also rejects zero weight.
The validation dataset is a separately opened receiver and its directory must
differ from the training directory after canonical path resolution. That path
check prevents accidental same-directory use; it does not prove statistical
independence or detect duplicate corpus content.

Construct the installed M3T diagnostic model with
`diagnostic-initializer-state` from X1 `run.seed`, obtain its P1 parameter tree,
deduplicate tied handles by identity, and pass the resulting canonical paths to
the public `optimizer-create`. The fixed O2 preset is the accepted CPU f32
AdamW one-group form from the public trainer evaluation witness: no clipping,
constant schedule, learning-rate word `3dcccccd`, beta words `3f000000` and
`3f000000`, epsilon word `3a83126f`, and weight-decay word `00000000`.
`trainer-create` binds the resolved config, tokenizer, training dataset, model,
and optimizer. No CLI-specific constructor, raw tensor access, or alternate
scalar training loop is permitted. The implementation gate must verify this
exact preset against the merged O2 facade; the cited witness is a composition
example, not evidence that this CLI already runs.

## Pretrain, resume, and artifact lifecycle

Fresh pretrain calls `trainer-stop-policy #f N #f`, then the two-argument
`trainer-train!`. Limits are deltas from invocation entry. The command creates
one complete `trainer-state` after a successful stop and publishes one C2
checkpoint at `--checkpoint` with the ordinary CPU `persistence-policy`
`(16777216, 524288, 8388608, 64, cpu)` and no required features. `--force`
selects C2's `:overwrite? #t`; without it an existing destination is rejected
before training. The output parent directory must exist. No intermediate or
periodic checkpoint is promised. A process loss or failed later update can be
recovered only from the last durable checkpoint; the CLI must not report the
unsaved updates as durable. C2 atomic replacement is the publication commit,
with its documented precommit temporary and postcommit I/O limitations.

For `--resume`, first reconstruct compatible fresh public operands. Discover
the actual K2 capability report, call `checkpoint-load --resume policy report`,
then `trainer-load-state!` on the fresh trainer and release the detached owner
with `trainer-state-release!`. The command must not use `checkpoint-inspect`
as a substitute for capability-gated load. It then calls the same bounded
`trainer-train!` and saves a fresh state. `--resume` may equal `--checkpoint`
only with `--force`; C2 replacement must leave the old complete file in place
until commit. A validation/load failure cannot mutate the existing checkpoint
or advance a live training cursor. A successful restore starts at the exact
next update, preserving the saved O2 schedule, counters, RNG and D2 cursor;
the CLI must not reset them or reapply the initial seed to the restored state.
Every successful or failing path releases each owned detached state and the
trainer lease, closes D2 receivers, and retains the public lifetime limits of
the model and optimizer. No promise of process-lifetime flat retention follows
from this bounded command.

## Evaluate and output

`evaluate` reconstructs the same fixed training operands from its config,
tokenizer and `--train-corpus`, loads `--checkpoint` by the same public K2/C2
capability and `trainer-load-state!` sequence, then opens a distinct validation
dataset and calls `trainer-evaluate! trainer validation`. This API supplies
exactly `loss` and `mask-weight` as f32 plus `tokens` and `batches` as i64,
read through `metrics-ref`. It does not supply perplexity or token accuracy to
this command. The validation cursor and model mode are restored by the
accepted evaluator on success and failure; evaluation does not change the
training cursor, parameters, optimizer or checkpoint. It publishes no file.

On success each command emits one compact JSON object plus LF, after all
operations and cleanup succeed, using the CLI3-A rendering rules. Pretrain
reports `{"artifact":"checkpoint","checkpoint":"...","epochs":E,
"loss_f32_bits":"hhhhhhhh","mask_weight_f32_bits":"hhhhhhhh",
"tokens":T,"updates":U}` with actual checkpoint path and invocation
summary. Evaluate reports `{"artifact":"evaluation","batches":B,
"loss_f32_bits":"hhhhhhhh","mask_weight_f32_bits":"hhhhhhhh",
"tokens":T}`. Each `h` is a lowercase hexadecimal digit of the physical
binary32 word returned by `metrics-ref`, most significant nibble first. A
reviewed CLI-local transport helper may extract these words without becoming a
public library API; its bit identity must be checked against the public TR3/E3
word witnesses. Nonfinite values are errors, never JSON NaN/Infinity tokens.
No success JSON is written before C2 publication or complete evaluation.
Failures retain CLI3-A's exit-category map and single bounded stderr line;
the stdout-write-error exception to empty stdout also remains unchanged.

## Dependency and negative gates

Implementation is admitted only after one same-registry aggregate contains
merged public `trainer-create/release!`, `trainer-stop-policy`, two-argument
`trainer-train!`, `trainer-state/load-state!`, `trainer-state-release!`,
`checkpoint-load/save!`, `capability-discover`, `trainer-evaluate!`, and
`metrics-ref`, with the accepted C2/X1/T1/D1/D2/M3T/P1/O2/E3 producers. In
particular, draft checkpoint PR #151, fresh-resume PR #153 and evaluator PR
#152 must be composed and retested; the current draft heads alone do not
establish that combined executable. CLI3-B must source-compose that aggregate,
not link independently localized CLI3-A and TR3 archives.

The focused end-to-end gate must build twice from fresh caches and compare the
closure, symbol inventory and executable bytes. It must use the delivered CLI
binary to build or inspect accepted tokenizer/corpus inputs, pretrain through a
finite D2 EOS with unequal active-token weights, save C2, evaluate a distinct
corpus, and resume in a fresh process. Compare canonical checkpoints and f32
metric words at every continuation point against the public one-update
reference, including a nonconstant schedule witness in the upstream TR3
gate; CLI3's fixed constant preset proves exact resume for that preset only.
Exercise missing/duplicate/unknown options, malformed numbers, wrong X1
shape/dtype/device/profile, BPE and tokenizer/corpus identity mismatch,
same validation directory, empty/corrupt/oversized D1, zero-weight validation,
corrupt/oversized/unsupported C2, incompatible run seed, missing K2 capability,
existing output without `--force`, failed save before and after replacement,
and stdout/diagnostic faults. Verify status, one bounded diagnostic, no
success JSON on failure, checkpoint byte preservation before commit, published
state after commit, cursor/model-mode restoration, and owned-state release.

The fixed model has no dropout: its training RNG counters remain zero and a
nonzero restored counter is rejected. This contract cannot claim continuation
after RNG consumption. GPU/HIP, f16/bf16, arbitrary architecture or tokenizer,
generic optimizer options, streaming data, stochastic training, periodic
checkpointing, perplexity, accuracy, and `generate` remain unsupported here.
Full CLI3 acceptance waits for their separately accepted contracts and tests.
