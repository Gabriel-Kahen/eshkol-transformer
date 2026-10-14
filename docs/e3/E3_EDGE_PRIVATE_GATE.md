# E3 private empty, singleton, EOS and one-token-tail gate

This is a **source-staged, pending compiled-private** §9 N5 gate. It extends
the existing noninstalled retention aggregate only; no public API, production
kernel or expected CE/logit carrier changes. `tests/e3_horizon/edge_generate.py`
creates deterministic checksummed D1 resources and independent packed D2
reference rows at the fixed N1/T2/V256/D4 profile:

| Case | D1 tokens | D2 rows at entry | Expected evaluator path |
|---|---:|---:|---|
| empty | 0 | 0 | Stage-12 `invalid-state`, four carriers unchanged, counters still unpublished |
| singleton | 1 | 0 | Stage-12 `invalid-state`, four carriers unchanged, counters still unpublished |
| entry-eos | 3 | 1, consumed before call | Stage-12 `invalid-state`, four carriers unchanged, counters still unpublished |
| one-token-tail | 4 | `11`, then `10` | Success, three active tokens, two batches, weight bits `1077936128`, accuracy bits `1065353216` |

The final valid row is `input (31,0)`, `target (14,0)`, mask `(true,false)`.
The gate derives these rows and exact entry cursor bytes from the D2 reference
before runtime. The compact selection record
`tests/e3_horizon/edge_accuracy_v1.json` is drawn from the independently
initialized all-correct M3 proof: the two exact row inputs have winners
`(14,31)` and `(14,56)`, so all three active targets are correct and the
accuracy is exactly one. It pins parameter digest `ccf18d65...` and SHA-256
of the sealed reference `d46f415d...` and selection `7dc9dfc4...` records
from `e3-section9-first-eval-oneuse-f7eb-20261013-n/run/container-output/gate/correctness/`.
No exact CE/loss/perplexity word is claimed by this bounded gate; it only
requires positive finite loss and perplexity from the real seed-1729 M3 model.

`tests/e3_private/edge_runtime.esk` invokes the genuine private
`e3-evaluate-into!` with the M3 model, raw T1 tokenizer, D2 dataset, E3 frame
and four I2 destinations. For zero-row cases, the native finalizer rejects
because both counts are zero; the driver asserts the first mapped E1 stage-12
error, unchanged carrier words, and continued nonpublication of both counters.
The public counter accessor requires `published_valid`, so it cannot snapshot
counters on a fresh frame. Every case compares the exact D2 cursor,
17-bit mode image, all parameter and gradient words/metadata, initializer RNG,
and live/retired native ownership controls to its pre-call image. The successful
tail asserts count, weight and the independently selected accuracy from
authentic masks and logits, without carrying fabricated logits into runtime.

`scripts/test-e3-edge-private.sh` is the proposed supported compiled gate. It
regenerates all four resources twice and compares bytes, builds canonical and
fresh A/B plus sanitized noninstalled packages, checks the existing closed
symbol/source inventories, compiles two optimized AOT binaries and a sanitizer
binary, and runs normal, poisoned and ASan/UBSan/LSan profiles. The observed
cursor lines and fourth I2 accuracy word must equal the independent D2 and
M3 selection oracles. A source-only causal control changes only the reported
fourth I2 word to zero and requires the observer to reject it. It retains all
build and runtime evidence in a fresh directory. No compiled execution has
been run from this source-staging branch.

Independent whole-tree review of this current-main source composition, then
the supported compiled gate, are required before marking these four N5 cases
proved. PR #206 CI passed before its merge and did not run this gate. Exact
8,192/8,193 and full resource/ownership horizons remain separately gated.

The reviewed `129460f4` compiled B attempt built the runtimes but its first
`normal-a` process exited 1 with only `Unhandled exception: user exception`;
it reached no oracle or acceptance result. This diagnostic successor marks and
flushes the case and phase to stderr before fixture creation, D2 open, seed,
each snapshot, evaluation, every postcondition and close. An outer edge-only
guard prints caught exception category, operation and message (or the raw
non-transformer value), then rethrows. The evaluator's own guard and exact
stage-12 zero-row assertion remain in place. The runner checks complete
four-case phase transcripts and runs deliberate pre-evaluation and post-
evaluation controls. The pre control passes a guaranteed missing corpus
directory into D2 open and requires its structured `io` exception at
`d2-open`. The post control first checks the expected stage-12 zero-row
outcome, then fails a labeled assertion on a readable output carrier through
its native invariant abort. Its flushed label and `postcondition-control` phase must
survive the failing process. Neither control can emit an edge pass.
These are source-staged diagnostics; independent review and a fresh compiled
result are still required before edge admission. The shared test driver also
regenerates the first-evaluation mutant and updates its source pins. The
candidate package inventories include the compiler's port, write and flush
imports, inferred from its codegen; source checks do not prove those exact
imports or package closure.

The `a007ab91` compiled attempt built all three runtimes and passed the
missing-path pre control, but its post control stopped at `outputs-snapshot`:
`e3-counter-ref` correctly rejected the fresh unpublished frame. The test-only
successor snapshots the four I2 words, checks both selectors' exact
`invalid-state`/phase-2 rejection before evaluation and after zero-row failure,
and keeps successful-tail count assertions on the public accessor. The post
control now fails on an actual I2 word after the evaluated outcome check. This
source repair still needs independent review and a fresh compiled edge gate.
