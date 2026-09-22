# TR3 fixed-profile trainer and exact-resume proposal

Status: **decision-ready proposal only**. This document adds no runtime API,
capability evidence, checkpoint format, or completion claim. All interfaces marked
**PROPOSED** are dependency gates; downstream code must not call them until their
contracts, implementations, independent reviews, and supported evidence are merged.

Baseline is `origin/main` at
`19f404cf21632944e1f2d5d4959e1240f9a79f5f` (PR #105 merge).
Accepted A0, M3, M3T, L2, L3S, O2, D2, C2, K2, and X1 behavior below is taken from
that baseline. PR #105 merged the shared M3 guard/fixed14 implementation source,
but its own record still labels runtime acceptance open. Local orchestration head
`b052385e742626fcfc2557b3b5a9f176f6234b29` additionally contains E3 metric and
E3-D2 prerequisite candidates. Those two candidates are **unmerged implementation
evidence**, not accepted runtime dependencies. E3's exact private design is accepted
on main, but its consumer implementation and runtime acceptance remain open.

## 1. Decision requested

TR3 should implement the existing A0 trainer surface for one bounded profile:

```scheme
(trainer-create resolved tokenizer dataset model optimizer)
(trainer-step! trainer)
(trainer-stop-policy max-tokens max-updates max-epochs)
(trainer-train! trainer stop-policy)
(trainer-evaluate! trainer dataset)
(trainer-state trainer)
(trainer-load-state! trainer state)
(trainer-state-release! state) ; already implemented by C2
(metrics-ref metrics key)
```

The admitted profile is the accepted M3 profile: CPU f32, `N=1`, `T=2`, `V=256`,
`D=4`, `Hq=Hkv=2`, `Dh=2`, `L=1`, `F=8`, learned positions, three affine
LayerNorms, exact-erf GELU, tied token/head weight, no dropout, and no cache. The
tokenizer is canonical T1 raw byte with vocabulary 256. The training D2 dataset is
finite, batch size 1, sequence length 2, and uses any accepted deterministic D2
shuffle/packing configuration bound into its cursor. This is a complete fixed-profile
trainer, not arbitrary-shape pretraining.

Implementation is blocked until the accepted/dispatched prerequisites are implemented,
reviewed, and merged and the proposed TR3 seams are accepted:

1. **SHARED-R2:** the issue #114 accepted/dispatched genuine rank-two
   `tensor.f32/storage.copy` provider and matching K2 successor audit.
2. **TR3-A:** one source-composed aggregate containing the accepted M3, D2, O2,
   C2, K2, T1, and numerical lineages exactly once.
3. **TR3-B:** a narrow D2/M3/L2/L3S owner bridge that feeds the accepted M3
   forward/output/VJP/release implementation and adds no second model schedule.
4. **TR3-O:** an O2-private prepare/commit/clear transaction whose post-commit tail
   cannot report a recoverable failure.
5. **TR3-C:** C2-to-live joint restore staging for model, optimizer, dataset cursor,
   RNG, and counters.
6. **E3-PRIVATE:** accepted implementation/runtime evidence for the existing exact
   E3 design, followed by a narrow TR3 adapter to the public evaluation map.

No trainer implementation should begin against guessed versions of those seams.

## 2. Shared rank-two persistence prerequisite

### SHARED-R2 (ACCEPTED/DISPATCHED in issue #114; shared with G3-R)

Keep K1 ABI 1.0, C1 and C2 bytes, all C2/K2 public names and arities, and the
currently verified rank-zero plus rank-one `[0,4611686018427387903]`
`tensor.f32/storage.copy` ranges unchanged. In the one successor registry aggregate,
replace the genuine I2 storage-copy provider and K2's exact provider audit together.
The provider and verified `tensor.f32` entry retain implementation name
`eshkol-transformer-f32`, advance to version `1.1`, and use evidence
`I2:bounded-exact-f32-storage.copy-v2`. Append exactly these ordered rank-two
alternatives:

```text
[2,4] [4,4] [4,8] [8,4] [256,4]
```

`[2,4]` is the M3 learned-position tensor. The other four alternatives cover every
other rank-two M3 parameter and its O2 moments. Rank-one `[4]` is already covered.
The model has 15 logical paths, 14 unique tensors, and one exact tie between
`head/weight` and `token_embedding/weight`.

The successor provider must require `f32`, `cpu`, dense row-major, zero offset,
exact request/view extents, natural alignment, checked exact byte lengths, and
disjoint input/output spans. Success copies exact bits once. It performs no cast,
reshape, transfer, fallback, allocation, or pointer retention. Every other rank-two
shape remains `unsupported`. K2 must discover and exact-audit this actual descriptor
and expose it through its unchanged twelve-operation facade. A report-only edit,
source-only shape matcher, or C2-local substitute validator is forbidden.

Acceptance requires real K1 validate/invoke tests for all five shapes; near-miss,
one-over, alias, layout, dtype, device, and poisoned-output negatives; all existing
rank-zero/rank-one regressions; and public C2 load/release of a checksummed genuine
fixed-model checkpoint containing all 15 paths, 14 unique parameters, the exact tie,
and 28 matching O2 moment tensors. A checksummed outside-profile rank-two artifact
must reject before any codec and return staging/live storage counts to baseline.

I2, P1, O2, C1, and C2 already own arbitrary-rank tensor lifetimes and detached
release. No new rank-two lifecycle operation is needed. The accepted process-local
K2 runtime and caller-region report-shell rules remain. TR3 must consume this one
shared identity and must not publish a competing provider. C2 codec bytes and public
APIs remain unchanged.

## 3. Accepted contracts TR3 can consume

| Owner | Accepted surface or invariant | Boundary relevant to TR3 |
|---|---|---|
| A0 | Nine trainer names above; transactional step; complete update-boundary state; token-weighted metrics | Declaration contract only; no trainer runtime exists |
| X1 | `config-*`; exact 14-key schema; `run.seed`; deterministic CPU/f32; `training.accumulation-steps` | No tokenizer, dataset, optimizer, scheduler, or checkpoint path fields |
| M3T | Fixed model construction, typed initializer, input/logits/workspace owners, 21 forward and 25 reverse fixed roles | Does not compose with D2/O2/K2/C2 and does not run a complete schedule |
| M3 | Eight installed model operations; immutable graphs; `diagnostic-output-vjp!` adds one atomic 14-unique-parameter contribution | Public outputs/VJPs retain cumulative identity/plan controls; no flat training-loop claim |
| L2 | K1 `indexed-cross-entropy.forward` and `.backward`; logits `[1,2,256]`, targets `[1,2]`, loss/upstream `[1,2]` | No mask, reduction, owners, or accumulation |
| L3S | `reduce.bool` and `numerator-seed.bool` for `[1,2]`, plus f32-mask and mean-seed variants | Numerator seed is the required training seed; it must not be divided again |
| O2 | Six public optimizer operations; dense CPU-f32 AdamW; constant/linear update schedule; global-L2 clipping; complete logical state | Step reads numerator/total-weight gradients and does not clear them; no destroy or token schedule |
| D2 | Eleven dataset/batch operations; one live batch; `ESHKDCU1` cursor; exact `17*N*T` carrier | Batch shells must remain in one lexical region; D2 owns no epochs |
| C2 | Checkpoint inspect/load/save, detached trainer-state owner/release, complete 1.0 bytes | Owns detached state only; joint live restore and trajectory proof are TR3 scope |
| K2 | Twelve capability operations; one verified I2 storage-copy row | Current provider genuinely supports rank 0/1 only; rank 2 is unsupported |
| E3 | Accepted exact private no-grad design with four f32 results and token/batch counters | Implementation/runtime unaccepted; no public evaluator exists |

### Exact existing call inventory

The relevant accepted names are fixed; a private proposal below does not rename or
silently replace them.

- X1: `config-parse`, `config-resolve`, `config-validate`, `config-canonical`,
  `config-fingerprint`, and `config-ref`.
- P1/model state: `module-parameters`, `module-state-dict`,
  `module-load-state-dict!`, `state-dict-release!`, `module-train!`,
  `module-eval!`, and `module-zero-grad!`, plus the parameter-tree and state-dict
  accessors defined by A0.
- M3: `model-forward`, `model-output-logits`, `model-output-loss`,
  `model-output-rng`, `model-output-release!`, `diagnostic-output-vjp!`,
  `diagnostic-model-logits-bits`, and `diagnostic-model-logits-release!`.
- M3T construction/owners: `diagnostic-initializer-state`,
  `diagnostic-initializer-algorithm`, `diagnostic-initializer-words`,
  `diagnostic-model-create`, `diagnostic-model-profile`,
  `diagnostic-model-initializer`, four `diagnostic-input-*` operations, four
  `diagnostic-logits-*` operations, and seven `diagnostic-workspace-*` operations.
  Its role surface is eight forward names with matching VJP names—embedding,
  linear, layer norm, GELU, residual, heads split, heads merge, and attention—plus
  `diagnostic-sum!`, for exactly 38 operations total.
- L2 K1 rows: `indexed-cross-entropy.forward` and
  `indexed-cross-entropy.backward` under `kernel.indexed-cross-entropy`.
- L3S K1 rows: `l3s.masked-objective.reduce.bool`, `.reduce.f32`,
  `.numerator-seed.bool`, `.numerator-seed.f32`, `.mean-seed.bool`, and
  `.mean-seed.f32` under `l3s.masked-objective`.
- O2: `optimizer-create`, `optimizer-step!`, `optimizer-zero-grad!`,
  `optimizer-state`, `optimizer-load-state!`, and `optimizer-state-release!`.
- D2: `token-dataset-open`, `token-dataset-next-batch`, `token-dataset-end?`,
  `token-dataset-cursor`, `token-dataset-seek!`, `token-dataset-close!`,
  `token-batch-inputs`, `token-batch-targets`, `token-batch-loss-mask`,
  `token-batch-validate`, and `token-batch-release!`.
- K2: `capability-discover`, `capability-request`, `capability-verified?`,
  `capability-require`, `capability-report-entries`, and the seven
  `capability-entry-*` accessors.
- C2 public: `checkpoint-inspect`, `checkpoint-metadata-ref`, `checkpoint-load`,
  `checkpoint-save!`, and `trainer-state-release!`. Relevant localized seams are
  `c2-training-state-compose-internal`, `c2-checkpoint-load-stage-internal`,
  `c2-checkpoint-load-capability-descriptors-internal`,
  `c2-checkpoint-load-reconstruct-internal`, and
  `c2-checkpoint-load-staging-release-internal!`; they grant no live-trainer
  mutation authority.

The accepted E3 design fixes, but has not implemented and accepted, these private
source calls: `e3-frame-bind!`, `e3-evaluate-into!`, `e3-stage-batch!`,
`e3-forward-schedule-internal`, `e3-forward-role!`, `e3-ce!`, `e3-reduce!`,
`e3-correct!`, `e3-accumulate!`, `e3-finalize!`,
`e3-cleanup-preflight!`, `e3-abort!`, `e3-publish!`, `e3-counter-ref`,
`e3-frame-unbind!`, `e3-frame-d2-record-internal`, and
`e3-d2-release-owned!`. TR3 consumes none until E3 runtime acceptance.

The M3 public contribution contract is exact. `diagnostic-output-vjp!` receives an
already weighted **numerator** seed, positive finite f32 normalization-weight bits,
and the common expected contribution ordinal. On success it contributes all 14
unique gradients once, adds the weight once to each destination, and increments
each contribution count once. O2 later divides every stored numerator by the one
common accumulated weight. Using L3S `mean-seed.bool`, scaling the numerator seed by
weight again, or dividing before O2 would be a correctness defect.

The following exact public/private sequence is already numerically specified, but
is not yet available as one bounded training call:

```text
D2 inputs/targets/bool mask
  -> fixed M3 21-role forward logits
  -> L2 indexed CE per token
  -> L3S reduce.bool => batch numerator N_b, weight W_b, mean
  -> L3S numerator-seed.bool => upstream mask
  -> L2 backward => dlogits numerator seed
  -> fixed M3 25-role reverse and four ordered sums
  -> one I2 14-entry numerator contribution(weight=W_b, ordinal=j)
  -> after A microbatches, one O2 AdamW update using total weight
```

The local orchestration candidate contains a carrier-neutral E3 BOOL metrics
provider and a fixed E3-D2 identity/idle prerequisite. Until merged and accepted,
TR3 may cite those files as candidate evidence only. It must not advertise their
capability rows or call their symbols as accepted runtime.

## 4. Aggregate and public ownership

### TR3-A (PROPOSED)

Build one registry-owning source aggregate. It must source-compose each accepted
E1/P1/I1/I2 lineage once and include exact M3/M3T, T1, D2, O2, C2, K2, L2, L3S,
and accepted E3 sources without linking any completed registry-owning sibling
archive. Raw dependency files, localized-symbol manifests, installed facade
closures, public strings, and one-member archive membership are frozen only after
the implementation exists and independent review classifies every generated name.

The installed facade set adds the remaining eight A0 trainer wrappers beside C2's
existing `trainer-state-release!`. It changes no public arity. Numerical providers
are fixed private resolvers in this closed aggregate. `trainer-create` performs
genuine descriptor/runtime admission for every provider it uses; it does not infer
numerical capability from source or from current K2's unverified kernel rows. Public
K2 remains the authority passed to `checkpoint-load` and therefore separately
requires SHARED-R2.

The aggregate has one persistence-policy authority: C2's authenticated public
`persistence-policy/5` closure. Nested C1/T1 consumers reuse the existing fixed
projection installed by `c2-public-extension.esk` through
`t2-wave2-policy-projector[0] = c2-policy-t2-entry-internal`. A legacy raw C1
five-slot vector is not a second accepted public representation in this lineage.
G3-R must consume this canonical C2 authority or return a concrete incompatibility
for joint review; TR3 must not invent another policy carrier or separately link an
M3/G3 registry aggregate beside C2.

The trainer is an opaque, exclusively mutable process-local receiver. It retains
the resolved config, immutable tokenizer identity, D2 dataset, M3 model, O2
optimizer, reusable M3T input/objective destinations, and, after E3 acceptance, one
reusable evaluation frame. The model, dataset, and optimizer receive one exact
trainer lease at construction. Their mutation adapters in this aggregate must reject
an external mutation while leased; trusted trainer calls present the exact lease identity.
Read-only metadata access is permitted only where it cannot expose mutable storage
or create a competing active frame. Copied/foreign lease tokens grant no authority.

A0 has no trainer destroy operation and the pinned runtime has no proved finalizer.
The trainer, its model, optimizer, reusable input/objective owners and evaluation
frame, and inert identity controls are therefore process-lifetime objects. This must
be reported, not hidden. Batch/view state and per-call regional scratch must never
be retained by the trainer.

## 5. Construction and initialization

`trainer-create` is an assembly operation. It does not construct a model, tokenizer,
dataset, or optimizer. The fresh path is:

```scheme
(diagnostic-initializer-state run.seed)
(diagnostic-model-create resolved initializer)
(module-parameters model)
(optimizer-create o2-config parameter-tree)
(token-dataset-open d2-options tokenizer)
(trainer-create resolved tokenizer dataset model optimizer)
```

Construction validates before publishing any lease:

1. `resolved` is the authentic X1 value for the exact M3 profile, CPU/f32,
   deterministic true, and positive accumulation steps `A`. The initial bounded
   TR3 profile requires `A <= 8192` and `2*A <= 16384`.
2. Tokenizer vocabulary is 256 and its fingerprint equals both D2 cursor identity
   and the trainer's retained identity.
3. D2 is open, idle, has no live batch/borrow, and uses `N=1,T=2`. Its accepted
   directory/limits/shuffle seed/window/packing identity is canonical in the cursor
   and is preserved rather than narrowed by TR3. The current cursor is captured as
   epoch start. Construction may retain an empty or entry-EOS interval, as A0
   permits; its first step and `trainer-state` then raise `invalid-state` without
   model/optimizer mutation.
4. Model profile, config identity, initializer algorithm/key, 15 logical paths,
   14 canonical handles, shapes, and exact tie agree. The cached initializer
   successor is the deterministic 290-block successor of `run.seed`.
5. O2 binds exactly those 14 handles once, has absent gradients, no live state or
   moment borrow, and `completed-updates=0`. Its algorithm/device/dtype and complete
   group/schedule/clipping configuration are admitted O2 1.0 values.
6. Every required fixed native provider/runtime, fenv rule, destination, scratch
   budget, counter bound, and lease slot is available.

Fresh trainer counters are total tokens 0, completed updates 0, and epochs 0. The
current D2 cursor is deep-copied as the immutable epoch-start cursor. The training
Philox state is version 1 with key `run.seed` and counter words zero. It is distinct
from M3T's typed initializer successor. The no-dropout profile consumes no training
RNG, so its counter remains unchanged, but it is still snapshotted, validated, and
restored exactly for C2 compatibility.

Nonzero optimizer/counter initialization is deliberately rejected. Resume always
creates an otherwise compatible fresh trainer and then calls `trainer-load-state!`.

## 6. Trainer state machine and lifetime

### Public phases

| Phase | Invariant | Admitted public calls |
|---|---|---|
| `idle` | train mode; no live batch/view/output/graph; gradients absent; O2/trainer update counters equal | step, train, evaluate, state, load |
| `step` | step-start cursor/RNG/counters saved; 0..A numerator contributions present | no reentry |
| `commit` | all inputs and O2/reset plans preflighted; first write makes the tail nonrecoverable | no reentry |
| `evaluate` | separate dataset and E3 frame active; original 17 modes recorded | no reentry |
| `snapshot` | P1/O2/C2 detached-owner construction in progress | no reentry |
| `restore` | complete live-restore plan staged; first write starts nonrecoverable tail | no reentry |

Every recoverable failure returns the trainer to `idle`. Before a step commit, this
means restoring the exact step-start D2 cursor, RNG, modes, and counters and clearing
only TR3's accumulated gradients. After the first parameter or moment write no
recoverable error may be returned. Process termination or hardware loss recovers
from the last durable checkpoint.

One source entry authenticates the trainer, its lease identities, phase, counters,
RNG, current/epoch-start cursor identities, reusable owners, E3 frame, and fixed
metric destinations. Native pointers are derived only after exact source and native
registry admission. No pointer, callback, provider, parameter selector, gradient
destination, lease token, or state slot is caller-selected.

### TR3-B objective bridge (PROPOSED)

TR3 consumes the accepted M3 implementation rather than duplicating its 21-role
forward, 25-role reverse, graph checks, fixed14 contribution, or cleanup. For each
microbatch it uses one reusable genuine M3T input owner and one reusable genuine
M3T diagnostic-logits owner as the VJP seed:

1. Copy the authenticated D2 input view synchronously into the M3T input owner.
2. Call accepted `model-forward model input ':deterministic? #t`, then
   `model-output-logits` to obtain the graph-retaining read-only logits owner.
3. Through synchronous scoped views, dispatch L2 forward, L3S `reduce.bool`, L3S
   `numerator-seed.bool`, and L2 backward directly into the genuine reusable seed
   owner. Targets and mask remain borrowed only inside the live D2 batch region.
4. Call accepted `diagnostic-output-vjp!` with that output, seed owner, exact `W_b`
   bits, and microbatch ordinal `j`.
5. Release the logits lease and output lease on every path before releasing the D2
   batch and leaving its lexical region.

Two concrete ownership gaps require this private bridge. `model-forward` admits an
exact M3T-owned input, not a D2 state-backed identity; and L2/L3S expose carrier-
neutral K1 operations while `diagnostic-output-vjp!` requires a genuine M3T seed
owner. The bridge supplies only synchronous authenticated view/copy/dispatch seams.
It introduces no alternative model schedule, graph, contribution implementation,
or public tensor conversion.

Proposed source-private logical calls are `tr3-stage-d2-input!/2` and
`tr3-objective-vjp!/5` (output, graph-retaining logits, live batch, seed owner,
ordinal). These names and arities are not tested upstream APIs and require exact
review. Failure before the accepted M3 contribution preserves all 14 gradient
states. A successful accepted contribution advances all 14 once and is followed
only by preflighted lease/batch cleanup; it must not be returned as retryable.

Each D2 batch and each M3 output/logits shell lives in one lexical `with-region`.
No batch shell, tensor shell, raw view, output, logits lease, seed view, or caught
error graph escapes. The accepted M3 graph/output and I2 plan tombstones remain
cumulative controls after release; they are measured and capped for the supported
Wave 3 horizon rather than misreported as live storage.

For microbatch ordinal `j` in `0..A-1`, every one of the 14 I2 destinations must
have contribution count exactly `j` and the same accumulated weight before the
contribution. L3S `numerator-seed.bool` supplies `mask`, L2 backward supplies dlogits,
and accepted M3 supplies the fixed reverse transcript and one 14-entry transaction.
Weight bits `W_b` are metadata only. Failure changes no gradient bytes/counts/weights.
Success changes all 14 once.

Step metrics aggregate batch numerators and weights in microbatch order using a
reviewed binary32 operation with no mean-of-means. The accepted E3 accumulation row
may be reused only after its implementation merges and its exact bounds fit; until
then this accumulator is a missing TR3-B numerical contract. The final step loss is
`f32(total_numerator / total_weight)` with positive finite weight.

## 7. Step transaction, counters, and schedules

`trainer-step!` executes exactly `A = training.accumulation-steps` microbatches and
one optimizer update:

1. Require `idle`, all 14 gradients absent, O2/trainer counters equal, canonical
   train mode, and no active D2 batch/view. Deep-copy the small step-start cursor and
   record RNG/counters/modes before mutation.
2. Fetch one D2 value. For a batch, run the accepted M3 path through TR3-B, release
   every output, and then release the batch. For EOS, increment the staged epoch
   count, seek the captured epoch-start cursor, and retry the same
   microbatch ordinal. If the seeked dataset immediately returns EOS, fail
   `invalid-state` and roll back the step. One update may cross multiple epochs.
3. Count true boolean mask positions. Stage checked additions to this step's token
   count, global total tokens, and epoch count. D2 guarantees one or two active
   positions per batch.
4. After exactly A contributions, require all 14 counts equal A and all 14 positive
   finite weights bit-equal. Prepare the O2 update and gradient-clear tail, including
   finite AdamW results, clipping, effective learning rates, counter increment, and
   every storage/alias identity.
5. Commit parameter values, moments, O2 `completed-updates`, gradient absence,
   trainer total tokens/update/epoch counters, and idle phase as one nonallocating,
   nonraising tail. Dataset/RNG advances made during accumulation become committed.

### TR3-O optimizer seam (PROPOSED)

The current public sequence `optimizer-step!` followed by `optimizer-zero-grad!`
cannot by itself prove A0's post-commit no-error rule. O2 must expose a localized
same-owner trainer transaction, inaccessible from an installed facade:

```text
o2-trainer-step-clear-prepare-internal(optimizer, expected-update,
                                       expected-contributions)
o2-trainer-step-clear-commit-internal!(plan)  ; nofail after first write
o2-trainer-step-clear-abort-internal!(plan)   ; before commit only
```

This is a **PROPOSED** logical interface, not a tested signature. Prepare reuses
O2's accepted numerical staging, requires count A and common weight for all 14
gradients, and preflights reset eligibility for the same canonical handles without
acquiring a conflicting second I2 plan. Commit keeps
the accepted AdamW order, clipping, schedule, storage identity, and failure atomicity,
then clears gradient bytes/count/weight without another fallible admission. It adds
no public optimizer operation and does not change O2 state bytes.

O2 owns the only schedule cursor: `completed-updates`. Constant and linear schedules
remain exactly as accepted. The trainer stores the same completed-update count only
for cross-component validation and C2 counters. A mismatch is `invalid-state` and
never repaired silently. Tokens and epochs do not select the O2 schedule.

Step result keys are exactly:

| Key | Exact meaning |
|---|---|
| `loss` | f32 weighted mean for this committed update |
| `mask-weight` | f32 sum of this update's boolean weights |
| `tokens` | i64 active positions in this update |
| `microbatches` | i64 exactly A |
| `update` | i64 global completed-update count after commit |

`metrics-ref` accepts only the exact keys of the metrics kind and returns a detached
scalar. Unknown key or wrong-kind receiver is `invalid-argument`; stale/foreign
identity is `invalid-state` or `invalid-argument` under the standard registry rules.
Metrics never retain trainer state or tensor storage.

## 8. Stop policy and `trainer-train!`

`trainer-stop-policy` requires each limit to be `#f` or a positive exact i64 and at
least one limit. Because D2 is finite, `max-epochs` is admitted. The policy is an
immutable detached CPU value.

Root selects an explicit TR3 refinement where stop-policy limits are invocation
deltas; A0 did not previously choose between deltas and absolute global counters.
`trainer-train!` checks supplied limits after each committed update, in the fixed
order tokens, updates, epochs, and stops when any delta from the entry counters is at
or above its limit. The last update may cross a token or epoch limit. O2 schedule and
C2 counters remain global. A resumed CLI must describe these as **additional**
tokens/updates/epochs, never an absolute target. There is no wall-clock limit,
hidden early stop, or logging side effect.

The summary covers only updates committed by this invocation:

| Key | Exact meaning |
|---|---|
| `loss` | weighted mean over every microbatch committed by this invocation |
| `mask-weight` | their f32 weight sum |
| `tokens` | i64 active-position delta |
| `updates` | i64 completed-update delta |
| `epochs` | i64 completed-epoch delta |

Summary numerator/weight accumulation follows microbatch order and binary32 checks;
it never averages step means. Counter or finite-f32 overflow fails after preserving
all already committed updates and rolling back only an in-progress update. The
supported TR3 profile must publish an explicit maximum updates/microbatches/tokens
per call justified by its accumulator and memory tests. A request whose stopping
limits cannot remain within that admitted bound is `unsupported` before the first
update; the implementation must not truncate it or lower limits silently.

## 9. Evaluation through E3

`trainer-evaluate!` requires an authentic open D2 dataset distinct from the leased
training dataset and compatible with the same T1 identity and fixed profile. The
initial E3 design further requires its evaluation dataset to use seed `#f` and
shuffle window 1; that restriction does not narrow the training dataset. TR3 enters
the trainer guard, invokes the accepted E3 private transaction, and projects only
these public values:

```text
loss mask-weight tokens batches
```

E3's private perplexity and token accuracy results may be computed as required by
its accepted transcript but are not added to the A0 evaluation map. The result is a
new immutable metrics value.

E3 owns exact all-17-mode snapshot/eval/restore and evaluation-dataset cursor
restore, no-grad execution, four-scalar/six-output publication, and token/batch
counters. TR3 additionally proves that the leased training cursor, RNG, parameters,
O2 state, gradients, scheduler, and trainer counters are unchanged. Zero total
weight is `invalid-state`.

The accepted E3 private design is currently insufficient to claim this public path:
the merged shared-guard source still requires its recorded runtime disposition, and
the E3 consumer, P1 mode seams, D2 restore/staging seams, BOOL metrics, package, and
independent acceptance must merge first. TR3 then needs one narrow lease-aware
adapter and public metrics projection. No second evaluator, mode implementation, or
cursor restoration protocol should be invented.

## 10. Trainer state and canonical checkpointing

`trainer-state` is admitted only in `idle`, with canonical train mode, absent
gradients, no batch/view/frame/state borrow, and O2/trainer update counters equal.
It obtains:

- a detached P1 model state;
- a detached O2 state, including optimizer config, schedule definition, moments,
  and completed updates;
- exact training RNG words;
- tokenizer fingerprint and vocabulary;
- current and epoch-start D2 cursor bytes;
- exact X1 canonical bytes and fingerprint; and
- total tokens, completed updates, and epochs.

It transfers those owners into the accepted C2 1.0 trainer-state owner. Failure
releases every constructed prefix and leaves the live trainer unchanged. The result
is the exact object accepted by `checkpoint-save!` and released by C2's existing
`trainer-state-release!`. Mid-accumulation snapshots remain unsupported.

C2 serialization stays canonical, checksummed, versioned, data-only, and
non-executable. It never serializes a trainer/frame/lease identity, active batch,
raw pointer, provider callback, capability report/evidence, mode scratch, live
gradient, logging state, filesystem path, timestamp, or host identity. Schedule
values are derived from O2's definition and completed update, not redundantly stored.

Repeated saves of one state with the same options must be byte-identical. The same
logical update boundary reached by uninterrupted and resumed execution must produce
byte-identical C2 bytes, including model bits, 28 moment tensors, both D2 cursors,
RNG words, config/tokenizer identities, and counters.

## 11. Joint live restore

### TR3-C (PROPOSED)

`trainer-load-state!` borrows an exact live C2 owner and restores an already-created
compatible trainer. It does not consume or adopt any detached input carrier. It is
admitted only in `idle` with absent gradients.

Validation precedes live mutation:

1. Authenticate the C2 owner and borrow all component state through existing C2,
   P1, and O2 authorities.
2. Require exact library/compiler/provider versions, X1 canonical bytes/fingerprint,
   tokenizer identity/vocabulary, fixed model profile, 15 paths/14 unique/tie,
   tensor shapes/dtypes/devices/layouts, O2 configuration validity, and D2 invariant
   cursor bytes. The epoch-start ordinal must be before EOS and no later than the
   current ordinal.
3. Require C2 counter `completed-updates` equal O2 state count; validate i64 bounds,
   RNG version/key, finite moments, and absent live gradients.
4. Stage every model and moment bit copy, optimizer control replacement, D2 ordinal,
   RNG/counters, and canonical train modes. Preallocate all plans and prove the tail
   contains no allocation, callback, provider dispatch, or recoverable check.

The exact implementation seam must combine rather than sequence the current public
`module-load-state-dict!`, `optimizer-load-state!`, and `token-dataset-seek!` calls.
Those operations are individually atomic but do not provide cross-component
atomicity. The fixed profile has 14 parameter values and 28 moment destinations,
well below I2's accepted 4,096-assignment transaction ceiling. A proposed joint plan
may therefore stage 42 tensor destinations plus optimizer, D2, RNG, mode, and counter
controls, then commit without changing tensor storage identities. A failed prepare
leaves every live receiver and the borrowed C2 owner unchanged. After the first
write, the tail completes or fail-stops; it never reports retryable partial restore.

The restore plan publishes the D2 current/epoch-start pair and trainer counters only
after their exact identities remain valid. Public seek is not called in the commit
tail. Successful restore ends in `idle`, train mode, absent gradients, no active
frame, and O2/trainer update counts equal.

SHARED-R2 is a hard prerequisite: current public `checkpoint-load/3` rejects every
genuine M3 rank-two model before reconstruction. Private genuine provider admission
inside TR3 cannot replace the caller's public K2 report at the C2 boundary.

## 12. Failure atomicity and error mapping

Validation order is receiver identity/lifecycle, phase/reentry, lease identities,
config/profile/tokenizer/D2 topology, model/P1 topology, optimizer binding/state,
capability/provider/fenv, then operation-specific values. Existing dependency errors
retain their A0 categories. TR3-specific rules use:

| Failure | Category |
|---|---|
| malformed option/policy/key/foreign receiver | `invalid-argument` |
| wrong fixed extent/path/tie/counter relation | `shape-mismatch` where structural, otherwise `invalid-state` |
| non-f32 or non-CPU object | `dtype-mismatch` / `device-mismatch` |
| non-dense or nonzero-offset carrier | `noncontiguous` |
| busy/leased by another trainer, live batch/view, present unexpected gradient, empty epoch interval | `invalid-state` |
| fixed profile, rank, accumulator bound, or provider unavailable | `unsupported` |
| required binary32/deterministic controls unavailable | `determinism-unavailable` |
| malformed/checksum-invalid detached state | `corrupt-data` |
| incompatible state or format version | `version-mismatch` |
| admitted invariant or impossible cleanup defect | `internal` or fail-stop after an irreversible write |

No error becomes a partial metrics value, skipped batch, NaN sentinel, cleared
checkpoint state, lower precision, CPU/scalar fallback, approximate gradient, or
smaller hidden limit.

## 13. Reproducibility and acceptance evidence

### Numerical and shape evidence

The compiled production transcript must be checked against an independent direct
mathematical fixed-model reference. Python/PyTorch may generate development fixtures
and references, but no production trainer source, object, archive, AOT executable,
or runtime may import or invoke Python. Required checks include:

- all 14 parameter gradients, including small Q/K/norm tensors and tied
  head-plus-embedding exactly once;
- unequal-mask-weight accumulation against one backward of the combined numerator;
- L3S numerator seed versus a deliberately wrong mean seed;
- every malformed shape/index/mask/cursor/state and all late failure points;
- exact fixed contribution ordinals and common weight metadata;
- O2 clipping, constant and linear schedules, AdamW continuation, and finite-result
  rejection; and
- normal plus ASan/UBSan/LSan runs with leak detection on a supported host.

### One-batch overfit proof

Freeze one nonempty T1/D2 corpus that produces one `[1,2]` batch with two active
positions and a fixed O2 constant-schedule configuration. From the canonical seed,
record evaluation loss, run a fixed positive number of committed updates, and
evaluate again on the same cursor-restored corpus. Both active target IDs must be
the unique logits argmax and final mean CE must be finite and `< 0.05`. Require
finite positive weights throughout, exact A microbatches per update, and no parameter
except the 14 admitted unique tensors to change. A deterministic predeclared
independent reference supplies the complete loss/parameter trajectory and tolerances;
loss decrease or reference parity without the argmax/CE thresholds is insufficient.

### Held-out evaluation proof

Use a distinct D2 dataset with at least two batches, unequal valid-mask counts, and
different token transitions. Compare E3/TR3 loss, weight, token, and batch values to
the independent global numerator/weight reference before and after training. Prove
the evaluation dataset cursor and all 17 modes restore on success and every injected
failure. Prove the training cursor, parameters, O2 moments/counter, gradients, RNG,
and trainer counters are unchanged by evaluation. Include one fixture whose held-out
loss decreases after the fixed training run; retain exact reference parity as the
correctness criterion even if monotonicity changes under a later fixture.

### Interrupted/resumed equivalence

Run each case in two fresh processes, with a resumed suffix `R >= 3` updates:

1. uninterrupted: perform `K+R` updates and save state;
2. resumed: perform K updates, take `trainer-state`, save, public-load through the
   SHARED-R2 K2 report, restore a fresh compatible trainer, perform R updates, and
   save again.

Require bit-identical 14 model tensors, 28 O2 moments, next-step metrics, effective
learning rates, O2/trainer counters, RNG words, current and epoch-start cursor bytes,
and final canonical C2 file bytes. Run one case with accumulation 1 and another with
`A >= 2` and unequal microbatch mask weights. Place K so a resumed update crosses D2
EOS and therefore proves epoch restoration. Inject a late failure immediately before
commit, retry the same step, and require the remaining multi-update suffix and final
checkpoint to equal the uninterrupted trajectory.

### Memory and lifetime evidence

TR3 must prove fixed live ownership, not infer total-memory flatness from component
payload arithmetic. On the supported Ubuntu 22.04/LLVM-Clang 21 lane, measure after
warmup at 1,024 and 8,192 committed updates on one trainer:

- exact live M3 output/graph/logits, reusable input/seed, runtime/pin/view counts and
  payload bytes;
- D2 live dataset/batch/view/carrier counts and file descriptors;
- O2 live optimizer/moment/transaction controls;
- Eshkol retained root and caller-region bytes;
- cumulative retired/tombstone controls separately from live storage; and
- advisory peak RSS separately.

Live graph, output, logits, batch, view, pin, and scratch-owner counts must return to
the same idle baseline after every success and injected failure. Per-microbatch
lexical regions must retain no growing live graph. The supported bounded run must set and pass an
explicit retained-byte/control ceiling and reconcile every nonzero 1,024-to-8,192
delta. Ordinary M3/I2/O2 identity tombstones are accepted measured component
behavior; TR3 must not relabel them as live payload or claim indefinite flat total
process memory. Any new unbounded live-frame or batch retention blocks acceptance.
Practical indefinite-memory reduction and activation checkpointing remain Wave 4.

## 14. Unsupported cases and follow-up risks

This proposal does not support arbitrary model shapes, batch/sequence sizes,
non-byte tokenizers, streaming datasets, f32 masks, dropout,
cache training, compiler autodiff, sparse gradients, mixed precision, GPU, concurrent
or reentrant mutation, mid-accumulation checkpoints, optimizer destruction, trainer
destruction, power-loss rollback beyond the last atomic checkpoint, or a Python
training runtime.

The material risks requiring independent review are:

1. one source aggregate must reconcile M3, D2, O2, C2, K2, and E3 registry ownership
   without duplicate symbols or guessed generated names;
2. TR3-B must preserve the accepted public M3 graph/output/VJP/release semantics
   while proving D2 and objective-owner lifetimes and measuring cumulative controls;
3. O2 commit plus gradient clear and C2 live restore need new cross-owner nofail tails;
4. E3 runtime remains dependency-gated and its local metric/D2 candidates are not
   merged acceptance;
5. current K2 rank-two rejection blocks public resume until SHARED-R2 merges; and
6. X1 intentionally excludes optimizer/dataset configuration, so their identities
   must continue to come from O2 state and D2 cursor rather than an incompatible X1
   schema extension.

The implementation order is SHARED-R2 and E3 prerequisites in parallel; then
TR3-A exact aggregate inventory; TR3-B and TR3-O; TR3-C; public trainer wrappers;
focused numerical, transaction, persistence, and bounded-memory gates; independent
TR3-R review; supported full CI; and only then Wave 3 acceptance.

## 15. Smallest dependency-ready implementation unit

The next TR3 unit that can start from current accepted dependencies is **TR3-B:
one-microbatch objective bridge and compiled witness**. It does not wait for
SHARED-R2, E3, O2 commit integration, C2 live restore, or public trainer wrappers.

Its exact bounded deliverable is one source-composed private artifact that:

1. constructs an accepted M3 model, genuine reusable M3T input and seed owners, and
   an accepted T1/D2 `N=1,T=2` dataset;
2. stages one live D2 batch into the M3T input;
3. calls the accepted M3 forward/output path, real L2 forward, L3S BOOL reduction
   and numerator seed, real L2 backward, and accepted M3 VJP at ordinal zero;
4. returns only detached exact-bit `N_b`, `W_b`, and mean observations to its test
   driver, then releases output/logits/batch and explicitly clears gradients; and
5. proves all 14 numerator gradients against the independent reference, including
   the tied sum and 01/10/11 masks, plus late-error atomicity and exact live-owner
   baselines under normal and sanitizer builds.

It adds no public operation, optimizer step, trainer state, checkpoint field,
capability row, alternate model schedule, or persistence policy. The focused handoff
must freeze the two private bridge calls, source/native closure, output/cleanup
ordering, and measured per-call cumulative control cost before TR3-A or TR3-O
consumes it. This isolates the only currently proven D2-to-M3/objective ownership
gap and gives the later trainer a real tested upstream seam instead of an invented
one.
