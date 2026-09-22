# E3 evaluation: proposed composition and reachability

Status: **proposed, design/reachability only** for [issue #99](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/99).
Integration owns acceptance, prerequisite dispatch, API/ABI decisions and merge.
No production implementation, public API amendment, runtime/format freeze, full CI,
or E3 completion is authorized by this document. Base is merged main
`fe2493168e72cd7bf81d810c9c172c40aacfee61` (tree
`902d4645e13c133a811f1c1cf72ffb799976a58a`). L3S runtime is merged; its historical
pending text on this base awaits PR #98 closeout. Neither separate component
acceptance nor private test carriers establishes a joint installed evaluator.

## 1. Decisions requested from integration

1. **Private core first; public choice remains open.** The smallest next slice is
   private E3 composition reusing accepted I2 scalar storage (comparison below).
   Separately decide whether to accept the proposed four operations and six-key
   report in section 3. The optional standalone report uses
   read-only rank-zero f32 handles explicitly, not ordinary Eshkol flonums.
   Leave A0 `trainer-evaluate!/2`, `metrics-ref/2` and its exact four-key map
   unchanged. TR3 must separately decide its scalar projection and owns authentic
   trainer construction, training-dataset distinction and lease validation.
2. **Byte-only topology.** Accept a new single source-composed M3-lineage sibling,
   a fixed D2 byte-tokenizer identity adapter, and the conditional 108-global /
   102-export / nine-facade target below. Do not combine completed archives or
   claim T2 BPE, K2, O2 or C2 composition. The exact build tuple/manifests require
   separate prerequisite review before any freeze.
3. **Shared no-grad prerequisite.** Extract/reuse the accepted G3-T guard/fixed14
   pin contract under one owner, with an explicitly reviewed sequence-length-2 (T=2) forward-only frame.
   Preserve G3-T's accepted guard semantics and re-review affected design text;
   do not silently fork the guard or implement against unmerged G3-N/S/T.
4. **Restoration and staging.** Accept sequential unchanged-type D2 staging and
   a narrow prevalidated nonallocating cursor/mode restoration contract. Public
   seek/mode setters are not sufficient rollback primitives.
5. **Numerics and bounds.** Accept fixed ordered batch aggregation, weighted
   accuracy, lowest-index ties, finite-only perplexity, bool-only D2 ingress,
   current-cursor-to-EOS traversal, and at most 8,192 batches per invocation.
   No arbitrary partition or permutation bit-equivalence is promised.
6. **Metric owner prerequisite.** Accept explicit release and f32 bit inspection,
   with bounded report/control retention and a reviewed authenticity mechanism.
   A report's f32 handles are not an implicit amendment of A0 scalar semantics.

These are concrete proposed decisions, not requests for routine execution approval.
Changes to names, arities, types, ownership, shape admission or topology return to
[issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1) before implementation.

### Smallest next slice versus optional public surface

| Choice | Metric representation and reachability | Added prerequisites and cost |
|---|---|---|
| **Recommended next bounded slice: private E3 core using existing I2** | Trusted caller preowns independent rank-zero I2 f32 output scalars plus exact i64 counters; synchronous accepted I2 bit-copy/borrow inspection, existing explicit destruction. No public metric shell, ordinary flonum, new scalar ABI or installed evaluator claim | E3-CG, E3-D, E3-N, E3-T and reviewed private source composition; still needs all actual no-grad/restore/identity work. Public E3 acceptance stays open |
| Optional standalone diagnostic API in section 3 | Four new public operations, six-key owned rank-zero-f32 report; exact bit inspection | Adds E3-V owner/authentication/quota/release, E3-P installed tuple and E3-E public facade proof; explicit new API decision |
| Wait for TR3/A0 only | Original four-key trainer result through authentic trainer | Does not remove no-grad/D2/numerical prerequisites; requires TR3 construction and an accepted true-f32 scalar projection, so cannot be E3's upstream dependency |

The private route uses actual accepted I2 owners/views, not copied test-bridge
carriers or a custom scalar encoding advertised as a numeric API. Proposed private
`e3-evaluate-into!/3 (model,dataset,trusted-frame)` stages results into a closed,
preconstructed destination frame. Its four caller-owned I2 output scalars/counters
remain unchanged on failure and become readable only after exact restoration;
private caller ends all borrows and destroys owners with existing accepted I2
operations. The frame is inaccessible to public callers. Reuse it in long-loop
witnesses; don't publish new I2 identities per batch. A new public owner is not a
prerequisite for proving private composition.

The minimum E3-N slice can expose bool correctness plus accumulation/finalization
privately; weighted f32-mask rows and the optional public report require explicit
root disposition and do not create public mask ingress. Existing L3S f32 rows are
sufficient to specify/test weighted reference semantics, not accuracy execution.
Regardless of route, exact private seam inventories and a source-composed tuple
must be reviewed before implementation. A private-only result is a composition
milestone, **not** complete roadmap evaluation or an installed public path.

## 2. Current source and compiled reachability

| Component | What merged source actually provides | Missing E3 authority |
|---|---|---|
| M3/M3T | Authenticated fixed model, 21 forward roles, graph-retaining public forward, reusable transport | True no-grad frame, D2 ingress, target/mask/metric transport |
| D2 | Finite dataset, i64 inputs/targets and bool mask, synchronous single-view borrow, detached cursor | Joint M3 identity, infallible transaction restore, f32 masks |
| L2 | Explicit native CE provider, borrowed logits/targets to per-token f32 losses | Eshkol owner or model/loss facade |
| L3S | Six native bool/f32 rows at `[1,2]`, numerator/W/mean and seeds | Accuracy, multi-batch accumulator, perplexity or scalar facade |
| I2/P1L | CPU-f32 owners, scoped native borrows, parameter/gradient binding | Public ordinary f32 scalar representation or general metric owner |
| K2 | Fixed I2 discovery report; storage.copy is its verified row | Evaluation dispatch/report or joint provider discovery |
| A0/TR3 | Declaration of trainer-evaluate!/2 and exact four-key map | trainer-create, trainer-evaluate!, metrics-ref implementations |

Source anchors: `native/m3_model_extension.esk` unconditionally captures a graph
following `m3-forward-schedule-internal`; `native/m3_schedule.esk` supplies the
21-role sequence. Eval mode does not suppress graph capture. M3 optional target,
mask and RNG ingress rejects. `lib/transformer/trainer.esk` provides only C2
`trainer-state-release!`. Existing L2/L3S AOT bridges are development fixtures,
not candidate production owners.

M3's root loads M3T, then I2/T1; D2's root loads T2 plus I2 and D2 extensions.
D2 `d2-tokenizer-identity` calls `t2-private-tokenizer-fingerprint/vocab-size`;
M3T tokenizer input admission calls `t1-wave1-tensor-admitted?`. Appending roots
or linking both localized archives does not reconcile those identities. D2
`d2-token-tensor-with-view-internal/2` rejects a second active borrow from the
same dataset, even for a different plane. Public `token-dataset-seek!` invokes
allocating cursor validation, then changes ordinal/status/cache validity.

### Bounded probes performed

Actual task `01a0c76a-e9eb-70a3-a0de-7d7f6991aa13`, workspace
`/home/gabe/.codex/worktrees/857e/eshkol-transformer`. Host permission instructions
report `danger-full-access` and `approval_policy=never`; no mismatch was observed.
All shell tools used `/usr/bin/bash`, `login=false`. Shared `.deps` source/build
were read-only; all scratch is workspace-local `.tmp/e3-probes/`.

Canonical compiler source was clean at
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`; version `1.3.4-evolve`;
binary SHA-256 `3e0b923e2e272a89474dff6739b6a2023d71ae483863a64b685ec8ea71ca1cc0`.
`scripts/common.sh` `verify_toolchain` checked source, version, CMake/provenance
and binary hash using explicit `ESHKOL_SOURCE_DIR`, `ESHKOL_BUILD_DIR`,
`LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config`, `ESHKOL_ALLOW_UNSUPPORTED_HOST=1`.
The host is CachyOS x86-64, Clang/LLVM 22.1.6: **compatibility evidence only**,
not supported Ubuntu 22.04 / LLVM 21.1.8 evidence.

Probe command form (absolute source/output/include paths, local cache, no JIT):

```text
<verified .deps/eshkol-build/eshkol-run> --strict-types --no-stdlib \
  -I <workspace>/lib --compile-only --emit-depfile <probe>.d \
  --dump-ir <probe>.esk -o <probe>.o
```

| Probe | Result | Evidential limit |
|---|---|---|
| Literal baseline, then AOT link/run | exit 0, exactly `E3-PIN-AOT-OK\n`, empty stderr | Compiler/AOT works; no evaluator exercised |
| Require model + data, call model-output-loss and token-batch-targets | object exit 0; `nm -u` contains M3 and D2 boxed wrappers | Source/object reachability only; no joint link/runtime attempted |
| Require trainer, call trainer-evaluate! | exit 1, `Unknown function: trainer-evaluate!` | Installed source lacks this implementation |
| Require trainer, call metrics-ref | exit 1, `Unknown function: metrics-ref` | A0 declaration is not a metric accessor |
| Require transformer.evaluation | exit 1, module not found | No current evaluation facade |
| Ordinary scalar `1.25` | object exit 0; IR stores tag 2 and payload 4608308318706860032 | Header maps tag 2 to DOUBLE; cannot call it physical f32 |
| C11 L2/L3S/I1/I2 header consumer | warning-clean object; unresolved L2/L3S accessors as expected | Native declarations coexist; no provider dispatch or owner composition proved |

C command: `clang -std=c11 -Wall -Wextra -Werror -I include -c
.tmp/e3-probes/provider-headers.c -o .tmp/e3-probes/provider-headers.o`.
Exact commands, exits and paths are in `results.json`; probe source/generator,
`.log`, `.d`, `.o.ll`, `.undefined.txt`, baseline stdout and toolchain log remain
in `.tmp/e3-probes/`. This document records the essential results independently
of ignored scratch. No toolchain build, full test run, aggregate build, numerical
campaign or runtime reachability claim is made. Exploratory reads of nonexistent
candidate filenames were corrected using `rg --files`; they are not capability failures.

## 3. Proposed API and report owner

Optional new standalone `transformer.evaluation`, proposed fixed arities; none is
approved or required for the recommended private milestone:

| Operation | Arity | Contract |
|---|---|---|
| `diagnostic-evaluate! model dataset` | 2 | Evaluate admitted suffix, restore cursor/all modes, publish one owned report |
| `evaluation-metrics-ref report key` | 2 | Exact six-key lookup; four floating keys return cached read-only rank-zero CPU-f32 handles; counters return exact i64 |
| `evaluation-metrics-f32-bits report key` | 2 | For a floating key only, return detached exact i64 in `[0,4294967295]` holding its IEEE binary32 word |
| `evaluation-metrics-release! report` | 1 | Invalidate report and dependent scalar handles, release four scalar owners exactly once; exact dead release idempotent |

No options, provider callbacks, generic owner factory, trainer alias or implicit
float widening. Proposed result has exactly:

| Key | Stored type and meaning |
|---|---|
| `loss` | f32[]; globally weighted mean CE |
| `mask-weight` | f32[]; ordered mask-weight sum |
| `perplexity` | f32[]; finite expf(loss) |
| `token-accuracy` | f32[]; globally mask-weighted correct fraction |
| `tokens` | i64; count of original mask positions > 0 |
| `batches` | i64; successfully evaluated batches; EOS excluded |

The four scalar owners have 16 total payload bytes and are mutually disjoint,
newly owned, CPU-only and graph-free. Handles are state-backed, immutable by
observation and cached; no ordinary Eshkol f64 value represents those fields.
The bit accessor is an explicit inspection facility, not a substitute arithmetic
path. Rank-zero f32 ownership and trusted borrowing require E3-V below; no generic
public scalar arithmetic or serialized metric format is proposed.

Report publication owns all four scalars atomically. It retains no model,
dataset, batch, workspace, gradient graph or raw view. It survives dataset close
and scratch destruction. Release first invalidates all handles, then drains the
four scalar owners through a prevalidated nonallocating tail. Unknown key,
non-symbol key, integer key passed to the bits accessor, forged/copied/wrong-kind
or foreign-aggregate owner: `invalid-argument`. Ordinary aliases to an authentic
owner remain valid; copied means reconstructed/forged shell identity, not aliasing. Authentic dead report/handle use:
`invalid-state`; active trusted borrow blocks release before mutation. Scalar
handles cannot individually release storage or authorize mutation. Detached i64
counters/bits may outlive the report. For this optional design, E3-V must retain the authenticated report and four
cached scalar shells in a proved root-owned lifetime before publication, and keep
inert identity roots after release within the reservation quota. The caller owns
payload-release responsibility; registry roots must never point into freed caller
regions or allow address reuse to authenticate stale shells. This deliberately
bounded cumulative storage is not claimed reclaimed by release. If safe promotion
cannot be proved, this candidate owner is blocked pending a separately reviewed
authentication design; no guessed pointer or implicit finalizer is acceptable.
Unrelated D2 batch shells retain their existing live-region requirement.

A0's current `trainer-evaluate!` map remains exactly `{loss, mask-weight, tokens,
batches}` with its stated f32 scalar semantics. These standalone handles do not
fulfill that interface. Future TR3 calls a shared private evaluation core after
its own lease/distinct-dataset checks, then uses a separately accepted metric
projection. E3 never imports TR3 and supplies no fake trainer constructor.

Proposed private Eshkol control seams (no public/native ABI frozen):
`e3-evaluate-internal!/3 (model,dataset,closed-operation-tag)`;
`e3-prepare!/2 (model,dataset)`;
`e3-stage-batch!/2 (frame,batch)`;
`e3-forward!/1`, `e3-accumulate!/1`, `e3-finish!/1`, `e3-abort!/1` (frame).
Only fixed wrappers choose the operation tag. Frame contains closed roles and
phase state. A successful finish commits exactly once and excludes abort; any
numerical/report finalization failure before commit leaves the frame abortable.
Abort then performs exactly-once cleanup without publication. These
names specify responsibility/arity, not callable merged seams. E3-T must enumerate
every concrete native call, slot, authority, error mapping and cleanup transition
for separate acceptance before production work.

## 4. Admission and held-out traversal

Only authentic sealed `eshkol-diagnostic-byte-decoder-v1` models: CPU f32,
N1/T2/V256/D4/Hq=Hkv2/Dh2/L1/F8, learned positions, separate bias-free projections,
tied head, exact-erf GELU, pre/final affine LayerNorm with epsilon `0x3727c5ac`,
no dropout/cache. Exact admitted X1 config/initializer identities, deterministic
construction, 14 unique handles, 15 logical paths and 1,184 values (4,736 bytes)
are verified; shape resemblance is insufficient. No config mutation or larger
context is admitted. All affected module modes must be enumerated and restored.

Dataset is authentic same-new-aggregate D2, open and idle with no live batch/borrow,
N1/T2/V256, canonical byte-tokenizer identity with no special tokens, finite order,
and no shuffle (`:shuffle-seed #f`, `:shuffle-window-rows 1`). Both D2 packed and
unpacked modes are admitted with their unchanged row/target semantics. An alternate
tokenizer with V256 is insufficient. M3 has no tokenizer binding: this is an E3
admission policy over D2's stored identity, not a claimed existing model fingerprint.

Evaluate from the saved **current** cursor through EOS; do not rewind, create an
epoch or repeat data. The caller supplies a held-out dataset/corpus; E3 does not
certify disjointness/contamination. TR3 later enforces identity distinction from
its training dataset. Preflight exact remaining rows/batches: at most 8,192; reject
one-over as `unsupported` before mode/cursor mutation, with no hidden truncation.
At most 16,384 true positions ensures bool W/count exactness in binary32; i64 adds
are still checked. Dataset generation exhaustion remains explicit `unsupported`.

Empty dataset or entry-at-EOS yields `invalid-state` (zero evaluated weight), with
exact restoration and no report. Every genuine D2 batch has at least one true mask.
All-masked native input rejects; a forged purported D2 all-masked batch is invalid,
not skipped. A partial final `[1,2]` row keeps D2 zero padding/false mask. Process
and validate all logits/targets/losses, including masked positions. Release the
last batch before testing EOS on the **returned sentinel**; `token-dataset-end?`
is a sentinel predicate, not a dataset-state query. Never count EOS as a batch.

F32 masks are outside this public loader path. Proposed native metric rows must
cover finite nonnegative f32 masks to establish weighted semantics, but no synthetic
fixture, numeric list, cast or borrowed pointer becomes public f32 ingress. A future
carrier/admission amendment is required. Mask signed zeros count as zero; positive
subnormals count as tokens even if their weighted loss rounds to zero.

## 5. Numerical contract to preserve and add

Eshkol owns validation, the batch loop, fixed forward schedule and state transitions.
Reviewed native tensor kernels own all hot f32 arithmetic; Eshkol never loops over
logit/token tensor scalars or implements exp/log/reductions in ordinary numbers.
Reuse exact merged L2 CE and L3S numerator/W/mean semantics, without changes.

L2 per row traverses vocabulary in increasing order: max, serial
`sum(expf(logit[j]-max))`, then `logf(sum)+(max-logit[target])`, all binary32.
Finite inputs are required; finite negative differences may overflow to -infinity
and expf(-infinity) becomes +0, as allowed by L2. Reject a nonfinite final CE.
Do not introduce a blanket finite-intermediate rule that changes L2 acceptance.

L3S validates both CE entries and masks, canonicalizes signed zero, computes
`W_b=fl(fl(+0+m0)+m1)`, `N_b=fl(fl(+0+fl(m0*ce0))+fl(m1*ce1))`, then
`mean_b=fl(N_b/W_b)`. Every L3S intermediate/result is finite and W_b positive.
Even an unused per-batch mean must meet the accepted L3S domain; E3 must not bypass
its validation because a later global quotient might fit. No mean-seed/backward
operation is used by evaluation.

Add separately accepted native E3-N rows, exact shapes `[1,2,256]` logits and
`[1,2]` targets/mask; rank-zero scalar inputs/outputs and exact i64 counters:

- `correct.bool` and `correct.f32`: validate all finite logits/indices/masks;
  choose argmax by scanning j=0..255, replace winner only for strict `>` (lowest
  index wins ties, including +/-0); produce `C_b=sum_in_token_order(m_i * hit_i)`
  in f32 and active-token count in i64. No softmax or approximate comparison.
- `accumulate`: given previous N/W/C and counters plus batch N_b/W_b/C_b/count,
  preflight then overwrite independent next scalars with `fl(N+N_b)`, `fl(W+W_b)`,
  `fl(C+C_b)` and checked i64 counter adds. Starts at canonical +0/zero; one call
  per batch in dataset order. No division or weighting of a batch mean.
- `finalize`: W>0, `loss=fl(N/W)`, `accuracy=fl(C/W)`, `perplexity=expf(loss)`;
  reject any nonfinite result, publish all outputs atomically. No clamp, infinity
  sentinel, f64 retry, saturation, reciprocal replacement or approximate log-limit
  test. Expf result itself determines overflow; exact-loss zero returns PPL=1.

These are proposed operation responsibilities, not existing K1 rows or a frozen
native ABI. E3-N must propose exact descriptors/schemas/error codes and versioned
explicit discovery before implementation. All new scalar inputs are finite: N>=0, W>=0 and 0<=C<=W; finalize requires
W>0. Counters are exact nonnegative i64 with checked additions. Batch correctness
rows require positive finite mask weight and reject all-zero masks consistently
with L3S. Trusted phase/authentication checks prevent fabricated accumulator state.
Separate L2/L3S/E3-N runtimes route
fixed rows; do not modify K2 or define a canonical global resolver.

Weighted accuracy is C/W, not C/tokens for f32 weights. Loss is N/W, not mean of
batch means. CE and all weights/products/sums are nonnegative; gradual underflow
and rounding to +0 are accepted. Native E3-N uses IEEE binary32 RN-even, separate
operations, no FMA/reassociation/fast-math/excess precision. Require L3S's admitted
x87/MXCSR controls (masked exceptions, FTZ/DAZ off); reject incompatible controls
without repair. Preserve validation fenv/sticky flags exactly; invocation may
accrue ordinary flags as in L3S. Numeric/domain overflow is `invalid-argument`,
unsupported FP controls `unsupported`; no partially advanced accumulator/report.

Determinism is repeated identical ordered inputs on the admitted platform/math
implementation. No cross-libm/platform bitwise claim. Regrouping three f32 sums
can change bits: `(2^24+1)+1 = 0x4b800000`, `2^24+(1+1) = 0x4b800001`.
Two-term L3S addition reversal is observationally equivalent and must not be
reported as a killed mutant. Changing model sequence boundaries can also change
causal logits mathematically. Only identical ordered batches/grouping promise bit
identity. Shard repartition is equal only when D2 packed mode yields those same
ordered input/target/mask batches. Approximate real-number parity is tested with
explicit tolerances, not mislabeled as partition invariance.

## 6. Transaction, no-grad and lifetime prerequisites

E3-CG must provide one common outer invocation guard and the address-stable fixed14
pins from the accepted [G3-T contract](g3/G3_T_PRIVATE_CONTRACT.md). Preserve all38
public M3T adapter checks, direct trusted internal nesting, model active token,
exact tie deduplication, value sentinel and parameter `plan_pins` semantics, held-bit
rollback and reverse nonallocating release. `plan_pins` is not a universal gradient
read lock; preservation also requires serialized execution, a closed role inventory,
no caller-selected callbacks/reentrancy, and no gradient-borrow/reset/plan/contribution
operation. The D2 bridge uses its existing fixed trusted synchronous consumer;
E3-D must review that closed consumer explicitly, while preserving G3-T's accepted
no-callback inventory for its own execution path unchanged.
Existing stack-only scoped guards cannot be retained across Eshkol calls.

The new sequence-length-2 (T=2) no-grad frame reuses the accepted 21 forward numerical roles without
M3 graph capture, graph leases, reverse schedule, RNG or new per-batch owner
publication. A forward-only activation workspace is allowed; retained differentiable
graphs are not. Shared contract extraction is an integration change with G3-T
review, not permission to duplicate its guard. E3 requires none of unmerged G3-N/S
kernels, T1 decode, cache, sampler, generator or G3-T runtime.

D2 staging uses three **sequential** authenticated synchronous copies into fixed
owned i64[1,2] inputs, i64[1,2] targets and bool[1,2] mask (16+16+2=34 bytes).
This is explicit same-dtype transport, not conversion or a fallback. Each original
view's begin/use/end completes before the next borrow. No nested D2 borrow or
escaped pointer. Native copies validate exact views; never manufacture an M3T/T1
public shell from D2 IDs. These fixed slots are private role storage, not copied
fixture carriers or general owner construction authority.

Proposed transaction phases:

1. Authenticate all receivers/profile/tokenizer/idle state, error precedence, FP
   controls and remaining-batch/resource bounds. Prepare detached cursor, exact
   fixed child-mode snapshot/restore plan, shared guard/pins, reusable typed scratch,
   report owner storage and bounded error transfer before any mode/cursor change.
2. Enter common busy state; set all prevalidated modes to eval with no allocation.
   Keep original parameter values, gradient presence/bytes/counts/weights and
   initializer successor/RNG untouched. Do not snapshot and rewrite gradients to
   conceal forbidden operations.
3. Each `next/validate/access/stage/forward/CE/reduce/accumulate/release` interval
   lives in one lexical `with-region`. End every borrow and release the live batch
   before leaving it. No batch shell or raw pointer escapes; only fixed frame
   state/counters survive. Keep an outer invocation region for other scratch.
4. Finish metrics in preallocated storage, then release frame/views/pins and restore
   the saved cursor and every mode before publishing. Failure unwinds the same
   resources and preserves the original structured error, with no partial report.
   Error/result transfer across the region is itself preallocated and proved.
5. A private D2 restore token authenticates the exact dataset and saved ordinal
   before mutation. Restoration after batch release assigns the original ordinal
   and invalidates temporary status/cache validity using a nonallocating, non-I/O
   tail. Generations stay monotonic; never restore the whole dataset byte image or
   resurrect stale batches. Recompute caches lazily, never treat overwritten shuffle
   slots as old contents. Public seek/close and public mode setters are excluded
   from asserted infallible rollback.

Preparation failure changes neither mode nor cursor. Only acquired tokens are
released. Post-preflight cleanup must have no recoverable failure or allocation;
impossible internal invariant failure terminates rather than claiming successful
rollback. Process termination/hardware loss is outside in-process guarantees.
Entry-live-batch rejection leaves that caller-owned batch intact. Ordinary I/O or
corrupt later shard errors restore logical cursor/mode but cannot repair external
file mutations. Saved canonical cursor bytes and next-batch semantics, not physical
cache bytes, are the preservation contract.

## 7. Proposed package and resource boundary

The minimum private artifact is a **test-only**, single source-composed/localized
aggregate with its own closed observer allowlist and exact depfile/native closure,
using genuine accepted I2/D2/M3 source owners. It installs nothing and exposes no
new public evaluator. Any proposed inherited public subset must be explicitly
inventoried; no 108-global target applies to this private artifact. It remains
separate from the optional normal package and cannot supply public-path acceptance.

For either route, the proposed source root loads the accepted M3 lineage once, then reviewed D2
semantic/dataset extensions and E3 components. A fixed byte-only tokenizer adapter
must authenticate T1 and produce the same canonical D2 identity fields; it may not
pretend T1 and T2 private names are interchangeable. D2's original T2 package keeps
its behavior. No user-selected adapter/provider/root or new public tokenizer API.
One bridge owns initialization and the E1/P1/I1/I2/T1 registry universe, with base
wrappers included once. The optional public package appends D2's eleven wrappers and E3's four wrappers.
The private milestone exposes no E3 public wrappers and claims no installed union.

Conditional target: M3's 87 boxed exports +11 D2 +4 E3 = **102 exports /108 globals**
including six E1 accessors; nine facades (M3 seven + data + evaluation). Existing
D1 eight wrappers already cover data facade's D1 surface. This is proposed counting,
not measured build evidence. No trainer, optim, persistence or capabilities facade.
Produce one localized object in one archive `libeshkol_transformer_e3.a`; preserve
predecessor packages, localize every private seam/provider accessor, and reject
linking any second registry-owning aggregate. No M3+D2 archive stitching.

Candidate native closure includes one K1/I1/I2 and identity/runtime infrastructure,
D2 native I/O/control, M3/M3T storage and common guard/frame support, N2, N3K,
A2-attention, L2, L3S and new E3-N. No A2 cache/G3-N/G3-S object. Exact source/object
paths, depfiles, allowlists, symbol/name/undefined inventories and init order must
be enumerated and independently reviewed in E3-P; the current proposal does not
invent an accepted tuple. Two-pass validate/invoke borrows remain unchanged.

Proposed admission ceilings, to prove jointly before freeze:

- 8,192 evaluated batches /16,384 active positions per call, no silent truncation.
- D2 configured semantic bound `manifest + shard + 34 + 8*min(B,M)` with B=1;
  proposed E3 cap on this sum is 16 MiB. All D2 exact/one-over rules remain.
- Additional E3 live tensor payload <=64 KiB: fixed role workspace including 2,048
  bytes logits, 34-byte staging, 8-byte CE, metric/accumulator scalar slots and
  fixed model value snapshots if needed. Parameter storage is separately 4,736
  bytes. No claim that this ceiling is measured or already attained.
- For the optional public owner design, four metric owners add exactly 16 semantic
  bytes/report; at most 8,192 E3
  invocation reservations per process (successful, failed or released), charged
  before **any** per-call cumulative identity/control allocation, including frame,
  workspace and I2/report owners. Pre-admission checks that allocate no such control
  may precede reservation. One-time reusable baseline controls are separately
  enumerated/capped; construction failures cannot escape accounting. No quota is
  recycled under a tombstone design. One-over is explicit `unsupported`, not an allocation retry. This caps
  new cumulative identity costs; E3-V must enumerate/count the exact controls and
  bytes charged even on failed reservation. Caller errors/snapshots and preexisting
  model/T1/I2 history remain separately accountable.
- Exact retained Eshkol arena, native live **and cumulative** controls/bytes, pins,
  graph owners, batch carriers and file descriptors must be measured. Proposed
  joint operational ceiling is peak RSS <=512 MiB on the supported test worker,
  reported separately from semantic payload. Exceeding any proposed ceiling
  requires root disposition; never claim it passes from a paper bound.

The minimal private witness preconstructs one fixed frame/accepted I2 output set,
then reuses it across calls. Its one-time construction/teardown live and cumulative
controls are measured separately, including failed setup; no repeated construction
or new public report shell is required. The optional report quota above is not
needed to justify this fixed-owner private numerical path. If a private implementation
adds per-call cumulative controls, it must declare and gate them before acceptance.

Frame storage is bounded/reusable without per-batch I2 identity publication;
per-call result tombstones may grow within the explicit reservation quota. Workspace
and report teardown need exact-once ownership and source/native count accounting.
Existing model/initializer destruction limits still apply; reuse one admitted
model and bounded worker processes. Neither flat per-batch arena nor RSS proves
flat process lifetime. Long loops use lexical batch regions; released native
carriers do not reclaim deliberately retained caller shells.

## 8. Prerequisite graph and acceptance sequence

E3-CG/D/N/T plus reviewed private composition are the minimum private milestone.
E3-V/P/E below describe the optional public route, not implied dispatch approval.

| Proposed workstream | Scope | Must merge first |
|---|---|---|
| E3-CG | Shared G3-T-compatible guard/fixed14 pin contract extraction and implementation; affected G3-T review | Root accepts exact sharing/disposition; merged M3/M3T/I2 |
| E3-D | Fixed byte-tokenizer identity adapter, sequential typed batch staging, closed dataset cursor restore | Root accepts exact D2 authority/ownership; merged D2/T1 |
| E3-N | Native correct/accumulate/finalize rows, independent numeric references/mutations | Root accepts exact ABI/rows; merged K1/L2/L3S |
| E3-V | Owned four-rank-zero-f32 report, accessor/release/error and quota contract | Root accepts separate standalone scalar representation; I2/P1L/E1B |
| E3-T | Graph-free sequence-length-2 frame, fixed mode restore, complete private seam/role/cleanup inventory | E3-CG, E3-D; exact native inventory accepted |
| E3-P | One canonical source/bridge tuple and package policy | E3-D/T/N/V; exact facades/exports/closures reviewed |
| E3-E | Eshkol evaluation orchestration and installed public evidence | All above merged; root accepts API/profile/numerics |
| E3-R | Independent integration review, supported exact-tree CI, merge/retest | Completed E3-E candidate; root owns acceptance |
| TR3 adapter | Authentic trainer lease checks + separately accepted A0 metric projection | E3 and trainer constructor/state machine; no E3 -> TR3 dependency |

Independent design/contract work may run in parallel; production integration must
follow actual merged seams. G3-N/S/T runtime is not an upstream shortcut. No task
may silently broaden G3-T guards, D2 public cursor format, K2 report, M3 forward,
A0 schema or I2 generic ownership. No persistent format is introduced.

## 9. Required references, failure and resource proof

These are **future required gates**, not tests executed in this design phase.
Native/private composition, restoration and retention gates apply to the minimal
slice; public facade/report/installed-boundary gates apply only if that successor
is accepted. Private tests do not satisfy or remove the eventual public-path gate.

- Independent frozen development oracle: pinned Python/PyTorch only under tests;
  build a direct mathematical fixed model from documented parameters, without
  importing production descriptors/roles/schedule. Freeze exact input/target/mask,
  logits, CE, batch N/W/C, global states and final output bits/tolerances with
  version/checksum/source/toolchain manifest. Regenerate twice byte-identically.
  Use an independent exact-binary32 rounding model for aggregation and rational
  witnesses; separate libm tolerance from f32-order bit assertions.
- At least three distinct batches with repeated/boundary IDs, unequal losses and
  valid counts, padded final row, all-correct/all-wrong and lowest-index argmax ties.
  Uniform logits give log(256), perplexity near256, and tie winner0; a nonuniform
  model case must distinguish input from next-token target. Include weighted native
  masks `(0.25,2)`, bool/0-or-1 f32 parity, subnormal positive weight, +/-0,
  masked invalid target/nonfinite logit, exact finite boundary and overflow.
- Empty, singleton corpus, entry-EOS/nonzero cursor, one-valid-token final batch,
  exact 8,192/one-over batches, exact invocation-reservation quota/one-over and exact
  jointly attainable payload/RSS limits. Packed shard-repartition equality only
  for identical emitted batches; unpacked and causal boundary changes explicitly
  need different expected answers. Demonstrate three-term grouping counterexample.
- Numerically kill wrong target offset, masked-token count/W denominator, mean of
  means, missing/double batch, second normalization, weighted-vs-unweighted accuracy,
  >= tie rule, wrong vocabulary axis, exp-before-aggregate, overflow clamp/f64 retry,
  FMA and observable sum reorder. Keep two-term reversal as a proved equivalent
  control with structural order inspection, not a fabricated kill. Mutations must
  reach compiled native/public execution and fail meaningful value/state assertions.
- Native malformed shape/dtype/device/layout/index/mask/alias/metadata/fenv cases,
  immutable inputs and byte-atomic outputs; all E3-N operations reviewed for no
  binary64/FMA/hot scalar Eshkol loops. Reused numerical primitives retain their
  accepted gradient evidence; E3 promises no differentiable output or backward path.
- Fail each allocation, copy, pin, provider admission/preflight, later shard read,
  numerical overflow and final metric preparation boundary. Inject orchestration
  errors before or after a completed provider call; accepted validated K1 invoke
  remains nonfailing, not a recoverable failure-injection promise. Preserve original error,
  exact saved cursor bytes and every mode; zero live batch/view/pin/frame/partial
  report, and stable retry. Check absent gradients and present nonzero gradients,
  parameter bytes, ties, counts/weights, initializer successor and typed RNG bits.
  No VJP/reset/contribution/graph capture/RNG operation may occur. Mutation witnesses
  must detect graph-producing eval, missed child mode, restored generation counter,
  escaped view/shell and allocating rollback. Public seek failure injection confirms
  it is excluded from the restoration tail.
- Installed source/object/AOT path must create real model/tokenizer/D1 corpus/D2
  dataset, call standalone evaluation and inspect/release metrics through proposed
  facade only. Two fresh caches with exact depfiles and output/artifact comparison;
  reverse imports, wrong arities, foreign/forged/stale/copied/busy owners, unknown
  keys, scalar lifetime, private guessed extern/import, ambient include, duplicate
  aggregate and public-name leakage negatives. Instrumented observers remain absent
  from normal package; production closure contains no Python or test-fixture carrier.
- Normal and poisoned optimized AOT compare exact retained Eshkol arena after
  1,024 and 8,192 **batches in one invocation** with no live batch. Require byte
  equality and fixed native live/cumulative workspace controls, zero graph owners,
  balanced pins/views/FDs. Separately repeat complete calls at 1,024/8,192 invocation
  reservations, release reports and measure exact cumulative result/I2 tombstones;
  reconcile growth to quota accounting instead of claiming flat memory. Report
  peak/retained RSS, payload, time and unsupported cases separately. ASan/UBSan/LSan
  with leak detection enabled required on supported lane; no disabled-leak evidence.

## 10. Independent review and limitations

Three independent Astra/high lanes reviewed actual source before drafting:
numerical semantics, composition/lifetime, and packaging/public API. They found
and informed the A0 schema restriction, scalar representation gap, graph capture,
T1/T2 identity mismatch, single D2 borrow, allocating rollback, child-mode restore,
generation monotonicity and nonassociative aggregation. All three re-read the amended proposal at SHA-256
`7c59525f07cc08cb3b91496fb7de5aa2c3575e491c27a30fb9795f0145782124`
(before this administrative disposition paragraph) and **approve posting the design
proposal**, with no remaining must-fix. This is not acceptance of implementation,
API/ABI, resource measurements or downstream dispatch.

| Review lane | Required corrections and final disposition |
|---|---|
| Numerical semantics | Added explicit N/W/C/counter domain and preserved K1 nonfailing invoke during failure injection; approved amended private-first/public-optional split |
| Composition/lifetime | Charge all cumulative optional-owner allocations, including failed setup; retain safe root-owned report identities; keep failed precommit finish abortable; distinguish fixed trusted D2 consumer from caller callbacks; approved |
| Packaging/public API | Make private existing-I2 route the minimum throughout topology/dependencies/gates; keep public counts/quota optional, aliases valid, and T=2 terminology unambiguous; approved |

Detailed source reviews and re-review dispositions are in local
`.tmp/e3-probes/{numerical,composition,packaging}-review.md` and the corresponding
`*-final-review.md` files. These independent agents used GPT-6 Astra/high as
requested. No lane approved runtime behavior or an installed public evaluator.

No runtime/ABI/format accepted; no new VERIFIED capability. Unsupported: arbitrary
profiles, public f32-mask ingress, BPE/shuffled datasets in this proposal, streaming
or unbounded evaluation, ordinary f32 numeric boxing, GPU/mixed precision/f64 retry,
compiler autodiff, public trainer trajectory, checkpoint/resume, generation,
performance or flat total-process memory. Proposed resource ceilings and owner
mechanisms remain implementation gates. This bounded phase is complete only as
proposal/reachability evidence; E3 itself remains active and dependency-gated.
