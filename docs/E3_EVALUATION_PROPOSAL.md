# E3 evaluation: proposed composition and reachability

Status: **private-first architecture accepted with conditions; exact design remains
proposed** for [issue #99](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/99).
The binding [root disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5771430755)
selects the bounded private route, not an ABI/runtime/public freeze. Integration
owns exact-contract acceptance, implementation dispatch, review and merge.

Original source/probe base was merged `fe2493168e72cd7bf81d810c9c172c40aacfee61`,
tree `902d4645e13c133a811f1c1cf72ffb799976a58a`. This repair merges current main
`f06ea0f35aec73d1a22bf073c861eedb9774ae5b`, including PR #98 L3S acceptance.
All accepted L3S and G3 histories are preserved. Original probes are not relabeled
as fresh evidence on this documentation revision. Separate component acceptance
still does not establish a joint installed evaluator.

## 1. Selected architecture and remaining decisions

1. **Private existing-I2 outputs.** A closed private evaluator uses four preowned
   rank-zero CPU-f32 I2 scalar outputs and two exact i64 counters. No new public
   operation, scalar encoding, report shell/registry/release/quota or trainer is
   part of this slice. A0 names/arities and exact four-key trainer map stay unchanged.
2. **Bool-only numerical minimum.** The three proposed new rows are `correct.bool`,
   `accumulate`, and `finalize`. Defer `correct.f32` and weighted-mask accuracy
   execution/tests. Preserve merged L2/L3S behavior; existing weighted L3S reference
   semantics provide future context, not an extra required E3 row.
3. **One shared guard owner.** [E3-CG](E3_SHARED_CALL_CONTRACT.md), issue #101, owns the exact shared
   guard/fixed14 proposal in task `01a0c789-bd3f-7b21-b865-8b9365c7a282`.
   Original G3-T task `01a0bd6a-1e15-7662-a245-0eaa201426d3` retains affected
   G3-T review. This E3 task owns its consumption requirements,
   sequence-length-2 (T=2) no-grad frame and evaluation transaction. No duplicated
   guard, strengthened I2 lock or dependency on unmerged G3-N/S/T runtime.
4. **One private source composition.** Use M3 lineage, an authenticated fixed-byte
   D2 identity adapter and three sequential unchanged-type copies. Closed identity,
   restore, native and package inventories below are proposals pending exact review;
   never join completed registry-owning archives or assume T1/T2 interchangeability.
5. **Transactional publication.** Restore logical cursor and every affected child
   mode before one prevalidated nonfailing commit publishes all four scalars and
   both counters. Every recoverable failure leaves all six caller outputs unchanged.
6. **One bounded design package and private milestone.** Consolidate the E3 frame,
   D2/P1 restoration, bool numerical rows, source tuple and tests here. The same E3
   task owns private orchestration/composition plus its independent integration
   review after required exact contracts and implementations are accepted/merged.
   This milestone cannot close E3 roadmap acceptance or claim a public evaluator.

This selects architecture only. Primitive descriptors, private/native call authority,
restoration token layout, exact source/native manifests and proposed resource
ceilings still require root acceptance before implementation against them.

### Selected route and deferred alternatives

| Route | Metric representation/reachability | Disposition |
|---|---|---|
| Private E3 | Genuine preowned I2 f32[] outputs, exact i64 counters; trusted synchronous inspection and existing destruction | Selected minimum; exact inventory still proposed |
| Standalone diagnostic facade/report | Prior proposal's four operations, six-key handle report, registry/quota/release and 108-global/102-export/nine-facade installation | Deferred in full; not a prerequisite, current test gate or dispatched workstream |
| TR3/A0 adapter | Authentic trainer and unchanged four-key map, requiring an accepted true-f32 scalar projection | Downstream; E3 never depends on TR3 or invents trainer construction |

The optional public names from the earlier proposal were `diagnostic-evaluate!/2`,
`evaluation-metrics-ref/2`, `evaluation-metrics-f32-bits/2` and
`evaluation-metrics-release!/1`. They are retained here only to identify the deferred
choice; none is approved, implemented, installed or required by this private design.

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

## 3. Private metric destinations and publication

The sole proposed orchestration boundary is
`e3-evaluate-into!/3 (model,dataset,trusted-frame)`, unavailable to public callers.
The trusted frame binds four independent **existing I2** rank-zero f32 owners and
two preallocated exact-i64 control cells. It does not create a scalar ABI, report
receiver, accessor family, serialized format or identity registry for results.

| Destination | Physical type | Meaning |
|---|---|---|
| loss | I2 f32[] | ordered total CE numerator / total bool weight |
| mask-weight | I2 f32[] | ordered bool weight, exactly equal to tokens within the cap |
| perplexity | I2 f32[] | finite expf(loss) |
| token-accuracy | I2 f32[] | correct selected tokens / selected tokens |
| tokens | exact i64 control cell | number of true mask positions |
| batches | exact i64 control cell | completed batches, excluding EOS |

The private caller creates/owns/destroys the four I2 objects using accepted I2
operations and keeps them live/exclusively available for the entire evaluation.
Destination identity is fixed during frame setup; every call re-admits those same
live owners. All four data spans are mutually disjoint, distinct from every scratch,
input and control span; the two counter cells are exact frame-owned control slots.
No per-batch publication, I2 construction, ordinary Eshkol f64 boxing or result
shell is allowed. Existing I2 exact-bit inspection/scoped borrows remain the only
numeric inspection mechanism; ordinary references grant no new authority.

The reusable frame owns separate staged results: four private f32 words plus two
i64 values, independent of caller destinations. Native `finalize` writes staging
only. After all batch borrows/carriers are ended, cursor and **all** modes restored,
and acquired frame/pin resources validated/released (model/outer exclusion remains
held), a closed publication helper:

1. authenticates the frame and exact destinations, checks readiness and unchanged
   ownership, then opens all four destination I2 scopes within this one synchronous
   native call; no scope survives its return;
2. completes every type/shape/liveness/alias/control/cleanup check before any write;
3. copies exactly four f32 words and two i64 values, marks output ready, and ends
   the prevalidated scopes through a nonallocating, nonfailing tail.

No arithmetic, allocation, provider dispatch, callback or recoverable error occurs
from the first output write onward. Failure before that write preserves all six
outputs and leaves the private frame clean/reusable after abort. A failing numerical
finalization remains abortable; successful publication commits once and excludes
abort. The impossible post-preflight invariant-defect policy is termination, never
a misleading recoverable partial-result error. Native I2 scope compatibility with
this helper is an explicit implementation proof obligation, not an existing API.

No output is published early because a caller cannot observe intermediate staging.
The closed helper is not a public multi-owner transaction API. Future TR3 remains
responsible for its genuine trainer lease and distinct training-dataset checks and
for any separately accepted A0 scalar projection. The six private destinations do
not silently extend A0's four-key result map.

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
exact restoration and all six output destinations unchanged. Every genuine D2 batch has at least one true mask.
All-masked native input rejects; a forged purported D2 all-masked batch is invalid,
not skipped. A partial final `[1,2]` row keeps D2 zero padding/false mask. Process
and validate all logits/targets/losses, including masked positions. Release the
last batch before testing EOS on the **returned sentinel**; `token-dataset-end?`
is a sentinel predicate, not a dataset-state query. Never count EOS as a batch.

F32 masks are outside the selected private loader and numerical slice.
`correct.f32`, weighted accuracy execution and associated weighted-E3 fixtures/mutants
are deferred. For future context only, merged L3S already accepts finite nonnegative
f32 masks, signed zeros and positive subnormals; this does not create E3 mask ingress
or a required accuracy row. No synthetic fixture, cast or list grants carrier authority.

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

- `correct.bool` only: validate all finite logits/indices/canonical bool masks;
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

For the selected bool slice, accuracy is C/W and W==tokens exactly; replacing W
by an exact-f32 tokens value is observationally equivalent and is not a mutation kill.
Loss is N/W, not mean of batch means. Future weighted-mask accuracy would use C/W,
not C/tokens, but that row and its execution tests are deferred. CE and all weights/products/sums are nonnegative; gradual underflow
and rounding to +0 are accepted. Native E3-N uses IEEE binary32 RN-even, separate
operations, no FMA/reassociation/fast-math/excess precision. Require L3S's admitted
x87/MXCSR controls (masked exceptions, FTZ/DAZ off); reject incompatible controls
without repair. Preserve validation fenv/sticky flags exactly; invocation may
accrue ordinary flags as in L3S. Numeric/domain overflow is `invalid-argument`,
unsupported FP controls `unsupported`; no partially advanced accumulator or caller destinations.

Determinism is repeated identical ordered inputs on the admitted platform/math
implementation. No cross-libm/platform bitwise claim. Regrouping three f32 sums
can change bits: `(2^24+1)+1 = 0x4b800000`, `2^24+(1+1) = 0x4b800001`.
Two-term L3S addition reversal is observationally equivalent and must not be
reported as a killed mutant. Changing model sequence boundaries can also change
causal logits mathematically. Only identical ordered batches/grouping promise bit
identity. Shard repartition is equal only when D2 packed mode yields those same
ordered input/target/mask batches. Approximate real-number parity is tested with
explicit tolerances, not mislabeled as partition invariance.

### 5.1 Proposed exact bool-row schemas

This is the numerical portion of **E3-DESIGN**, not a separate dispatched task.
Proposed fixed K1 request for all three rows: `[1,2,256]`, compute f32, CPU;
scalar operands below remain rank zero. Proposed local operation suffixes are
`correct.bool`, `accumulate`, `finalize`; final capability/provider strings,
version and error-code assignments remain an explicit root review item.

| Row | Ordered inputs | Ordered outputs |
|---|---|---|
| correct.bool (3/2) | logits f32[1,2,256], targets i64[1,2], mask bool[1,2] | correct-weight f32[], active i64[] |
| accumulate (9/5) | N f32[], W f32[], C f32[], tokens i64[], batches i64[], N_b f32[], W_b f32[], C_b f32[], active i64[] | next_N f32[], next_W f32[], next_C f32[], next_tokens i64[], next_batches i64[] |
| finalize (3/4) | N f32[], W f32[], C f32[] | loss f32[], mask_weight f32[], perplexity f32[], token_accuracy f32[] |

All views are dense/zero-offset, CPU, naturally aligned and exact-span; rank-zero
shape pointer is null. Sizes are 2,048/16/2 bytes for correctness inputs and 4/8
bytes for f32/i64 scalars. All inputs are pairwise disjoint; outputs are mutually
disjoint and disjoint from inputs and live metadata. Use fixed disjoint ping-pong
accumulator slots, never in-place accumulation. Error storage retains K1's disjoint
trusted-control precondition. Borrow through both validate/invoke, retain no pointer,
allocate nothing during dispatch, validate every prospective output before writing.

Correctness validates all 512 logits and both targets even when masked, and both
mask bytes are exactly 0/1 with at least one true. Positions 0 then 1; initialize
winner 0, scan j=1..255 replacing only on strict greater. C_b is an exact integer
in `[0,active]`; active is 1 or 2. Explicit bool-to-f32 conversion is exact.

Accumulator admission: all six floating inputs finite/nonnegative; counters exact
nonnegative i64, `batches<=8192`, `batches<=tokens<=2*batches`, `tokens<=16384`.
W equals exact-f32(tokens), C is integral in `[0,W]`; batches=0 also requires
N=0. Batch active is 1 or 2, W_b equals exact-f32(active), and C_b is integral in
`[0,W_b]`. Bounds are checked before conversion/products/addition. Checked next
counts must fit both i64 and the profile cap; then compute next N, W, C in that
order with separate f32 additions. W/C remain exact within this bool bound.

Finalize validates finite N>=0, integral W in `[1,16384]`, integral C in `[0,W]`;
computes loss, accuracy, expf(loss), then stages outputs in the table order.
All results finite, `0<=accuracy<=1`, perplexity>=1. Its three numerical inputs
cannot authenticate counters or earlier execution history: only the closed frame's
phase/transcript proves provenance. Do not advertise a scalar-domain check as
proof that N_b came from L3S or that batches were traversed in order.

Proposed errors preserve K1 generic-validation precedence, then schema/alias,
FP controls, basic domain, bounded-count admission, cross-field consistency and
prospective arithmetic. Schema failures use existing shape/dtype/device/layout
categories; target range is shape-mismatch as in L2. Bad bool, nonfinite/negative
values, fractional C/W, inconsistent counts, zero mask/final W and arithmetic
nonfiniteness use invalid-argument/provider-rejected. Structurally valid profile
one-over and unsupported FP controls use unsupported/provider-rejected. Empty
traversal is Eshkol invalid-state before finalize. Exact table/code/precedence and
metadata span exclusions must be reviewed before ABI acceptance.

An independent bool-compatible order witness uses three one-active batches with
N_b `[64,2^-18,2^-18]`: serial total bits `0x42800000`, reassociated/retained-f64
total `0x42800001`; dividing by 3 gives loss `0x41aaaaab` versus `0x41aaaaac`,
with finite perplexity. Unequal bool weights `(N_b,W_b)=(1,1),(6,2)` distinguish
7/3 from the incorrect mean-of-means 2. These are native-row reference states;
no claim that the public fixed model emits exactly these CE words.

## 6. Transaction, no-grad and lifetime prerequisites

The [E3-CG exact contract](E3_SHARED_CALL_CONTRACT.md) proposes canonical
`native/m3_call_adapters.esk`, `src/eshkol_transformer/m3_call_pins.h` and
`src/eshkol_transformer/m3_call_f32_integration.c`, preserving the existing
`et_g3t_model_pins_internal` type and three helper signatures. E3 authenticates
its own frame/model before deriving embedded pins and retains model/outer tokens
after pin-end until its restoration/publication tail. The common layer has no
E3 registration or token API. Exact source design remains under review.

The canonical shared-guard owner must provide one common outer invocation guard
and the address-stable fixed14
pins from the accepted [G3-T contract](g3/G3_T_PRIVATE_CONTRACT.md). Preserve all38
public M3T adapter checks, direct trusted internal nesting, model active token,
exact tie deduplication, value sentinel and parameter `plan_pins` semantics, held-bit
rollback and reverse nonallocating release. `plan_pins` is not a universal gradient
read lock; preservation also requires serialized execution, a closed role inventory,
no caller-selected callbacks/reentrancy, and no gradient-borrow/reset/plan/contribution
operation. The D2 bridge uses its existing fixed trusted synchronous consumer;
the E3 D2 design must review that closed consumer explicitly, preserving G3-T's accepted
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

1. Enter the sole outer `m3-call` guard once; reject reentry before mutation.
   Under that exclusion authenticate receivers/profile/tokenizer/idle state,
   error precedence, FP controls and remaining-batch/resource bounds. Prepare
   cursor and fixed child-mode restore records, verify reusable typed scratch,
   staged scalar/counter storage and bounded error transfer before mode/cursor writes.
2. Require idle consumer/model tokens, acquire the canonical fixed14 pins, then
   publish exact matching model/frame/native active tokens. Set all prevalidated
   modes to eval with no allocation. Keep original parameter values, gradient
   presence/bytes/counts/weights and initializer successor/RNG untouched. Do not
   acquire a second outer guard or snapshot/rewrite gradients to conceal writes.
3. Each `next/validate/access/stage/forward/CE/reduce/accumulate/release` interval
   lives in one lexical `with-region`. End every borrow and release the live batch
   before leaving it. No batch shell or raw pointer escapes; only fixed frame
   state/counters survive. Keep an outer invocation region for other scratch. Caller destinations remain
   outside that region and unchanged until the final commit in section 3.
4. Finish metrics in preallocated staging, then release views/pins and restore the
   saved cursor and every mode. Retain acquired model/outer exclusion through
   section 3's single nonfailing six-output commit; clear it only in the final
   validated tail. Commit is possible only after restoration.
   Failure unwinds the same resources, preserves the original structured error and
   leaves every caller scalar/counter byte unchanged. Error transfer out of the
   invocation region is preallocated/proved; no result shell escapes that region.
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

### 6.1 Shared-owner coordination and bounded G3-T wording

The original G3-T owner and its independent Astra/high reviewer supplied the split;
root accepted the [extraction direction](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5771464954).
The following are the four proposed clarifications to the accepted G3-T document,
recorded here for its owner's exact-wording review. That document is not rewritten
by this E3 proposal; its 29 calls/95 bindings and 93-global/87-export/seven-facade
contract, slots/kinds, cache/RNG/decoder publication and N/S implementation hold stay
unchanged. No sampler work is included.

1. **Shared definitions:** “Existing `m3-call-state`/`m3-call` remain the sole outer
   guard. One common M3-lineage source owns all 38 checked M3T adapter definitions;
   retain their exact suffixes/arities/C targets and the eight unchanged M3 targets.
   Trusted original M3T functions remain direct internal calls and are never wrapped.”
2. **Pin embedding:** “The one fixed14 pin object is embedded in an authenticated,
   address-stable consumer context: the existing G3 generator context for G3, or
   a separately authenticated reusable E3 frame. Closed consumer admission validates
   M3T ownership, rejects occupied model slot10/native owner.active, acquires pins,
   then publishes exact matching consumer tokens. No generic owner-kind, arbitrary
   token pointer, parameter-index or callback acquisition authority is introduced.”
3. **Source closure:** “One common private header/implementation in the single f32
   replacement translation unit owns fixed14 begin/check/end. Preserve accepted
   `et_g3t_*` private spellings through common definitions or source-local aliases;
   never compile a second f32 registry/pin implementation. Shared sources import
   no G3 registry, cache, sampler, decoder, provider or initializer.”
4. **Restoration:** “E3 owns mode/cursor restoration and staged-result publication
   inside shared exclusion;
   G3 remains eval-only. Common pin-end clears only its held pins, never the outer
   guard or consumer restoration state. Keep model/aggregate exclusion through
   restored cursor/all modes and atomic four-scalar/two-counter publication, then
   clear only matching acquired tokens through the validated nonfailing tail.”

The shared layer has no `accept-train` flag or restoration callback. It preserves
exact14 ties/identities, 4,736 bytes, self/view sentinels, count0-to1 admission,
held-mask partial rollback and reverse drain. E3's own native/source inventory is
not implicitly part of G3's 29/95 inventory. The common exact source/contract must
be reviewed and merged before either consumer implements against it.

### 6.2 Fixed forward frame and private call responsibilities

Current M3T executors bind saved parameter tensors and allocate reverse storage.
Suppressing M3 graph capture is therefore insufficient. The proposed E3 frame
binds parameter operands directly to the canonical held fixed14 views, owns only
forward scratch/readiness, and keeps the Eshkol 21-role schedule:

```text
0 token embedding; 1 position embedding; 2 embedding residual;
3 block-input LayerNorm; 4 Q linear; 5 K linear; 6 V linear;
7 Q heads split; 8 K heads split; 9 V heads split; 10 causal attention;
11 heads merge; 12 attention-output linear; 13 attention residual;
14 ffn-input LayerNorm; 15 ffn-up linear; 16 GELU; 17 ffn-down linear;
18 ffn residual; 19 final LayerNorm; 20 tied-head linear.
```

Each destination is ready once in this order. Eshkol invokes each fixed role; no
native full-model loop, provider-selected operation or arbitrary tensor selector.
Numerical rows are unchanged merged N3K/N2/A2-attention, then L2/L3S and the three
proposed bool metric rows. No backward/gradient/reset/contribution, RNG, graph,
A2-cache or G3 runtime call exists in the closed E3 transcript.

Proposed E3-private responsibilities/arities (not an accepted C ABI):

| Seam | Arguments and closed authority |
|---|---|
| `e3-frame-bind!/2` | authentic model, exact preowned destination record; one-time frame/staging setup |
| `e3-evaluate-into!/3` | model, dataset, that exact frame; owns Eshkol loop/transaction |
| `e3-stage-batch!/2` | frame, current batch; only the three fixed sequential copies |
| `e3-forward-role!/2` | frame, next fixed role ordinal 0..20; no caller-selected provider |
| `e3-ce!/1`, `e3-reduce!/1`, `e3-correct!/1`, `e3-accumulate!/1` | frame; four Eshkol-ordered fixed native dispatch boundaries |
| `e3-finalize!/1` | frame; computes separate result staging only |
| `e3-cleanup-preflight!/1`, `e3-abort!/1` | frame; checks all acquired resources and drains only acquired state |
| `e3-publish!/1` | restored, cleanup-preflighted frame; single six-output native commit |
| `e3-counter-ref/2` | idle committed frame, fixed selector tokens/batches; return immediate exact i64 |
| `e3-frame-unbind!/1` | idle frame; one-time teardown, caller keeps its four I2 destinations |

The result-control record holds four exact I2 owner identities plus native stable
`int64_t counters[2]`; it is not an Eshkol-vector memory cast or public report.
Separate stage storage is four f32 words and two i64 cells. Frame binding is the
only construction authority; copied frame/record/tag fields reject. The concrete
native struct, signatures/status/error spans, source names and readiness-state
encoding remain acceptance items; the table is a complete responsibility inventory,
not permission to invent a generic carrier or reusable external owner API.

### 6.3 All 17 modes and P1 lexical authority

Source-derived canonical mode order is:

```text
0 root; 1 blocks; 2 blocks/0; 3 blocks/0/attention;
4 blocks/0/attention/key; 5 blocks/0/attention/output;
6 blocks/0/attention/query; 7 blocks/0/attention/value;
8 blocks/0/ffn; 9 blocks/0/ffn/down; 10 blocks/0/ffn/up;
11 blocks/0/norm1; 12 blocks/0/norm2; 13 head; 14 norm_final;
15 position_embedding; 16 token_embedding.
```

Authenticate every distinct module and parent/child binding, not just root mode.
The tied parameter does not merge head and token_embedding module nodes. Raw P1
module vectors have 13 fields: mode is internal slot1, children slot4, parent slot5,
finalized slot7, subtree node-count slot10. Root count is exactly17. All original
train/eval symbols, including mixed child modes, must be saved and restored.

These raw fields and `raw-for-shell`/`finalization-plan` are lexical inside
`p1-trusted-surface`. Existing trusted slot26 reads one authenticated shell's mode;
it is not a raw getter/setter. Public mode setters allocate their plan. Therefore
this package proposes **new narrow P1-owned lexical authority**, not opaque-shell
indexing or an already callable restore seam, preserving current slots0..63:

| Proposed seam | Arity | Transition |
|---|---:|---|
| `p1-e3-mode-bind!` | 2 | authentic fixed model + exact frame-owned record; validate/bind all17 raw identities at setup |
| `p1-e3-mode-prepare!` | 1 | idle record; validate identity/topology, snapshot all17 symbols; prepared, no mode change |
| `p1-e3-mode-enter!` | 1 | prepared record; preflight all destinations, set all17 eval; nonallocating tail |
| `p1-e3-mode-restore!` | 1 | closed cleanup; restore exact saved17 if entered, otherwise no mode writes; clear snapshot/idle |
| `p1-e3-mode-unbind!` | 1 | idle record only; clear bound roots during teardown |

Record fields: exact private authority/frame identity, idle/prepared/eval phase,
root shell/raw root, fixed raw-node vector[17], saved-mode vector[17]. No arbitrary
raw-vector access. Root lifetime/promotion and exact append-only slot or conditional
source integration must be accepted with P1/root; they are not supplied by the
current 64-slot table. Failed preflight changes no modes; post-preflight restore
has no allocation/recoverable branch. Native code never interprets P1 vector layout.

### 6.4 Closed D2 identity, staging and restore record

Dataset admission first validates compiled constructor identity using D2's native
factory (kind1), then its private query; state[0] is the exact private tag,
state[1] the exact shell, state[2] open. The fixed byte adapter must authenticate a
real same-root T1 tokenizer, obtain its canonical fingerprint/V through existing
T1 private authority, require V256 and empty specials/prefix/suffix, and bind the
same D1 metadata identity that D2 checks. It is not an alias equating T1/T2 registries.

Proposed adapter is `e3-d2-byte-identity/1 (tokenizer) -> private fingerprint/V pair`.
Its source binding replaces only D2's two hardwired T2 identity reads inside this
private tuple, by a fixed trusted source binding (never runtime callback injection).
The original D2/T2 package retains its existing behavior. Exact lexical factoring
must be accepted; merely appending a competing definition is not valid composition.

Dataset state has19 slots. Relevant ones:3 normalized config,4 D1 metadata,
5 manifest,6 core cursor,8 last batch generation,10 batch status,15 borrow marker,
16 shuffle-window index,17 row count M,18 semantic bound. Slots12..14 are currently
unused #f, not a hidden shard-byte cache; shard I/O bytes are lexical-region scoped.
Core cursor has17 slots with immutable identity/config0..14, next ordinal15,
logical-only marker16. Native dataset control independently owns monotonic batch
and lease generations, current batch and bounded shuffle slots.

The reusable restore record's proposed fields are exact authority/frame identity,
idle/prepared phase, dataset shell, dataset state, core state, saved ordinal i64,
M i64, config identity, metadata identity and manifest identity. Snapshot only after
all profile/idle/identity checks succeed. Optional detached canonical cursor bytes
are evidence storage, not authority and never reparsed during cleanup.

| Proposed seam | Arity | Responsibility |
|---|---:|---|
| `e3-d2-prepare!` | 2 | dataset + exact frame record; authenticate/bind/snapshot; failed admission leaves record idle |
| `e3-d2-restore-preflight!` | 1 | exact binding, open/same identities/M, no live batch/borrow; native idle check |
| `e3-d2-restore-commit!` | 1 | after preflight, core[15]=saved ordinal; state[10]=none; state[16]=#f; clear record/idle |
| `e3-d2-cancel-prepare!` | 1 | clear only acquired prepared bindings when no cursor mutation occurred |

A new closed native dataset-idle preflight must verify enrolled exact owner and no
current batch/lease; current D2 has no standalone exported equivalent. Do not reuse
live-batch release-preflight as if it admitted an idle dataset. Exact signature/error
mapping is still a contract item. No restore writes state[8] or native batch/lease
generations. Unshuffled admission makes shuffle contents irrelevant; marker
invalidation still matches public seek. No close, I/O, cursor parse or new allocation
in the restore tail. Clear record roots so fixed frame does not retain old datasets.

Staging uses existing single-view D2 begin/use/end three times, with fixed trusted
consumer code for inputs, targets and mask, in that order. Copy exact16/16/2 bytes
to fixed private typed slots; validate exact view/owner/generation and disjointness.
No view/consumer callback escapes or reenters. Complete each borrow before the next.
A stage failure clears readiness; the transaction releases its own batch and aborts.
Public/live entry batches rejected at admission remain untouched.

### 6.5 Acquired-state cleanup order

Track separate acquisition flags for outer guard, model token,14 pin held bits,
mode-prepared/entered, D2-record prepared, batch owned, active synchronous view,
forward readiness and staged result. Flag publication follows successful acquisition.

On failure preserve first error; end the currently owned D2 view, release the exact
owned batch before its region ends, preflight remaining resources, drain forward
views/pins in reverse acquisition order, restore D2 cursor/status and all17 modes,
clear per-invocation roots/readiness, then matching model token and outer guard.
Do not release another caller's live batch or clear an unacquired token. Shared
pin-end does not clear model/outer exclusion. Error rethrow/promotion capacity is
prepared before mutation; no new error replaces the original in a valid tail.

On success the same restoration happens while model/outer exclusion stays held.
Keep finalized staging and its publication-readiness flags live until the six-output
commit; the failure-path clearing of staged result/readiness does not run on success
before publication. Then section3's six-output commit, matched model-token cleanup and outer-guard
exit run without a recoverable tail. Counter readout is an immediate i64, not
fallible late boxing. Any scope/sentinel/end check that could reject is performed
before publication, including output scopes inside the one native commit call.
Proof covers failure at destination1/2/3/4 admission with all six entry values
unchanged, and confirms no ordinary allocating I2 leases/copy-plan tombstones are
introduced per call. Finalized staging itself is never caller-visible publication.

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
One private bridge owns initialization and the E1/P1/I1/I2/T1 identity universe,
with base wrappers included once. It exposes only an exact test-consumer/observer
allowlist under a private artifact; no installed evaluation facade or public
export count is selected. The earlier 108/102/nine-facade package is deferred.
Preserve predecessor packages, localize private provider/accessor/helper symbols,
and reject any second registry-owning aggregate. No M3+D2 archive stitching.

Candidate native closure includes one K1/I1/I2 and identity/runtime infrastructure,
D2 native I/O/control, M3/M3T storage and common guard/frame support, N2, N3K,
A2-attention, L2, L3S and new E3-N. No A2 cache/G3-N/G3-S object. Exact source/object
paths, depfiles, allowlists, symbol/name/undefined inventories and init order must
be enumerated and independently reviewed in this consolidated design; the current proposal does not
invent an accepted tuple. Two-pass validate/invoke borrows remain unchanged.

### 7.1 Concrete input-source/native inventory for exact review

Conservative inherited Eshkol input closure is exactly the 18 paths in merged
`native/m3_package_source_closure.txt`, once, plus
`internal/d2/lib/d2_semantic_core.esk` and
`internal/d2/lib/d2_dataset.esk` after the accepted fixed identity factoring.
Inherit M3 declarations without invoking its graph/reverse entry points; trimming
that source closure is a separate change, not a shortcut around reviewed reset
and native replacement recipes. No D2/T2 aggregate root or bridge is appended.

| Existing native translation unit/object | Composition rule |
|---|---|
| native/data_io.c; native/checkpoint_io.c; native/kernel_abi.c | one of each |
| src/eshkol_transformer/m3_i64_integration.c | includes i64_tensor.c; no additional i64_tensor.o |
| native/t1_i64_shell.c | one exact T1 shell authority |
| src/eshkol_transformer/m3t_f32_integration.c | includes f32_tensor.c; canonical common-pin replacement TU includes this once, no additional f32_tensor.o |
| src/eshkol_transformer/m3_model.c | includes m3t_transport.c with reviewed replacements; no separate m3t_transport.o |
| native/d2_native.c | existing D2 carrier/I/O authority, proposed narrow idle-preflight addition only after review |
| native/indexed_cross_entropy.c; native/l3s_masked_objective_provider.c | existing L2/L3S provider recipes, localized accessors |
| n2/n2_primitives_provider.o; n3k/n3k_primitives_provider.o; a2/a2_attention_provider.o | inherited fixed provider objects; no A2-cache or G3 objects |

Compile one E1 consumer bridge and one private package bridge with the existing
M3→M3T→I2→T1→X1/P1/D1 wrapper lineage once. P1 bridge already includes
p1_identity.c; never add a second identity object. Private source calls to D2 do
not require its eleven installed wrappers or data facade. Trusted include roots
are the inherited P1/C1/T1/src roots plus D2 internal source, not T2 include roots.

New source roles to name/accept in the **same E3-DESIGN package**: private root and
closed entry/observers; canonical shared adapters/pins supplied by their owner;
P1 lexical mode extension; fixed D2 identity/restore/staging extension; forward-only
frame/result commit bridge; three bool metric rows; Eshkol orchestration. Each
concrete filename/TU/signature/slot and real preprocessor/compiler dependency must
be enumerated before exact-contract acceptance. Existing input lists here are
source-grounded; new file names, counts and generated closure are deliberately
not presented as measured or frozen. No copied development fixture supplies a
production owner, and no generic builder tuple is implicitly admitted.

One private root/object plus fixed E1/bridge/native inputs is partial-linked and
localized once. Its exact test-entry/observer/global/undefined/name/member manifests
must be separately reviewed, with consumers compiled after localization. No public
count from a predecessor authorizes extra globals. The builder must fail closed
on any source/object outside this fixed tuple; unfamiliar tuples are not permitted
by the current builder and require explicit reviewed changes.

Proposed admission ceilings, to prove jointly before freeze:

- 8,192 evaluated batches /16,384 active positions per call, no silent truncation.
- D2 configured semantic bound `manifest + shard + 34 + 8*min(B,M)` with B=1;
  proposed E3 cap on this sum is 16 MiB. All D2 exact/one-over rules remain.
- Additional E3 live tensor payload <=64 KiB: fixed role workspace including 2,048
  bytes logits, 34-byte staging, 8-byte CE and metric/accumulator scalar slots.
  Forward parameter operands use held views; parameter storage is separately 4,736
  bytes. No claim that this ceiling is measured or already attained.
- Exact retained Eshkol arena, native live **and cumulative** controls/bytes, pins,
  graph owners, batch carriers and file descriptors must be measured. Proposed
  joint operational ceiling is peak RSS <=512 MiB on the supported test worker,
  reported separately from semantic payload. Exceeding any proposed ceiling
  requires root disposition; never claim it passes from a paper bound.

The private witness preconstructs one fixed frame/accepted I2 output set and reuses
it across calls. Measure one-time construction/teardown live and cumulative controls
including every failed setup, separately from repeated calls. The private route has
no report registry, reservation quota or result tombstones; existing I2/model
controls may still be cumulative and must be counted, not hidden. No per-batch or
per-call owner publication is permitted in the fixed-frame success path. Error
objects remain region-owned unless deliberately retained and accounted by caller.

Frame/staging storage is bounded/reusable; teardown requires exact-once ownership.
Existing model/initializer destruction limits still apply: reuse one admitted model
and bounded worker processes. Neither flat per-batch arena nor RSS proves flat
process lifetime. Deliberately retained D2 shells remain caller-owned allocations.

## 8. One consolidated design package and private milestones

These are ownership/milestone labels, **not eight independent dispatch requests**.

| Milestone | Owner and deliverable | Gate |
|---|---|---|
| Shared guard extraction | [E3-CG #101](E3_SHARED_CALL_CONTRACT.md) owns the canonical source/guard/fixed14 inventory; original G3-T owner reviews affected contract; E3 supplies consumer requirements | Exact contract reviewed/merged before either consumer implements against it |
| E3-DESIGN | This E3 task consolidates bool metric descriptors, fixed frame/21 roles, P1 all-mode token, D2 identity/staging/restore, six-output commit and private package manifest proposal | Root approves exact inventories; architecture direction alone is insufficient |
| Required seam implementations | Root chooses bounded implementation grouping after exact design; shared guard/frame, P1/D2 restoration/transport and three bool metric operations | Independent substantive review, supported CI and merge for each actual prerequisite; no invented/unmerged upstream API |
| **E3-PRIVATE** | **This E3 task owns Eshkol private orchestration and real-source composition**, reusing preowned I2 outputs and a fixed frame | Required shared guard/frame, P1/D2 and numerical contracts/implementations accepted and merged; root authorizes implementation |
| **E3-PRIVATE-R** | **This E3 task supplies the handoff; root owns independent integration review, supported exact-tree CI, merge/retest** | E3-PRIVATE restoration/publication/numerical/resource/closure evidence; no optional public owner/facade dependency |
| Public evaluation / TR3 | Deferred separately selected successor, including authentic trainer and any A0 scalar projection | Private milestone alone cannot complete E3; no public workstream currently dispatched |

G3-N/S/T runtime is not an upstream shortcut. Shared guard and E3 design can proceed
in parallel with one canonical guard owner. No silent G3-T guard, D2 cursor-format,
K2 report, M3 forward, A0 schema or I2 generic-ownership change. No new file format.

## 9. Required references, failure and resource proof

These are **future required gates**, not tests executed in this design phase.
All current gates below use the selected bool-only private route. Optional public
facade/report and weighted-mask accuracy gates are deferred, not prerequisite tests.
Private evidence does not satisfy or remove a separately accepted future public path.

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
  model case must distinguish input from next-token target. Include all three
  nonempty bool patterns/partial final row, +/-0 logits, subnormal finite logits/CE,
  masked invalid target/nonfinite logit, exact finite boundary and overflow.
  Weighted L3S `(0.25,2)` and subnormal-weight references remain existing/future
  context only; no new weighted E3 row or weighted-accuracy gate is required.
- Empty, singleton corpus, entry-EOS/nonzero cursor, one-valid-token final batch,
  exact 8,192/one-over batches and exact jointly attainable payload/RSS limits. Packed shard-repartition equality only
  for identical emitted batches; unpacked and causal boundary changes explicitly
  need different expected answers. Demonstrate three-term grouping counterexample.
- Numerically kill wrong target offset, physical-capacity denominator, mean of
  means (unequal active counts), missing/double batch, second normalization,
  >= tie rule, wrong vocabulary axis, counting padded positions, exp-before-aggregate,
  overflow clamp/f64 retry and observable sum reorder. Two-term reversal, W versus
  exact-f32 token-count division, and contraction of canonical 0/1 weighting can be
  equivalent in this bool scope: use equivalence controls/structural no-FMA checks,
  not fabricated numerical kills. Mutations reach compiled native/private execution
  and fail meaningful value/state assertions. Weighted-accuracy mutants are deferred.
- Native malformed shape/dtype/device/layout/index/mask/alias/metadata/fenv cases,
  immutable inputs and byte-atomic outputs; all E3-N operations reviewed for no
  binary64/FMA/hot scalar Eshkol loops. Reused numerical primitives retain their
  accepted gradient evidence; E3 promises no differentiable output or backward path.
- Fail each allocation, copy, pin, provider admission/preflight, later shard read,
  numerical overflow and final metric preparation boundary. Inject orchestration
  errors before or after a completed provider call; accepted validated K1 invoke
  remains nonfailing, not a recoverable failure-injection promise. Preserve original error,
  exact saved cursor bytes and every mode; no live batch/view/pin or active frame,
  all four caller I2 scalar bytes and both counter values unchanged, and stable retry. Check absent gradients and present nonzero gradients,
  parameter bytes, ties, counts/weights, initializer successor and typed RNG bits.
  No VJP/reset/contribution/graph capture/RNG operation may occur. Mutation witnesses
  must detect graph-producing eval, missed child mode, restored generation counter,
  escaped view/shell and allocating rollback. Public seek failure injection confirms
  it is excluded from the restoration tail.
- Private source/object/AOT path creates a genuine model/tokenizer/D1 corpus/D2
  dataset in one source-composed identity universe, then calls the closed evaluator
  into accepted I2 destinations. Two fresh caches with exact raw depfiles and
  artifact/stdout comparison; wrong arities, foreign/forged/stale/busy inputs,
  destination aliases/dead owners, guessed extern/import, ambient include,
  duplicate-root and observer-leakage negatives. Test-only entry/observers are
  inventoried and never installed; no Python or fixture-carrier runtime enters
  the numerical closure. Eventual installed public-path proof remains deferred.
- Normal and poisoned optimized AOT compare exact retained Eshkol arena after
  1,024 and 8,192 **batches in one invocation** with no live batch. Require byte
  equality and fixed native live/cumulative workspace controls, zero graph owners,
  balanced pins/views/FDs. Separately repeat 1,024/8,192 complete calls reusing the
  same frame/destinations; measure exact live and cumulative I2/frame controls and
  caller-region bytes, including failure/retry trajectories. No report reservation
  or release loop is involved. Reconcile any growth instead of claiming flat memory. Report
  peak/retained RSS, payload, time and unsupported cases separately. ASan/UBSan/LSan
  with leak detection enabled required on supported lane; no disabled-leak evidence.

## 10. Independent review and limitations

Historical initial-proposal review (before root disposition): three independent
Astra/high lanes reviewed actual source before drafting:
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

That approval applies to the original proposal. A separate bounded repair review
by the three Astra/high lanes approved architecture submission at SHA-256
`b5b457879da5f92ad0cdbc80ac430bb72ed88293a7155be0b03c4a03e3b47791`.
Numerical review confirmed bool-only rows/gates and six-output staging; lifetime
review confirmed all17 modes, private reuse and retained exclusion; packaging
review verified every upstream L3S/G3 byte outside the E3 row/log insertion.
Nonblocking wording fixes clarified three nonempty bool patterns, the physical-
capacity denominator mutant, single outer-guard acquisition and direct-pin storage.
Lifetime re-review approved those final cleanup clarifications at SHA-256
`f016f5fa920315a55a97441fd6b64983a1b211cb54aa7829d3ed23717aea6c93`.

The original G3-T owner separately approved exact section6.1/6.2/6.5 wording at
`ba4e6d264c8c840d21bf8f204a4d5b23c3791c8c99e2ae1466bb800ad574ad67`,
after clarifying staged-result publication (not result restoration) and preserving
finalized readiness until success commit. No G3 contract file changed. Reviews
approve this architecture documentation PR only; exact P1/native/source inventories,
ABI and implementation remain held. Detailed delta reviews are local
`.tmp/e3-probes/{numerical,composition,packaging}-delta-review.md`; common-owner
coordination is `/tmp/g3-e3-shared-guard-review.md`.

Detailed source reviews and re-review dispositions are in local
`.tmp/e3-probes/{numerical,composition,packaging}-review.md` and the corresponding
`*-final-review.md` files. These independent agents used GPT-6 Astra/high as
requested. No lane approved runtime behavior or an installed public evaluator.

Private-first architecture alone is accepted; no runtime/ABI/format or new VERIFIED
capability is accepted. Unsupported: arbitrary
profiles, public f32-mask ingress, BPE/shuffled datasets in this proposal, streaming
or unbounded evaluation, ordinary f32 numeric boxing, GPU/mixed precision/f64 retry,
compiler autodiff, public trainer trajectory, checkpoint/resume, generation,
performance or flat total-process memory. Proposed resource ceilings and owner
mechanisms remain implementation gates. This bounded phase is complete only as
proposal/reachability evidence; E3 itself remains active and dependency-gated.
