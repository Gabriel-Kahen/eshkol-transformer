# TR3 public trainer metrics proposal

Status: **decision-ready successor proposal; not accepted; runtime and E3 prerequisites missing**.

This proposal is based on accepted transformer commit
`d5fa9a71b000ab1139a25e9b734b230ab5bda759`, the original TR3 proposal at
`3e28db6c9d1f97bd9ce9d6f2d05be51182a8bd9d`, the accepted E3 private contract,
and genuine Eshkol source commit
`81298b4a9608fb92eb6f351a2eabd8392da7d9ef`. It defines the downstream result
path only. It implements no trainer, evaluator, scalar representation, registry,
capability row, C2 change, or completion claim.

## 1. Decision and blocker

Keep the exact A0 maps and implement them through one TR3 aggregate authority.
The result authority stores physical binary32 words and exact i64 counters; it
does not store boxed binary64 values, rank-zero tensor shells, or references to
trainer/E3 storage.

The primary implementation blocker is an upstream runtime contract: an ordinary Eshkol
value has no true binary32 scalar representation that `metrics-ref` can return.
The current floating immediate is `ESHKOL_VALUE_DOUBLE`, backed by a C `double`.
Relabelling that value as f32 would make the public type claim false. A rank-zero
I2 tensor would preserve storage precision but would be an owned tensor, not the
detached scalar selected by the original TR3 proposal. A transformer-local boxed
scalar would invent a new public representation and is not selected here.

The generic hash table is also not a usable result carrier: its public
`hash-set!`, `hash-remove!`, and `hash-clear!` operations make it mutable. The
selected fixed-schema opaque identity avoids requiring a new generic immutable
map primitive.

## 2. Exact public schemas

A0 requires opaque immutable maps, the exact schemas and types below, newly owned
CPU values with no gradient, and `invalid-argument` for unknown keys. No aliases,
optional keys, or extra diagnostic fields are admitted.

| Kind | Exact keys and types |
|---|---|
| step | `loss:f32`, `mask-weight:f32`, `tokens:i64`, `microbatches:i64`, `update:i64` |
| train | `loss:f32`, `mask-weight:f32`, `tokens:i64`, `updates:i64`, `epochs:i64` |
| evaluation | `loss:f32`, `mask-weight:f32`, `tokens:i64`, `batches:i64` |

Every i64 is exact and nonnegative. Every f32 is the accepted producer's finite
physical binary32 result. Evaluation additionally requires positive mask weight;
zero is `invalid-state` before result publication.

The original TR3 proposal further selects a detached scalar and no retained trainer
or tensor storage. This successor selects an immediate result, exact preservation of
the producer's binary32 payload, and allocation-free lookup. Those are TR3
refinements, not claims about A0 text. For i64 fields the result is the existing
exact immediate integer. For f32 fields it is the accepted upstream binary32 scalar
described in section 8. Inspection performs no arithmetic, f32-to-f64 conversion,
tensor allocation, or storage borrow.

## 3. One authority and one representation

The genuine lease-core source at `73cd89b` and its corrected successor
`f602a66644ed4ce8519d14c9f2142fe5b9a4d3a5` define `tr3-trainers` as a two-slot
root: slot 0 contains only exact 17-slot trainer records and slot 1 is its temporary
promotion root. Admission and overlap scans depend on that layout. Metrics must not
enter `tr3-trainers`, extend its record kind set, or change the 17-slot contract.

The selected successor adds exactly one lexically hidden `tr3-metrics` strong
registry in the same final source aggregate, loaded after the lease core and defined
nowhere else. It is an exact two-slot root: slot 0 is the published metrics list and
slot 1 is the sole temporary promotion root.
This is the only metrics authenticity/lifetime authority. E3, trainer-step,
trainer-train, public facades, and native code do not own another metrics registry.
The change is a dependency contract for the final aggregate, not an existing API or
an edit to the independent lease candidate.

Each returned map is an existing fieldless native condition object used only as an
identity token, following the accepted E1 pinned-runtime pattern. Its message or
irritants carry no fields or brand. The lexically hidden registry owns all copied
data and authenticates only exact token identity; `metrics-ref` never invokes or
reflects on a caller value. This choice rejects an ephemeral record/hash carrier:
the pinned compiler lowers records to caller-mutable vectors and ordinary hashes
are mutable. It also makes the lifetime choice explicit: strong-registry entries
are process-lifetime and remain valid until process exit.

All three kinds use one private exact ten-slot record:

```text
0 tag            private metrics-record-v1 identity
1 self           exact record self-identity
2 token          fieldless native condition returned as the map
3 kind           step | train | evaluation
4 state          ready while staged; published before the nonfailing enrollment tail
5 loss-bits      exact i64 containing one uint32 binary32 word
6 weight-bits    exact i64 containing one uint32 binary32 word
7 counter-0      exact nonnegative i64
8 counter-1      exact nonnegative i64
9 counter-2      exact nonnegative i64; #f for evaluation
```

The schema descriptor for `kind` determines the exact key set, counter count and
key-to-slot mapping. Step and train therefore share construction, authentication,
publication, and lookup code. They differ only in the fixed descriptor. Evaluation
uses the same record with two counter slots and cannot expose slot 9.

| Kind | slot 7 | slot 8 | slot 9 |
|---|---|---|---|
| step | `tokens` | `microbatches` | `update` |
| train | `tokens` | `updates` | `epochs` |
| evaluation | `tokens` | `batches` | `#f` |

Each successful operation returns a fresh CPU token and record. Slots 5 and 6 are
exact integers in `0..UINT32_MAX`; slots 7 through 9 follow the table above. The
record copies its
two f32 words and counters and retains no trainer, dataset, tensor, E3 frame, result
destination, borrow, graph, or region-local value. A byte-identical foreign
condition, copied vector, forged token, or ordinary hash grants no authority.

A0 defines no metrics release operation, and the accepted runtime has no proved
finalizer for this owner. The selected strong registry therefore retains every
published map until process exit, with memory linear in the number of successful
result-producing calls. It has no total-memory bound. `trainer-step!` and
`trainer-evaluate!` each publish one map per successful call. `trainer-train!`
publishes one summary per invocation, not one map per internal
step. Failed operations publish and retain no registry entry. They can still leave
unreachable root-arena, E1, or runtime allocations made before failure; Wave 3 must
measure repeated failpoint retention slopes and totals and must not claim failed-call
memory is flat without that evidence.

Wave 3 must measure the exact incremental retained bytes per result kind and live
retained totals after 1,024 and 8,192 successful publications. Those measurements
describe the slope and do not imply flat reuse or a lifetime bound. This proposal
adds neither a hidden quota nor a release operation. A future release/finalizer
improvement requires a separate public/runtime contract and cannot invalidate
already returned identities silently.

## 4. `metrics-ref` contract and validation order

`metrics-ref metrics key` follows this fixed order:

1. Authenticate `metrics` against the sole `tr3-metrics` registry without
   dereferencing untrusted pointer-shaped data.
2. Require an exact ten-slot, self-identical, tagged, `published` record with one of
   the three metrics kinds. A nonmetrics aggregate identity, foreign value, forged
   value, or ordinary hash is `invalid-argument`. An authenticated record in any
   other state is `invalid-state`, and no field is read from it.
3. Require `key` to be an exact interned symbol. Other types are
   `invalid-argument`.
4. Resolve the symbol only in that entry kind's fixed descriptor. Unknown names,
   case variants, and a valid key from another result kind are
   `invalid-argument`.
5. Return the selected detached immediate. An impossible corrupt kind, slot, or
   stored-type invariant is an internal fatal defect, never a fabricated scalar.

Lookup does not return an internal pair/vector/hash slot and exposes no mutation
operation. Repeated lookup is observational and bit-stable.

## 5. Allocation, promotion, and publication

After ordinary receiver/argument authentication and before allocation or any
trainer/evaluator mutation, construction requires metrics slot 1 empty and a
self-consistent published list. The
token, fixed record, complete successor list, and any error-delivery
controls are then allocated before an operation reaches a state whose successful
effects cannot be rolled back. The checked region write barrier promotes every
object needed by the longer-lived registry before that boundary. Allocation or
promotion failure aborts while all public state is still unchanged. The complete
staging envelope remains in `tr3-metrics` slot 1 and the token is withheld from the
caller. A prepublication failure clears that staging root and returns no map; slot
0 remains unchanged.
After promotion, construction rereads the canonical successor, record, token, and
supporting values from the slot-1 envelope and uses only those canonical values for
every later store, enrollment, and return. Region-local aliases are not reused.

Publication follows the proved lease enrollment shape: set the private record state
to `published`, remove the staging edge, install the already-promoted successor as
slot 0, and return the token. Those stores have no allocation, checked promotion,
callback, provider dispatch, numeric conversion, or recoverable error path. The
successor list is constructed from an observed metrics head. For step, that head is
rechecked before the first parameter write; for train, before the first update; and
for evaluation, before the first public-record store. The metrics-construction
exclusion remains held through publication, so the head cannot change afterward.
An impossible postcommit mismatch is fatal rather than a recoverable error.
Reentrant or concurrent metrics construction is `invalid-state`; no concurrency
claim is made.

### Step

Allocate/promote the result entry before batch consumption. The step's physical
f32 numerator/weight path determines loss and weight before the first parameter
write, and the staged counter values are already known. Fill the private entry and
validate all fields before the O2 commit. After the first parameter write, finish
the accepted nonraising parameter/optimizer/scheduler/counter tail and publish the
prevalidated entry. A recoverable precommit failure leaves the entry unreachable
and the whole current step uncommitted.

`tokens` and `microbatches` describe only this update. `update` is the global
completed-update count after this commit.

### Train

Allocate/promote one summary entry before the first update. `trainer-train!` calls
the private step transaction rather than the public result-producing wrapper, so
it does not create and discard one public step map per update. It accumulates the
same physical-f32 numerator/weight representation in microbatch order in separate,
preallocated TR3-B summary scratch, plus exact invocation-delta counters. Before
each step's first parameter write, it computes and checks the candidate N/W scratch,
finalized `loss=fl(N/W)`, weight, and counters. Only the finalized loss bits, weight
bits, and counters enter record slots 5 through 9; no numerator slot or mean-of-means
path exists. The step's postcommit tail does not touch the summary. Each update
remains its own commit. On a later failure,
earlier updates remain committed, the current step rolls back, and the unpublished
summary entry is abandoned. Successful stop finalizes and publishes the one
preallocated summary entry.

### Evaluation

Allocate/promote the public entry before invoking E3. E3 retains sole authority for
its mode/cursor snapshot, no-grad traversal, restoration, and atomic six-output
private publication. Only after E3 has restored all state and published its private
destinations and `finish` has returned the frame to idle does the TR3 adapter call
the proposed E3-owned private seam twice:

```c
int64_t et_e3_private_selected_metric_bits_ref_v1(
    void *frame, int64_t selector);
```

The current E3 ABI exposes counters but no destination getter or exact-bit copy, and
the fixed 17-slot trainer lease retains neither destination. TR3 therefore must not
call `et_f32_tensor_copy_bits_to_v1` on an assumed pointer or add destination fields
to the trainer record.

Selector 0 names loss and selector 1 names mask weight; every other selector returns
`-1` with the existing E3 invalid-argument diagnostic. The seam
registry-authenticates before dereference and requires a live frame, `E3_IDLE`,
`published_valid==1`, `committed==2`, inactive model and no cleanup residue. An
invariant disagreement returns `-1` with the existing internal diagnostic. For a
valid selector it copies only that bound private destination through existing
`et_f32_tensor_copy_bits_to_v1` into a local `uint32_t` and returns the widened word
as an `int64_t` in `0..UINT32_MAX`. Success therefore cannot collide with `-1`.
It allocates nothing, exposes no tensor pointer, output pointer, borrow, perplexity
or accuracy, and retains no TR3 object.

An unregistered/foreign pointer returns `-1` with E3 `invalid-argument/identity`;
an authentic dead frame, nonidle or unpublished frame, occupied model, or invalid
selector returns `-1` with the corresponding existing E3 `invalid-state` or
`invalid-argument` diagnostic. A published/committed/cleanup invariant disagreement
returns `-1` with E3 `internal/invariant`. An I2 exact-bit-copy rejection returns
`-1` while preserving its validated domain-3 category/code in the E3 diagnostic
triple. No failure changes frame, destination, counter, or caller storage.

E3's evaluation `m3-call` guard has ended at this point; the proposal does not
pretend it remains held. Before any callback, provider dispatch, or user code can
run, TR3 reacquires the existing `m3-call 'trainer-evaluate!` guard and keeps the
metrics-construction exclusion held around the closed sequence `bits_ref(0)`,
`bits_ref(1)`, `counter_ref(0)`, `counter_ref(1)`. There is no second model token or
new guard API. The existing guard rejects reentry, and this proposal makes no
concurrent-thread claim. TR3 stages all four returned i64 values locally and writes
no public entry slot until every call succeeds. It then validates finite loss and
positive finite weight by binary32 bit classification without widening, validates
the exact counters, installs the staged words/counters, and publishes through its
nonfailing tail. Every seam call, check, and diagnostic delivery completes before
the first public-record store.

The runtime checked barrier supports this ordering: its checked ABI stages the
possibly promoted value and reports allocation/layout/overflow/call-state failure
without changing committed graph/map state. Its no-promotion path allocates no
transaction storage. That is precommit evidence, not permission to call a fallible
barrier after the first parameter write.

## 6. Exact E3 projection

The accepted E3 private frame publishes these destinations in order:

```text
loss:f32[]  mask-weight:f32[]  perplexity:f32[]  token-accuracy:f32[]
tokens:i64  batches:i64
```

The public adapter maps them as follows:

| E3 private output | Public evaluation entry |
|---|---|
| loss f32 word | `loss` f32 word, copied exactly |
| mask-weight f32 word | `mask-weight` f32 word, copied exactly |
| perplexity f32 word | no public field |
| token-accuracy f32 word | no public field |
| tokens i64 | `tokens` exact i64 |
| batches i64 | `batches` exact i64 |

Perplexity and accuracy remain required private E3 computations and destinations;
they do not enter the public entry, alter its schema, or gain an accessor alias.
The adapter does not recompute loss, weight, perplexity, accuracy, tokens, or
batches and does not widen then narrow either f32 word.

## 7. Source and probe evidence

Genuine runtime source is commit
`81298b4a9608fb92eb6f351a2eabd8392da7d9ef` from
`/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery/eshkol-source`:

- `inc/eshkol/eshkol.h:99-121` enumerates immediate tags and provides only
  `ESHKOL_VALUE_DOUBLE` for an inexact real. Lines 180-200 define the tagged union
  with `double double_val` and no float member.
- `lib/backend/tagged_value_codegen.cpp:115-149` packs only LLVM `double` as the
  floating immediate and maps an unrecognized raw type to a warned null value.
  Lines 330-353 unpack to LLVM `double`.
- `lib/backend/vm_core.c:126-201` uses `VAL_FLOAT` backed by `double` and calls it
  the VM's only inexact representation.
- `lib/backend/hash_codegen.cpp:391-429`, `534-555`, and `707-729` expose mutation
  of ordinary hash tables through set, remove, and clear.
- `lib/core/runtime_regions.cpp:2074-2119` and
  `lib/core/arena_memory.h:675-686` define the checked promotion contract used by
  the precommit ordering above.
- Transformer `include/eshkol_transformer/f32_tensor.h:136-142` and
  `native/f32_tensor.c:961-985` define the existing fallible exact-bit copy that the
  proposed E3-owned seam can use internally; it preflights then `memcpy`s without
  numeric conversion.
- Current `src/eshkol_transformer/e3_frame_internal.h:12-36` exposes no destination
  getter/copy seam. `src/eshkol_transformer/e3_frame.c:1180-1271` publishes four
  destinations and two counters, then exposes only the two counter values after
  finish. This is the concrete E3-to-TR3 ownership gap.
- Lease successor `f602a66644ed4ce8519d14c9f2142fe5b9a4d3a5`,
  `native/tr3_lease_core_extension.esk:5-8,224-296`, fixes `tr3-trainers` as the
  two-slot root and its enrolled records as exact 17-slot vectors.

The accepted E3 proposal's recorded strict-AOT probe found that ordinary `1.25`
emits tag 2 with binary64 payload `4608308318706860032`, while `metrics-ref` fails
as an unknown function. A fresh host attempt with the exact cached executable
failed before compilation because the host lacks `libLLVM.so.21.1` and
`libopenblas.so.0`; both processes exited 127. No compiler was rebuilt.

The supported rerun uses immutable image ID
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`
(Ubuntu 22.04/LLVM 21.1.8) and the recovered supported runner SHA-256
`f4ddddb7e441cdf3002bc374501cff62a508b021fccd99e5365b819e8fed3aeb`.
Its recorded source is the same clean commit and tree
`7669312845a9d8d372006af52271045e69505813`. The durable record is
`/home/gabe/.codex/evidence/eshkol-transformer/tr3-metrics-probe-81298-20260923T-final`;
`command.sh` invokes the image with `--network none`, mounts the recovered build
read-only, and runs this exact inner form for each probe:

```text
/candidate/eshkol-build/eshkol-run --strict-types --no-stdlib --compile-only \
  --dump-ir /evidence/<probe>.esk -o /evidence/<probe>.o
```

`runtime-provenance.tsv`, `probe-provenance.tsv`, `SHA256SUMS`, the sources,
stdout/stderr, object/bitcode/IR, and `results.tsv` are retained there. Results were:

| Probe | Exit/result |
|---|---|
| ordinary `1.25` | 0; LLVM IR stores `{ i8 2, i8 32, ..., i64 4608308318706860032 }` |
| `(metrics-ref #f 'loss)` | 1; `Unknown function: metrics-ref` |

This proves the current tagged behavior on the supported toolchain; it does not
exercise a trainer, evaluator, metrics owner, or proposed f32 representation.

## 8. Minimal upstream issue/contract candidate

Dispatch one runtime-owned prerequisite, **true binary32 tagged scalar**. Its
acceptance contract should require:

1. Freeze native `eshkol_value_type_t` as a pure enum and reserve runtime-owned
   `ESHKOL_VALUE_FLOAT32=11`. Its canonical immediate payload is one IEEE-754
   binary32 word in `data.raw_val[31:0]`; the upper 32 payload bits, `reserved`, and
   padding are zero. `flags` contains `ESHKOL_VALUE_INEXACT_FLAG` and no exact flag.
   Every producer and classifier requires `type==11` exactly. The legacy macros
   that OR exactness bits into `type` are forbidden for f32 because those bits
   collide with real tags; global cleanup is a separate hardening item unless an
   audited f32 path reaches them.
2. Keep the VM namespace distinct. Native tag 11 collides with current VM
   `VAL_RATIONAL=11`, so append `VAL_FLOAT32=34` carrying raw `uint32_t` bits. FFI
   metadata may name native type 11, but no code copies native tag numbers into VM
   values.
3. Add a versioned core out-pointer raw-bits ABI for checked construct, inspect and
   predicate operations (`from_bits`, `to_bits`, `is_f32`, with float conversion
   helpers optional). Inspection rejects DOUBLE, tensor, integer and foreign values
   and leaves output unchanged. No transformer public constructor is requested.
4. Cover native/AOT pack and unpack, generic native/AOT and VM function-return
   transport, embedding headers, ABI layout assertions, FFI/VM-host raw-bits
   boundaries, and
   byte-preserving region/cons/vector/hash copies. Immediate construction and
   inspection allocate no owner or payload.
5. Define all classifiers and observational operations. For f32, `number?` and
   `real?` are true, `rational?` is true exactly when finite, `integer?` is true
   exactly when finite and integral, `exact?` is false, `inexact?` is true, and
   `type-of` reports `float32`. Equality/hashing must be mutually consistent across
   the admitted numeric tower, and display must handle every payload. Raw-bits ABI
   round-trip is mandatory. Source f32 literals,
   type-preserving textual read/write syntax and ESKB f32 constants/versioning are
   optional follow-ups and do not block metrics.
6. Fix arithmetic promotion rather than inheriting binary64 accidentally:
   f32-with-f32 and exact-integer-with-f32 produce f32 with actual binary32
   rounding; any binary64 operand produces binary64. For an exact integer operand,
   convert the integer to f32 first with round-to-nearest, ties-to-even, then perform
   the f32 operation with the same rounding mode. Exact-result-then-round is
   forbidden: `16777217 + f32(-16777216)` must produce positive zero, not one.
   A mixed nonintegral exact rational and f32 is rejected pending a separately
   reviewed numeric-tower contract; it must not silently widen or route through
   binary64. Existing binary64
   AD/dual/Taylor/complex paths either reject f32 or cross an explicit documented
   widening boundary.
7. Add exact-bit tests for normals, signed zero, subnormals and finite extrema;
   negative tests proving ordinary boxed binary64 never passes f32 inspection;
   checked type/flags collision tests; and native/AOT/VM/embedding/FFI parity on the
   supported host.
8. Repin the accepted Eshkol runtime only after the implementation and runtime
   contract pass independent review. F0 provenance/build, R0 executable capability
   audit, and A0 declaration/compile fixtures must all consume that exact repin.

After that contract is accepted and implemented, TR3 may add its private result
entry/lookup implementation and public `metrics-ref`. Until then, all public
trainer/evaluator wrappers that promise these maps remain blocked.

## 9. Dependency order and root decisions

The dependency order is:

1. Root accepts the exact true-f32 runtime issue, the unbounded process-lifetime
   metrics authority with measured per-result retention, and the E3 selected-metric
   scalar-bits seam.
2. The runtime owner implements and independently verifies section 8, followed by
   exact runtime repin and F0/R0/A0 adoption evidence.
3. SHARED-R2 and the checked-promotion/guard prerequisite for E3-P1 are accepted;
   E3-P1 mode restoration and the E3 private evaluator, including the new
   selected-metric scalar-bits seam, are independently accepted and integrated.
4. The corrected fixed-17-slot trainer lease successor is independently accepted
   and integrated. TR3-A then source-composes exactly one metrics extension after
   that lease core without changing `tr3-trainers`.
5. TR3-B supplies physical-f32 step/train accumulation and TR3-O supplies the
   accepted nonfailing update-and-clear commit. Their combined step must preserve
   the allocation and head checks in section 5.
6. TR3-C joint live snapshot/restore is integrated without changing C2 bytes or
   adding metrics to checkpoint state. Metrics are observational and are not
   reproducibility state.
7. Public trainer/evaluator wrappers and `metrics-ref` receive exact-type,
   unknown-key, forgery, transaction, E3 projection and lifetime tests;
   one-batch overfit, held-out improvement and interrupted/resumed equivalence then
   exercise the actual public path.
8. Independent aggregate review, the complete affected 19-suite union, supported
   Ubuntu 22.04/LLVM 21 CI and final exact-tree acceptance run last. No component
   evidence alone completes Wave 3.

Three decisions remain open for root and are not silently chosen by implementation:

- accept unbounded process-lifetime retention with measured successful and failed
  call slopes, or select a separately reviewed release/finalizer contract;
- accept the E3-owned authenticated selected-metric scalar-bits seam and its exact
  signature/state/error contract;
- dispatch and accept the runtime representation in section 8. Returning a DOUBLE,
  rank-zero tensor, bytevector, or transformer-local scalar wrapper is not an
  admissible substitute.

## 10. Required implementation handoff

Every prerequisite and downstream implementation handoff must record, rather than
infer from a passing aggregate:

1. the exact public and private APIs/contracts added or changed, including type,
   ownership, state, error, and tensor-shape rules;
2. every test/probe/CI command run and its result, tied to the exact supported image
   or host and immutable input artifacts;
3. measured retention slopes/totals, unsupported dtypes/devices/features, and every
   other limitation that remains;
4. follow-up dependencies and risks, including the exact accepted upstream commits
   and any still-proposed seams; and
5. an exact commit and tree (or pull request plus both identities) suitable for
   independent integration and identical-tree retest.
