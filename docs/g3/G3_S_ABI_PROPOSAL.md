# G3-S bounded sampler ABI proposal

Status: **exact ABI accepted for implementation**, under [decision 5751933660](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5751933660)
and the binding C2 [decision 5748532295](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295).
The accepted contract is `d2b0858a39d99954396ade51b019f6094a00a0f3`, reviewed
SHA-256 `a27f281bba82359995a6d13bc8061990fb9fe5b196199572427c9dd81d584f16`.
The proposal text below is preserved; its requested ABI decisions are now settled.
No code, build, numerical run or capability
claim is delivered here. G3-S is independent of G3-N and has no L3S dependency.

## Provider and ABI identity

- Public header: `include/eshkol_transformer/g3s_sampling_abi.h`.
- Semantic macros: `ET_G3S_SAMPLING_ABI_MAJOR 1u`,
  `ET_G3S_SAMPLING_ABI_MINOR 0u`; these version the schemas below, not K1 layouts.
- Only public native symbol:
  `const et_kernel_provider_v1 *et_g3s_kernel_provider_v1(void);`.
- Provider name and capability implementation: `g3s.cpu-f32.serial`; version
  `1.0`; evidence identity `G3-S:cpu-f32-v256-v1`.
- K1 ABI major/minor 1/0, required features zero, immutable static provider and
  metadata, no provider registry, owned tensor, dynamic discovery or canonical
  `eshkol_transformer_kernel_provider_v1` symbol.

Two capabilities prevent any inactive-policy ambiguity:

| Capability | Sole operation | Exact request row | Compute dtype/device |
|---|---|---|---|
| `g3s.greedy` | `g3s.greedy.forward` | rank 2 `[1,256]` | `f32` / `cpu` |
| `g3s.categorical` | `g3s.categorical.forward` | rank 2 `[1,256]` | `f32` / `cpu` |

Each row uses min=max extents, no unbounded flag. Both capabilities advertise
deterministic=1 only after their evidence passes. K1 requests may specify its
ordinary boolean deterministic value 0 or 1; both execute the identical checked
deterministic path. The later G3 aggregate always requests 1. No batch, vocabulary
range, backward, gradient, RNG-only operation or implicit policy mode is admitted.
Direct provider validation enforces the same operation/row matrix as K1 discovery.

## Exact call schemas

K1 has no scalar-parameter struct. All parameters are ordered tensor views; rank
zero denotes one scalar with four bytes for f32 or eight for i64. All views use
CPU, dense row-major layout, offset zero, naturally aligned storage and exact
byte spans. Input count is 2 for greedy and 5 for categorical; output count is 2
for both. There are no optional views or omitted/default scalar fields.

| Table/index | Greedy | Categorical |
|---|---|---|
| input 0 | logits f32 `[1,256]`, 1,024 bytes | same |
| input 1 | numeric RNG i64 `[4]`, 32 bytes | temperature f32 rank 0, 4 bytes |
| input 2 | absent | top-k i64 rank 0, 8 bytes |
| input 3 | absent | top-p f32 rank 0, 4 bytes |
| input 4 | absent | numeric RNG i64 `[4]`, 32 bytes |
| output 0 | selected token i64 `[1]`, 8 bytes | same |
| output 1 | unchanged numeric RNG i64 `[4]`, 32 bytes | successor numeric RNG i64 `[4]`, 32 bytes |

Temperature is finite and positive, top-k is 1..256, top-p is finite and in (0,1].
The public configuration's exact uint32-range bit integers are materialized into
these f32 scalars by bit copy in G3-T; the sampler performs no generic-real cast.
Greedy receives no inactive policy fields. G3-G still rejects noncanonical greedy
temperature/k/p before dispatch, as already accepted. No policy bytes or logits
are mutated and no probability/debug tensor is a production output.

RNG words are numeric data `[1,seed,counter-low,counter-high]`, not a native owner
or an authenticated G3 object. Seed is 0..INT64_MAX. Each counter half represents
an unsigned 64-bit word as u or u-2^64 when u exceeds INT64_MAX. All counter bit
patterns are admitted, including the exhausted sentinel `[-1,-1]`. Version other
than 1 rejects. G3-T separately owns authentic shells, explicit RNG release,
logical copying, lifetime and wrong-kind/cross-aggregate rejection. This numeric
schema is not a new serialized format or authority to accept N2/M3T state shells.

## Settled arithmetic and draw semantics

The literal arithmetic specification is the accepted sampling section of
[NUMERICAL_REVIEW.md](NUMERICAL_REVIEW.md); the binding decision controls its
exhaustion wording. All 256 logits are finite, even discarded entries. Greedy
selects numeric maximum/lowest-ID tie, treating signed zeros as ties, with no
division, exponentiation or RNG draw. Categorical executes separate binary32
statements in this order:

1. Maximum-subtracted logits, division by temperature, `expf`; reject nonfinite
   subtraction/division/results, explicitly permit exponential underflow to +0.
2. ID-order sum and division; sort rounded probabilities descending then ID
   ascending. Retain top-k; sorted-order sum and renormalization.
3. Shortest nonempty prefix whose rounded cumulative probability meets p;
   retain all k if none does. No epsilon; p=1 may reach one early.
4. Sorted-order sum W of retained renormalized weights, `r=u*W`, strict `CDF>r`
   selection, otherwise last positive-weight token. Zero-weight tails never win.

No host `qsort`, rejection sampling, clipping, second draw or alternative softmax
order. Use bounded nonrecursive native kernel loops and a total probability/ID
ordering. Proposed normal-build automatic-storage ceiling is 16,384 bytes for
the maximum simultaneously live G3-S provider/helper frames per validation or
invocation, including scratch and compiler spills; exclude the caller/K1 frames
and external libc/libm frames and report those exclusions. No recursion, VLA or
dynamic `alloca`. Prove the ceiling with compiler stack-usage output plus runtime
high-water evidence; this is a proposed limit, not a measurement. This is a
reviewed numerical kernel, not an Eshkol scalar fallback.

Algorithm identity is `g3.philox4x32-10.categorical-f32.v1`. Reproduce the literal
N3K Philox round/lane order; key is `uint64(seed) XOR 0x4733434154454731`, split
low/high into k0/k1. Counter c0/c1 are low/high 32 bits of counter-low; c2/c3 are
low/high 32 bits of counter-high. Ten rounds use multipliers 0xd2511f53 and
0xcd9e8d57 and key increments 0x9e3779b9 and 0xbb67ae85, with no increment after
round ten. Categorical uses lane0 high24 bits times exact f32 2^-24, discards the
other lanes, and advances the full 128-bit counter once. EOS and singleton
selection consume one block. The final consumable block is maximum minus one;
it may return the maximum exhausted successor. Never consume or wrap maximum.

**Exhausted state is valid for greedy and is copied unchanged.** G3 construction,
manual prefill/decode and zero-budget generation also admit and preserve it;
these operations do not call categorical. A required C2 categorical draw rejects
the sentinel before generate's initial prefill commit. G3-G performs that early
budget/state check; native categorical repeats it before any numerical work or
output publication. A native speculative success returns a candidate/successor,
not an emitted-token commit: G3-G must discard both on model/cache failure and
publish neither until its joint token/cache/RNG commit. Multi-token raw sampler
calls can test Philox continuation without admitting larger-model generation.

## Validation, floating environment and publication

Audit basis: K1 size-tagged tables in `kernel_abi.h` and `native/kernel_abi.c`;
N3K `fp_environment_supported`, `provider_validate`, `philox4x32_10` and dry
`run_operation`; N2 `begin_preflight`/`preflight_result`. No existing sampler was
found. N3K adds x87 exception-mask checks beyond N2's MXCSR/rounding checks; use
that stronger exact control contract here.

Build only the reviewed x86-64 binary32 lane: CHAR_BIT=8, sizeof(float)=4,
FLT_RADIX=2, FLT_MANT_DIG=24, FLT_MAX_EXP=128, FLT_EVAL_METHOD=0. Require x87
exception mask bits 0..5 all set and rounding bits 10..11 zero; require MXCSR
exception-mask bits 7..12 all set, rounding bits 13..14 zero, DAZ bit 6 clear and
FTZ bit 15 clear. Check controls before reading/classifying any floating payload,
including signaling NaNs. Existing sticky flags are allowed. Use FENV_ACCESS ON,
FP_CONTRACT OFF, `-ffp-contract=off -fexcess-precision=standard -fno-fast-math`.
Exact `expf` results are scoped to the recorded supported toolchain/libm lane;
compatibility execution cannot establish supported bitwise determinism.

Validation saves the incoming fenv and restores it on every return, including
malformed inputs and unsuccessful numerical dry runs. Unobservable environment
is unsupported; failed restoration is internal and cannot claim preservation.
Validate K1 struct prefixes (call >=88, request >=56, view >=72, view size <=
table stride), aligned table/count/stride/span arithmetic, exact capability/row,
dtype/device/layout and exact ordered operands. Accept future prefix-compatible
tails as K1 requires; never dereference their contents. Then reject aliases,
check controls, version/seed/policy and conditional exhaustion, and dry-run every
prospective arithmetic operation into bounded stack scratch. No output writes,
retained pointers, callbacks, provider state, heap allocation or counter mutation.

All input payloads are pairwise disjoint, even equal scalar controls; every output
is disjoint from all inputs and other outputs. These payload overlap rules are
checked before computation. Writable output storage must also be disjoint from
request/call/table/shape/string metadata and the error record: this is the existing
K1 caller stable-metadata/control-span precondition, not a newly promised general
metadata-alias detector. Inputs and all descriptors remain stable until synchronous
dispatch returns. Caller-provided error storage is not an owned result or part of
output preservation. Arbitrary unreadable native pointers remain outside K1's
C-caller threat model.

K1 dispatch completes generic validation and provider validation before `void`
invoke. Invoke recomputes into local token/state scratch under the unchanged
controls/inputs and writes both complete outputs only after computation finishes.
It allocates nothing, retains nothing and has no recoverable branch after the
first write. Success may accrue ordinary arithmetic exception flags without
changing controls. Logical publication is atomic under the serialized-call
contract, not a claim of simultaneous hardware stores or concurrent observers.
No new prepare-plan handle, RNG registry, public fenv switch or K1 ABI is needed.

ABI-local errors use existing `et_kernel_error`. K1 generic validation and
capability admission retain their existing precedence and reason codes. The table
below describes direct/provider-local validation unless marked otherwise:

| Condition | Category / code |
|---|---|
| Truncated prefix or bad RNG version | VERSION_MISMATCH / INVALID_STRUCT_SIZE or PROVIDER_REJECTED, respectively |
| Missing, misaligned, overflowing or wrong-count table/span | INVALID_ARGUMENT / NULL_ARGUMENT, INVALID_BUFFER or INTEGER_OVERFLOW as applicable |
| Wrong operand shape, dtype, device or layout/offset | SHAPE_MISMATCH / INVALID_SHAPE; DTYPE_MISMATCH / INVALID_TEXT; DEVICE_MISMATCH / INVALID_TEXT; NONCONTIGUOUS / INVALID_BUFFER |
| Output alias | INVALID_ARGUMENT / ALIASING_OUTPUT |
| Input overlap, invalid policy/seed, nonfinite input/intermediate, required draw exhausted | INVALID_ARGUMENT / PROVIDER_REJECTED |
| Outside exact capability/operation/row, unsupported FP controls | UNSUPPORTED / PROVIDER_REJECTED (K1 capability admission retains its own earlier reason) |
| Cannot observe/restore fenv | UNSUPPORTED / PROVIDER_REJECTED; INTERNAL / PROVIDER_REJECTED, respectively |

In particular, through K1 an operand device mismatch is DEVICE_MISMATCH /
INVALID_BUFFER; shape-product overflow is SHAPE_MISMATCH / INTEGER_OVERFLOW;
an exact byte-length mismatch is SHAPE_MISMATCH / INVALID_BUFFER; and generic
output overlap/data-address wrap is INVALID_ARGUMENT / ALIASING_OUTPUT. Direct
provider byte-length/alignment mismatch is INVALID_ARGUMENT / INVALID_BUFFER,
and data-address wrap is INVALID_ARGUMENT / PROVIDER_REJECTED. Missing exact
rows/operations through K1 retain UNSUPPORTED / CAPABILITY_NOT_VERIFIED.
Operation text identifies the attempted `g3s.*` operation, or `g3s.provider`
before a valid operation is available; explanatory message text is not a reason
discriminator. No generic error is remapped to disguise provider validation.

G3-T/G3-G own public E1 translation, including early draw-exhaustion
`invalid-state` and unsupported deterministic environment
`determinism-unavailable`; these names are not new K1 enum values.

## Build inventory and acceptance

Proposed native source closure is exactly
`native/g3s_sampling_provider.c`, its public header above, and
`include/eshkol_transformer/kernel_abi.h`. Any copied N3K Philox/control helper
stays static and records source provenance; do not include or link another whole
provider to expose hidden helpers. Proposed archive:
`build/g3s/libeshkol_transformer_g3s.a`, exactly one member
`g3s_sampling_provider.o`, exactly one defined external symbol, the accessor above.
Public installed Eshkol exports added by G3-S: zero. Provider validate/invoke and
all numerical helpers remain static. Later source composition preserves one
P1/T1/E1/I2 owner; this registry-free archive introduces none.

Build/test script candidates are `scripts/build-g3s.sh`, `scripts/test-g3s.sh`;
tests live under `tests/g3s/`. Use the verified-toolchain pattern: C11, `-O2`, PIC,
strict warnings, `-fstack-protector-all` and the exact FP flags above, dependency
files, deterministic `ar rcsD`, and staged atomic artifact replacement.
The allowed normal-object undefined-symbol ceiling is exactly
`__stack_chk_fail`, `et_kernel_error_clear`, `expf`, `fegetenv`, `fesetenv`,
`memcpy`, `memset`, `snprintf`, `strcmp`. Freeze the actual exact supported-compiler
subset in implementation review; this proposed ceiling is not a measured object
inventory. Reject allocation, qsort, dynamic loading, thread/runtime registry,
Python, predecessor provider accessors and alternate/binary64 math symbols.
Sanitized artifacts have separate instrumented inventories and are never shipped
as canonical objects. No full aggregate build is required for contract review.

Acceptance must independently prove:

- literal domain-tagged Philox vectors, lane/counter/key ordering, high24 mapping,
  carry, last consumable block/sentinel, greedy exhausted-state preservation;
- literal token/probability/CDF boundary vectors, signed-zero and rounding ties,
  k=1/256, p equality/p=1 early reach/full-sum fallback, underflowed tails,
  overflow/nonfinite rejection, minimum positive policy values, u=0/max uniform,
  singleton one-draw behavior and mutation witnesses for alternate ordering;
- separate high-precision mathematical reference with declared tolerance, never
  self-comparison as the sole oracle; no differentiability/backward claim;
- exact two-row metadata and both full operand schemas via K1 and direct
  validation, adjacent unsupported shapes, every count/stride/prefix/future-tail,
  dtype/device/layout/span/alias negative, output/input byte preservation;
- all x87/MXCSR rounding/mask/FTZ/DAZ negatives, signaling-NaN trap subprocesses,
  preset sticky flags preserved by every validation return, and invoke controls
  unchanged; allocation-disabled dispatch and bounded measured native stack use;
- ASan/UBSan/LSan, independent native caller plus compiled-Eshkol development
  transport, wrong-symbol/private-export negatives, fresh archive reproducibility
  and exact closure/export/undefined manifests on the supported lane.

Authenticated-shell lifetime, G3-G prefill-before-draw admission, cache failure
rollback and public no-draw operations receive their own later aggregate tests.
Numeric primitive evidence cannot discharge those ownership or generation gates.
