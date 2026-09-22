# G3-C4: exact four-position generation proposal

**Proposed; requires integration-owner decision.** Baseline main
`ba0e37d06076d0a16c473ff742ab95723cb2cb89`. This document adds no implementation,
capability evidence or accepted contract amendment. It leaves the accepted
[C2 generation design](../G3_GENERATION_PROPOSAL.md),
[G3-T contract](G3_T_PRIVATE_CONTRACT.md), and shared implementation ownership
intact. G3-S's restricted-review hold remains in force; no sampler implementation,
review, substitute or reproduction is part of this work.

## Decision requested

Approve a separate `diagnostic-c4` generation profile and the bounded prerequisite
contracts below, followed by exact private-seam review before implementation.
The profile exists to prove repeated decode, later-token rollback/EOS and
[data-only continuation](G3_R_PERSISTENCE_PROPOSAL.md). It is not arbitrary-context
support or a performance claim. C2 acceptance can proceed independently.

The proposed additions are one explicit model constructor, one continuation
operation, one forward-only numerical provider, one shared-owner closed C4 pin
variant, and profile-specific transport/composition. No backward capability,
training model, generic tensor/provider API, X1 schema change or change to C2
checkpoint bytes is requested. The consolidated ordering and remaining freeze
points are in [the full acceptance plan](G3_FULL_ACCEPTANCE_PLAN.md).

## Profile and authenticated model

| Field | Exact proposed value |
|---|---|
| Generation config profile | `diagnostic-c4` |
| Native/model identity | `eshkol-diagnostic-byte-decoder-c4-v1` |
| Dimensions | N1, V256, D4, Hq=Hkv2, Dh2, L1, F8, C4 |
| Position table | f32[4,4], learned absolute positions 0..3 |
| Architecture | Existing 21-role pre-LayerNorm, three affine norms, epsilon bits `0x3727c5ac`, bias-free projections, exact-erf GELU, tied token/head, no dropout |
| Execution | CPU f32, dense row-major, exact accepted deterministic environment, eval only |
| Tokenizer | Authentic same-aggregate raw-byte T1 V256 with empty specials/prefix/suffix |
| Scope | Forward no-grad generation only; no C4 `model-forward`/VJP/training assertion |

Propose `generation-c4-model-create/1(seed)`, where seed is an exact nonnegative
signed-i64 integer. The constructor returns a genuine sealed P1 module in eval
mode, with a distinct private C4 model registration and native owner. There is
no arbitrary configuration input: all profile fields above are fixed. Existing
`diagnostic-model-create`, its X1 admission, M3/M3T profile identity, initializer
accessors, workspace and graph APIs remain C2-only and reject C4 model identity.
Existing P1 parameter/state/mode operations keep their own contracts; setting any
required model mode to train makes subsequent C4 generation reject, never switch
it back implicitly. The exact mode/topology check follows the canonical module
tree, not just a caller-supplied profile tag.

The C4 model has the same 15 paths/14 unique parameters and tie group as
[M3T](../M3T_TRANSPORT_PROPOSAL.md#profile-and-initialization-identity), with only
`position_embedding/weight` changing from [2,4] to [4,4]. There are **1,192 f32
values / 4,768 parameter bytes**. Index13 has 64 bytes; all other canonical indices
and shapes are unchanged. Construction validates the complete path/tie schedule
before publication and never enrolls a C4 model as an M3T native OWNER.

Reuse the existing N3K matrix initializer row [4,4] for the new position table;
no new numerical initializer row is needed. The algorithm remains
`n3k.philox4x32-10.uniform-f32.v1`; the distinct model profile defines the changed
schedule. Canonical random groups consume the existing intervals [0,288), then
position consumes [288,292), for **292 blocks** from initial counter0. Six norm
vectors use exact one/+zero and consume no draws; tied head initializes once.
Record original/successor words as private model metadata, never sampling RNG.
The same seed does not make C2 and C4 models interchangeable. No padded/copy-
extended position table or seed-only persistence proof is admitted. R-created
models instead carry private restored-origin/capsule provenance: they have no
claimed original initializer state, and C2 initializer getters reject them just
as they reject every C4 model. The C4 owner seam freeze must represent these two
construction origins explicitly without forging an initializer identity.

Successful P1 modules retain the existing process-local lifetime: there is no new
model destructor. Generator close does not destroy model parameters. Report
cumulative model/parameter identities and payloads separately. Unpublished native
construction failure must reclaim candidate payloads; staged P1 publication and
any unavoidable identity retention need explicit private-seam review and tests.
Do not promise a recoverable destructor for an already published P1 module.
The existing I2 construction bridge dispatches teardown eligibility exclusively
to pending M3T OWNER/staged_owner admission. A distinct C4 owner therefore needs
an explicitly reviewed closed C4 eligibility/revocation adapter in the same
source lineage, as part of the joint P1/I2 construction prerequisite. Passing a
C4 pointer through the M3T adapter or accepting an arbitrary registered parameter
is forbidden; C2 eligibility remains unchanged.

## Prompt, decode, output and continuation contracts

Keep the accepted eight config pairs, replacing only the profile selector and
allowing stored `:max-new-tokens`0..3. All other exact bits, token authority,
error ordering, policy, draw-dependent exhaustion and EOS rules are inherited.
`generator-create/3` selects C2 or C4 only after authentic model/profile admission.
A C4 config on C2, or C2 config on C4, rejects as unsupported before acquisition.

- Prefill accepts authentic generation input i64[1,P], P1..4, replacing cache only
  after complete candidate preparation. `generation-input-create/1` may copy
  authentic raw T1 lengths1..4 in the successor; C2 receivers still reject P>2.
- Manual decode accepts the existing scalar-created or copied i64[1,1] input,
  requires committed length1..3, and appends exactly one token at that position.
  It consumes no sampling RNG. Prefill/decode each returns independent [1,256]
  last logits and also commits an internal last-logits copy for continuation.
- `generator-generate!/2` always starts a new prompt. Before replacement require
  P+stored-budget<=4 even if EOS might occur early. Budget0 still prefills.
  Output IDs have logical shape[G], G0..3; text has exactly G raw bytes; length
  is G and cache length is P+G. EOS is included only after its model append.
- Propose `generator-continue!/2(generator,budget)`: budget is an exact integer
  0..3 with budget<=4-current-length, independent of stored max-new-tokens.
  Require a live, nonstale committed cache/history/internal logits before work.
  There is no prompt replacement or hidden replay. Return the ordinary detached
  output for only these newly emitted tokens. Budget0 returns empty IDs/text,
  unchanged cache/RNG and the current cache length, even at capacity or EOS stop.
- A nonzero continuation after a generated EOS rejects as invalid-state. A fresh
  successful prefill resets that stop state. An explicit successful manual decode
  ignores EOS matching and clears the stop state; it expresses new caller intent.
  Failed manual calls preserve it. A last history byte equal to EOS alone does
  not establish a generated-EOS stop (it may be a prompt/manual token).

Every generator retains exact committed byte history of length<=4, last logits,
policy, original prefill length P0 and EOS-stop bit, bound to its exact model identities/value bits,
profile/eval modes and tokenizer fingerprint. A successful manual prefill/decode
updates history/logits atomically with cache. A generated token commits its
history/logits, candidate RNG and stop bit in the same publication tail. Later
failure leaves all these fields at the same last successful boundary. The history
is bounded private storage, not an accessor that grants input or RNG authority.
P0 is set by each successful prefill and unchanged by appended tokens; 1<=P0<=H<=4.
Persistence replays the original P0 prefill followed by single-token appends, so
bitwise reconstruction does not rely on differently grouped schedules being equal.

## Exact missing numerical rows

The separately dispatchable [exact numerical ABI](G3_C4_N_ABI_PROPOSAL.md)
freezes the proposed operands, aliases, errors, arithmetic and source provenance.
Existing G3-N/N3K/N2/A2 rows remain frozen. Add a distinct explicit K1 provider
`et_g3c4_kernel_provider_v1`, header `g3c4_primitives_abi.h`, ABI1.0, provider and
implementation `g3c4.cpu-f32.serial`, version `1.0`, evidence identity
`G3-C4:cpu-f32-c4-forward-v1`. One object `g3c4_primitives_provider.o` supplies
only the missing rows listed below. All rows have min=max, compute f32, device
cpu, deterministic true; deterministic=false may use the same deterministic
implementation. No canonical resolver, implicit fallback, allocation, registry,
retained view or public tensor operation is introduced.

Operands, output order, alias policy, validation-before-nonfailing-invoke and
arithmetic derive from the corresponding accepted G3-N/N3K/N2/A2 operations.
New capability/operation spellings are `g3c4.`-prefixed; every operation/row pair
in each table cell is supported, no Cartesian product with another capability.
The source implementation may copy reviewed arithmetic with provenance; it may
not call unadvertised predecessor rows or mutate predecessor discovery reports.

| Capability | Operation(s) | New exact request rows |
|---|---|---|
| `g3c4.embedding-forward` | `g3c4.embedding.forward` | position [1,T,4,4] for T1,2,3,4; token [1,T,256,4] for T3,4 |
| `g3c4.linear` | `g3c4.linear.forward-no-bias` | [1,T,4,4], [1,T,4,8], [1,T,8,4], [1,T,4,256], each T3,4 |
| `g3c4.layer-norm` | `g3c4.layer-norm.forward` | [1,3,4], [1,4,4] |
| `g3c4.gelu` | `g3c4.gelu.forward` | [1,3,8], [1,4,8] |
| `g3c4.residual` | `g3c4.residual.forward` | [1,4,4] |
| `g3c4.head-layout` | `g3c4.heads.split.forward`, `g3c4.heads.merge.forward` | [1,3,2,2], [1,4,2,2] |
| `g3c4.causal-attention` | `g3c4.causal-attention.forward` | [1,2,2,T,4,2] for T1,2,3,4; uncached [1,2,2,3,3,2] |

These are seven capabilities, eight operations and **28 operation/row pairs**:
6 embedding +8 linear +2 norm +2 GELU +1 residual +4 layout +5 attention.
T4 full-prefix attention and T4 full-capacity attention share the same row.
Attention uses explicit positions/mask, learned-position projected K and no RoPE;
its arithmetic retains A2's reciprocal-then-multiply scale and serial f32 order.
The exact C4-N proposal explicitly rejects overflow in score-minus-maximum
before expf, a checked multikey-domain refinement rather than a predecessor edit.
The T3 residual already has N2 `kernel.residual`/[1,3,4]; reuse it. T1/T2 token,
linear, norm, GELU, residual and layouts retain the accepted G3-N/N3K/N2 routes.
Position lookup always selects the new [1,T,4,4] row, including single decode.
Uncached T1/T2 parity retains G3-N/A2; T3/T4 selects the rows above. No initializer,
sampler, backward, activation-sum or tied-sum capability is added.

Every input and intermediate must meet the corresponding finite/FP/alias rules.
Position embedding validates the entire [4,4] table and IDs0..3; token
embedding validates [256,4] and IDs0..255 before access. Layout
materializes true [1,2,T,2] storage; T3 and T4 must test non-self-inverse
permutations. Primitive mathematical references cover every output, marked-layout
and attention-mask cases, rejected neighbors and unchanged outputs on all errors.
A backward claim would require a new contract and analytic/numerical-gradient
proof; none is included here.

## Cache and shared pin prerequisite

A2 already admits positive capacity4 at the cache ABI layer; it does not supply
C4 attention capability rows. Use one exact cache K/V[1,1,2,4,2], lengths[1],
keep[1,4], stage[1,2,A,2], full layer views[1,2,4,2]. A=P for prefill and 1 for
decode. Materialize bool[1,A,4], query positions[1,A], keys[1,4]={0,1,2,3}.
Unused tails remain finite +zero under false mask. Close the nested view before
return from attention role10 and before another stage/commit/abort/destruction.
No short capacity-strided view, implicit broadcast, ring buffer or eviction.

The accepted shared pins fix position[2,4] and 4,736 bytes. They **cannot** be
reused for C4 by relabeling. Request the shared owner add a separate source-private
`et_g3c4_model_pins_internal` type with the same named fields and a self pointer
of its own type, and these closed helper signatures:

```c
int32_t et_g3c4_model_pins_begin_internal(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    et_g3c4_model_pins_internal *pins, et_f32_tensor_error *error);
int32_t et_g3c4_model_pins_check_internal(
    const et_g3c4_model_pins_internal *pins, et_f32_tensor_error *error);
void et_g3c4_model_pins_end_internal(et_g3c4_model_pins_internal *pins);
```

They use the exact C4 table (index13[4,4],4,768 bytes), canonical IDLE/FULL,
all 14 preflight, stable embedded sentinels, reverse rollback and fail-stop corrupt
end from the [shared contract](../E3_SHARED_CALL_CONTRACT.md). Existing C2 names,
type, shapes and evidence remain unchanged. Shared ownership must prevent a
second independent implementation of the pin mechanics; any internal factoring
needs predecessor regression proof and cannot become a caller-selected shape or
profile API. No C4 helper is installed, boxed, serialized or available via extern.

The C4 consumer authenticates its distinct native/Eshkol model and generator
registries before deriving pin arrays or embedded addresses. It uses the same
one outer `m3-call` guard and exact active-token discipline. A model stores one
active token; every later step checks the same model/generator/call links. Pin-end
clears only pins; consumer cleanup owns tokens, and the outer boundary owns its
guard. No mode-restoration authority, callback or generalized gradient lock.

## Multi-token transactions and private-seam freeze

Before the first irreversible commit, reserve one rooted output envelope and
private exact I1[G]/raw-byte[G] candidates for every G=0..budget (at most four),
plus RNG snapshot, bounded generated-ID scratch and cleanup/selection ledger.
I1 already supports zero extents. Each candidate has its actual logical shape;
there is no shape mutation, overcapacity public tensor or later shrinking
allocation. Root/promote the raw candidates before entering the commit sequence.

Nonfinal tokens commit bounded private history/generated-ID scratch, without
sealing an output or reusing C2 once-only readiness flags. For the final token
(budget or sampled EOS), copy the full candidate generated sequence into the
exact final-G candidate, decode/check its raw bytes, prepare its one-time
readiness and preflight destruction of every unused candidate before token
commit. G0 selects the preallocated empty candidate. The final tail selects and
seals that candidate and drains the unused owners without allocation or failure.
Account for all temporary candidates and retained identity controls. This closed
C4 publication protocol needs transport review, but no I1 shape-changing ABI.

Each successful token commits cache + history + private last logits + RNG +
generated count + EOS bit. Keep all 14 pins and guard across the complete call.
Between tokens, reset only per-frame role/readiness/candidate state through a
nonallocating committed-frame transition, leaving previous committed state intact.
A later candidate/view/provider failure aborts and scrubs only its tail, preserves
earlier committed tokens and RNG, releases the unpublished output, and raises
without partial output. A required draw that becomes exhausted mid-call follows
this same later-token rule; an initially exhausted required draw rejects before
prefill. No lookahead reserve of the entire RNG budget changes that rule.

Final token/EOS, G0 prefill and zero-budget continuation prepare all cleanup and
sealing before their final no-failure tail. Impossible postcommit invariants
fail-stop; recoverable failure injection belongs only at valid precommit points.
Error delivery allocations remain distinct from nonallocating cleanup.

The exact C2 29-seam/95-binding contract is not silently widened. Before any C4
transport code, submit one exact successor seam delta covering: distinct C4 owner
construction/admission and exact I2 abort-eligibility routing; profile-authenticated generator construction; input P1..4;
continue-call kind and explicit budget; repeated-frame transition; internal
last-logits/history/stop access for R; exact-size output candidate selection;
shared C4 pin consumption; and staged model restoration. Existing C2 signatures
and state meanings remain testable through their original path. Native and
Eshkol seam counts must be derived from that proposal, never guessed here.

## Required future proof

All evidence below is pending. In addition to C2 regressions and primitive gates:

1. Prefill P1/P2/P3/P4; P1+three manual decodes, P2+two, P3+one; compare each
   prefix's last logits, K/V, masks, positions and lengths to independent full
   prefix reference. Test exact cache/output/RNG repeatability separately from
   justified mathematical parity tolerance.
2. Generate every legal P/budget pair, continue at lengths1/2/3/4, G0, overflow,
   missing/stale cache and exact value restoration. EOS at first, second and
   third generated position, prompt/manual EOS, continuation stop and reset.
3. Fail once-per-call pin acquisition and each later token's allocations/views/
   preflights (pins remain held, with no repinning); distinguish
   initial-prefill commit from token1/token2 commits. Verify history, logits,
   cache and RNG all preserve the last complete boundary, with no partial output.
4. Validate exact 14-parameter initialization and 292-block successor; tied-head
   order, position2/3 sensitivity, unchanged model gradients/counts/initializer
   state during generation, every new identity/lifetime negative and C2 exclusion.
5. Independently reviewed normal versus instrumented artifacts, fresh Eshkol AOT,
   ASan/UBSan/LSan, supported exact-head CI and all inheritor regressions. Preserve
   one registry aggregate and actual source/object/ELF/public-string inventories.
6. Measure 1,024/8,192 repeated decode/re-prefill and create/close/failure loops:
   peak live payload, retained identity controls, Eshkol arena/error growth and
   RSS, with model construction counted separately. Existing I2 module provenance
   is capped at 4,096 modules (a model contains multiple modules); do not claim
   8,192 successful model constructions or hide this limit by silently shortening
   a test. Reuse a fixed model for generator loops, and measure separate bounded
   construction counts plus explicit ceiling rejection. Fixed capacity is not flat
   process memory. No long-context/throughput conclusion follows from these runs.

Together with R, this profile can discharge the roadmap's explicit repeated-
decode and save/reload requirements after implementation proof. It cannot prove
useful language quality, arbitrary batching/context, accelerators, mixed precision,
training resume, cache serialization, cross-platform bitwise identity or public
G3 completion before all dependency and integration gates pass.

## Source basis

The independent read-only row inventory agrees on28 new pairs and 4,768 parameter
bytes. This is source-derived planning, not executed C4 evidence. Sources are
[M3T shape/initialization](../../src/eshkol_transformer/m3t_transport.c),
[profile/path admission](../../native/m3t_transport_extension.esk),
[G3-N exact rows](../../native/g3n_primitives_provider.c),
[N3K rows and initializer](../../native/n3k_primitives_provider.c),
[N2 residual and other rows](../../native/n2_primitives_provider.c),
[A2 attention metadata](../../native/a2_attention_provider.c), and
[cache ABI](../../include/eshkol_transformer/a2_kv_cache.h).
