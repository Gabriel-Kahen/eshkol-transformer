# Full G3 acceptance: consolidated contract decision

**Proposed planning package, not implementation authorization or completion.**
Prepared from main `ba0e37d06076d0a16c473ff742ab95723cb2cb89` under the new
Wave 3 integration owner's explicit G3-C4/G3-R planning dispatch. Only that owner
can accept these additions, consolidate shared prerequisites, dispatch dependent
implementation and accept full G3. Existing accepted C2/N/S/T decisions stand.

## Concrete decision package

Accept the following bounded extension direction, or disposition individual rows.
The detailed proposals make their profile, rows, public semantics and wire fields
reviewable now; the last column names prerequisite contracts that still need an
exact owner-reviewed seam/report freeze before any implementation uses them.

| Decision | Proposed boundary | Required follow-through |
|---|---|---|
| [G3-C4](G3_C4_PROFILE_PROPOSAL.md) | Separate learned-position C4 profile; P1..4, G0..3, P+G<=4; repeated single decode; 14 parameters/4,768 bytes; constructor and continuation API | Root acceptance of the [exact C4 model/transport seam](G3_C4_MODEL_TRANSPORT_CONTRACT.md), accepted shared-owner C4 pins and construction rollback admission |
| [C4 numerical prerequisite](G3_C4_N_ABI_PROPOSAL.md) | One distinct registry-free forward provider: 7 capabilities, 8 operations, 28 missing row pairs; retain all frozen predecessor reports | Exact ABI/header/source arithmetic/error/metadata review and independent numerical implementation proof |
| [G3-R](G3_R_PERSISTENCE_PROPOSAL.md) | C4-only data capsule containing unchanged C1 model bytes, policy, typed RNG, tokenizer fingerprint, history, prefill partition and EOS stop; fresh model+generator restore | Shared I2/K2 storage admission, staged P1/I2 prepare/seal/abort and closed candidate replay |
| Shared persistence | C4 exact rank1 [4], rank2 [4,4]/[4,8]/[8,4]/[256,4]; TR3 C2 additionally needs [2,4] | Root-directed I2 tensor.f32/storage.copy successor plus K2 audit; five exact rank2 alternatives, no alternate capability; [SHARED-R2 #114](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/114) freezes version1.1/evidence-v2 and dispatches one owner; implementation acceptance pending |
| Deterministic continuation | Same original prefill width then single-token append replay, no random draw; exact supported-lane identity | Freeze reviewed execution-lane manifest values and prove fresh-process continuation; never infer bit equality from tolerance parity |

No C1/C2 serialization bytes, X1 keys, M3 C2 model
identity, A2 cache ABI or shared C2 pin signatures are changed by this package.
The explicit shared storage proposal adds five rank-two alternatives to the
existing I2 capability and updates K2 provenance; all other rows remain unchanged.
Shared implementation work belongs to its designated owners. Proposed public
calls below are additive within `transformer.generation`:

| Call | Arity | Purpose |
|---|---:|---|
| `generation-c4-model-create` | 1 | Fixed-profile model from an exact seed; no generic config or model destructor |
| `generator-continue!` | 2 | Generate a bounded suffix from committed cache/logits/RNG, without replacing prompt |
| `generation-checkpoint-save!` | 4 | Snapshot an idle C4 generator/model to a bounded atomic data-only file |
| `generation-checkpoint-load` | 3 | Strictly validate and reconstruct fresh `(model generator)` using supplied tokenizer |

The accepted fifteen C2 generation names retain their arities. If these are the
only new public names, the projected generation successor is eight facades,
106 boxed exports and 112 globals (M3 87/93 +15 accepted +4 proposed). This is
an arithmetic budget, not a measured/frozen artifact inventory. K2 admission is
private in G3-R: compose its reviewed extension once over the existing registry
lineage, without an extra public report facade or a second owning aggregate.
The single eventual Wave3 lineage also uses C2's authenticated
`persistence-policy/5` and its reviewed private C1/T2 projection; replace the
current M3 producer in that successor, never expose conflicting same-name policies.
Actual source/native/member/defined/undefined/name inventories must be reviewed
on the implemented successor and reflect all merged prerequisite changes.

## Dependencies and order

The baseline already includes G3-N acceptance and the exact shared/E3 contracts.
After baseline preparation, the integration owner reported shared PR #105
accepted and merged as `19f404cf21632944e1f2d5d4959e1240f9a79f5f`, tree
`d5e8a801eeaab3ea666f37dfd8d83daaf473f379`, with proportional merged verification
underway. That is attributed integration status, not a run performed here.
The sampler PR #96 restricted-review hold is unchanged. It may not be inspected,
reproduced, rerouted, mocked or bypassed to advance this plan. Draft PR #111
remains the separate C2 G3-T readiness update; this proposal does not amend it.

| Stage | Scope and blocking inputs | Acceptance means |
|---|---|---|
| Existing N/S/shared gates | Accepted numerical/typed sampler implementations and common C2 pins, each independently approved/merged; sampler hold resolved only through authorized process | Prerequisites only |
| C2 T → M → G | Exact private transport, no-grad schedule, then public composition; follow accepted gates/explicit dispatch | Bounded C2 generation only |
| C4 numerical | Approved 28-row contract and merged numerical foundations | Missing C4 forward rows, no model/generation claim |
| Shared C4 ownership | Approved closed C4 pin variant plus staged model eligibility/revocation and P1/I2 finalization split | Safe exact C4 model ownership; no public generation proof |
| C4 transport → M/G extension | Accepted model/numerical/pin dependencies plus C2 base; exact C4 seam delta reviewed before code | Repeated decode/continuation, later-token failures/EOS within C4 |
| Shared persistence admission | [Shared storage requirement](MODEL_STATE_ADMISSION_REQUIREMENTS.md), exact profile subsets and authentic I2/K2 successor evidence | Model-state copy/restore admission only, not trajectory equivalence |
| R compositor | All above plus accepted staged unpublished replay and deterministic-lane manifest | C4 data-only save/reload and no-draw cache reconstruction |
| Full G3 review | Every accepted public operation and all evidence rows below on one supported artifact | Generation roadmap obligations satisfied only within explicitly accepted C4 scope |

Independent contract/primitive work may progress before C2 composition finishes;
no dependent runtime may use an unmerged or invented API. No shared persistence
implementation, new source registration, sampler dispatch or full CI is authorized
by this planning document. The integration owner assigns those stages separately.

## Full acceptance evidence matrix

Every row is **pending** until the implemented aggregate executes it. Existing
primitive evidence can be cited for its exact scope; it is not consumer proof.

| Obligation | Discriminating future evidence |
|---|---|
| Reachable installed surface | Fresh public Eshkol AOT exercises all 19 generation operations, exact arities/options, owning aggregate and negative private imports/links |
| Numerical correctness | All28 new pairs, independent full-model references, all prefix lengths1..4 and every vocabulary logit; cached versus uncached parity, marked head layout, position2/3 and causal masking; justified tolerance distinct from exact repeatability |
| Actual no-grad composition | Closed21-role Eshkol schedule, actual provider rows, no retained graph/gradient plan; model gradient contents/presence/counts/weights and initializer metadata unchanged |
| Repeated decode and EOS | Every legal prompt/budget, two and three appends, EOS at each generated position, prompt/manual EOS, zero budget, capacity stop and continuation stop/reset |
| Later-token failure | Initial prompt commit and each token commit distinguished; failed token preserves exact prior cache/history/last-logits/RNG/EOS state, emits no partial output; exact-size preallocated final output |
| Typed RNG | Use only the eventual independently accepted provider; no draws on manual/replay/G0, exact continuation/carry/exhaustion and immutable detached snapshots; no sampler review substitution |
| Persistent trajectory | Fresh-process SAVE/LOAD split at different prefixes/partitions, altered finite model values, nonzero RNG, exact logits/IDs/bytes/lengths/RNG/EOS equivalence after reconstruction and continued generation |
| Parser/capability boundary | Checksum plus recomputed-checksum semantic corruption; no decoder/owner construction before full byte/schema/capability admission; no serialized pointer/provider authority; C1/C2 bytes preserved; exact rank-two admission follows shared#114 |
| Atomic ownership | Allocation/codec/view/replay/finalization fault cuts, no leaked unpublished model payload, exact active-token cleanup, nonallocating commit tails, independent output clones, authenticated tombstones |
| Resource honesty | Fixed-model reuse at 1,024/8,192; live payload versus cumulative identities/arena/error growth/RSS; successful model LOAD retains process-local modules and reaches existing provenance ceiling explicitly |
| Integration | Normal/instrumented closures separate, one registry lineage, full predecessor union, fresh reproducible packages, ASan/UBSan/LSan and supported exact reviewed/tested/merged tree evidence |

## Limits and handoff

C4 is deliberately the smallest extension that can execute multiple appends and
fail after an earlier generated token. It supplies no arbitrary context, long
conversation, language quality, sliding window, batching, accelerator, mixed
precision, cache-file compatibility, training resume or cross-platform bitwise
claim. Persistence restores a generator trajectory, not optimizer/dataset/trainer
state. C2 training checkpoint semantics stay with TR3.

The source audit found an existing 4,096-module provenance ceiling and no published
P1 model destructor. Fixed-model generator loops can measure8,192 iterations;
8,192 successful fresh model LOADs cannot be promised under that ceiling.
Report actual successful model counts and explicit rejection, model payload
retention, temporary output candidates and parser staging. No flat-memory claim
or hidden smaller test loop substitutes for these measurements.

This package is ready for profile/public/wire-direction decisions. It deliberately
identifies the remaining exact seam/report freezes: C4 native/Eshkol transport;
shared C4 pins and construction eligibility; P1/I2 prepare/seal/abort plus candidate
replay; accepted implementation/report provenance for dispatched SHARED-R2#114; canonical C2-policy/C1-projection composition; reviewed
lane-manifest and compiler-identity values after the accepted upstream repin; final package inventories. Approval of the direction must
not be recorded as approval or implementation of those unresolved prerequisites.
