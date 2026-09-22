# G3-R model and generation-state persistence proposal

Status: **proposed; requires orchestrator decision and prerequisite implementation**.
Source audit base: `ba0e37d06076d0a16c473ff742ab95723cb2cb89`. No runtime,
wire format, capability, public API, test result or acceptance is added by this
proposal. The sampler review restriction remains unchanged; this design uses
only the accepted logical RNG contract in [G3](../G3_GENERATION_PROPOSAL.md).

## Decision requested and scope

Approve a separate `eshkol-generation-state` 1.0 capsule for the proposed
`diagnostic-c4` profile, containing an unmodified C1 model-state container,
normalized generation policy, typed generation RNG, tokenizer identity and
committed token history. Approve the two public operations below, consumption of the shared
rank-two storage successor, and the private staged-construction/replay
transaction specified here. G3-C4 must be accepted before their implementation.
This is the persistence gate for full G3; the accepted diagnostic-C2 generation
contract and C1/C2 wire formats remain unchanged.

C2 training-state persistence already exists. Its public loader authenticates
K2 capability requests before reconstruction, but its current `tensor.f32` /
`storage.copy` route admits rank zero/one only. Its optimizer, dataset,
resolved-training-config and dropout RNG fields do not represent generation
state. Reusing that envelope by inventing optimizer/cursor values, laundering
rank-two shapes through rank-one copies, or interpreting C2 RNG words as a G3
owner is prohibited. C1 remains the nested model codec, with the exact I2
provider selected by trusted artifact code.

The proposal restores into a **fresh model and generator**, never a supplied
live model. The caller supplies an authentic raw-byte T1 tokenizer. Model values
and ties are restored from bytes, not recreated from an initializer seed. The
model receives fresh local identities; "identity restoration" means the same
profile, schema, value bits, alias topology and tokenizer fingerprint, not
reusing addresses or registry identities from another process. The restored model
records private restored-origin/capsule provenance, not a fabricated initializer
state. C4 exposes no C2 initializer getter or training-resume authority.

## Public and private surface

The following two names are proposed additions to the eventual generation
facade, not changes to existing `checkpoint-load/3` or `checkpoint-save!/4`:

| Proposed call | Result and ownership |
|---|---|
| `generation-checkpoint-save!(generator, path, policy, options)` | `#t` after atomic publication and successful durability checks; snapshot an idle C4 generator and its currently bound model |
| `generation-checkpoint-load(path, policy, tokenizer)` | Fresh two-element list `(model generator)` containing authentic C4 model and generator shells; publish both together after full replay |

The eventual single Wave 3 aggregate uses the canonical C2 authenticated
`persistence-policy/5` closure, shared with training/checkpoint and tokenizer
surfaces. G3-R calls `c2-require-policy` for exact factory/entry authentication
and revalidation, and obtains nested C1 bounds only through
`c2-policy-c1-subpolicy-internal` (validated private entry slot8). Public mutable
C1 policy vectors, copied closures and foreign-aggregate identities reject.
The raw C1 projection stays private and grants no additional authority.

This requires a reviewed source-composition change from today's M3 policy
producer, preserving one public name/arity and one policy factory/registry.
Follow the existing `c2-policy-t2-entry-internal` projection installed by
`c2_public_extension.esk`; retain tokenizer predecessor tests. Never link M3/C2
owning archives together or install competing public producers. The G3 capsule
applies the lower of authenticated caller limits and its own ceilings; nested
C1 receives the private lowering projection, not the wider aggregate tensor limit.
`options` is exactly the finite flat list
`(:overwrite? boolean)`; unknown, missing, duplicate, cyclic or extra elements
reject. Path admission and atomic publication follow C1. A result list may be
copied as ordinary data; its elements gain authority solely from exact registry
membership, never list shape or serialized tags.

The returned generator is closed with the existing `generator-close!`; that
operation does not destroy the model or tokenizer. A successfully published
C4 model has P1's existing process-local lifetime, including its parameter
storage. This proposal adds no model destructor, finalizer, restore-owner or
claim that successful repeated LOAD returns model memory to baseline. Tokenizer
retention is the same as ordinary generator creation. The two-element result list is not itself an
owned releasable carrier. Detached output/RNG/tensor
clones retain the existing independent-lifetime rules.

The following **new source-private seams require owner review**; these are
semantic signatures, not already accepted ABI declarations:

| Proposed seam | Exact authority and obligation |
|---|---|
| `g3r-snapshot-internal(generator, operation)` | Authenticate idle C4 generator/model/tokenizer, exact live binding and committed history; hold shared guard and C4 parameter pins while making an independent model/policy/RNG/history snapshot; release all holds before I/O |
| `g3r-stage-internal(path, policy, operation)` | Strict bounded same-descriptor read and full inert validation; return unpublished byte staging and canonical tensor descriptors; no tensor/model/RNG construction |
| `g3r-capability-admit-internal(stage, operation)` | Obtain the genuine current K2 report privately from the single composed K2 lineage and authenticate it via the reviewed successor route below; require every exact tensor descriptor and deterministic execution lane |
| `g3r-reconstruct-prepare-internal(stage, tokenizer, operation)` | Create one unpublished C4 construction context, restore 14 independent parameter allocations with the one head/token tie, form an unpublished typed RNG and generator, replay committed history without draws, and preallocate/promote the final pair |
| `g3r-reconstruct-finish-internal(context)` | One-shot nonallocating, nonraising joint publication of model and generator after all sealing/cleanup eligibility has been established |
| `g3r-reconstruct-abort-internal(context)` | Consume unpublished ownership once; revoke pending identities, release cache/RNG/scratch/parameters and drain all independent cleanup; retain first error unless cleanup itself proves a trusted invariant defect |

The existing P1/I2 construction transaction is useful evidence of an abortable
registration mechanism, **not proof of this extended transaction**. In current
`construction-seal!`, schedule recomputation, finalization and promotion precede
the native seal inside one function. Existing G3 admission requires a live,
sealed model; it cannot replay against an unpublished model. Implementation
therefore needs an owner-reviewed split between fallible construction finalization
and the final seal, plus closed C4 candidate-value access during replay. Candidate
access must authenticate the exact current construction context and complete
14-parameter schedule; it must not publish the pending model or weaken ordinary
model admission. Whether that requires new P1/I2 entry points is an explicit
prerequisite disposition, not authorization to add slots or bypass those owners.
If the split cannot meet this contract, LOAD implementation remains gated; sealing
a model early and leaking it after replay failure is not a conforming fallback.

Current I2 construction rollback additionally routes only to the pending M3T
OWNER/staged_owner preflight in `m3t_transport.c`. A closed C4 construction
eligibility/revocation adapter is an explicit prerequisite, with exact owner,
parameter/handle and complete-collection checks. No generic callback registration,
raw-pointer trust or weakening of C2 eligibility follows from this proposal.

## Capability admission that is missing today

Root's dispatched [SHARED-R2 contract #114](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/114) specifies one successor of the existing I2
`tensor.f32/storage.copy` provider and matching K2 audit: preserve rank0/rank1,
add only rank2[2,4], [4,4], [4,8], [8,4], [256,4]. The
[shared requirement](MODEL_STATE_ADMISSION_REQUIREMENTS.md) records genuine
provenance, exact schemas, transactional lifetime and independently dispatchable
work. The frozen provider/verified-entry name and implementation are
`eshkol-transformer-f32`, version `1.1`, evidence
`I2:bounded-exact-f32-storage.copy-v2`. Root dispatched its independent owner;
implementation acceptance/merge remains required. G3 must not create an alternate
capability or parallel implementation.

C4 uses rank1[4] plus rank2[4,4], [4,8], [8,4], [256,4]. Its exact15-path/
14-unique schema still rejects position[2,4], despite that storage row being
verified for C2. Rows must be backed by actual disjoint K1 copies and genuine I2
owned/borrowed carriers. Existing higher-rank codec constructors alone are not
capability proof. A copied, foreign or stale report rejects before reconstruction.

The existing seven-facade M3/G3 lineage exposes no K2 capability facade. LOAD
therefore creates and admits its current report privately; public arity3 accepts no
caller-selected provider/report. Compose the reviewed K2 extension once over the
existing I2/P1/E1 lineage, using authentic private report/request/require calls.
Never link separately owning K2/C2 roots or use a local boolean as proof.

K2 still has 11 capability entries and 12 public operations; its I2 entry changes
from2 to 7 shape alternatives. Exact provider/callback/runtime audits and report
hash/manifests must track the successor. This intentionally enables existing C2
rank-two LOAD requests for the five rows through its genuine gate, without
changing C1/C2 bytes or admitting arbitrary rank2. Detached C1/C2 ownership is
already available; fresh C4 model replay/publication remains a separate R seam.
No report/verified flag is serialized, and storage admission grants no numerical,
sampler, model-registration or training-continuation authority.

## Exact proposed wire format

All integers are unsigned little-endian unless stated otherwise. There are no
optional sections, padding, text expressions, compression or extensions. The
file is `header[256] || metadata || nested-C1 || history || digest[32]`.

| Offset | Bytes | Field / required value |
|---:|---:|---|
| 0 | 16 | Magic bytes `89 45 53 48 4b 47 33 53 4e 41 50 0d 0a 1a 0a 00` |
| 16 | 2+2 | Major 1, minor 0 |
| 20 | 4 | Header bytes 256 |
| 24 / 28 | 4 each | Endianness 1 / checksum SHA-256 identifier 1 |
| 32 | 8 | Required-feature bits 0 |
| 40 | 8 | Exact physical file bytes |
| 48 / 56 | 8 each | Metadata offset 256 / metadata bytes |
| 64 / 72 | 8 each | Nested C1 offset `256 + metadata bytes` / complete nested bytes |
| 80 / 88 | 8 each | History offset `C1 offset + C1 bytes` / history bytes `H` |
| 96 / 100 | 4 each | Model entry count 15 / unique parameter count 14 |
| 104 / 108 | 4 each | Profile identifier 4 / context capacity 4 |
| 112 | 4 | Committed history length `H`, 1..4 |
| 116 | 4 | Sampling mode 0 greedy or 1 categorical |
| 120 | 4 | Exact binary32 temperature bits |
| 124 | 4 | Top-k, 1..256 |
| 128 | 4 | Exact binary32 top-p bits |
| 132 | 4 | Configured max-new-tokens, 0..3 |
| 136 | 4 | EOS code: 0..255, or `0xffffffff` for absent |
| 140 | 4 | Terminal latch: 0 or 1 |
| 144 / 152 | 8 each | G3 RNG version 1 / seed in 0..`INT64_MAX` |
| 160 / 168 | 8 each | Exact 64-bit counter-low / counter-high bit patterns |
| 176 | 32 | Execution-lane identity digest |
| 208 | 4 | Original committed prefill length `P0`, 1..`H` |
| 212 | 44 | Reserved zero bytes |

Metadata is the exact concatenation, without additional length prefixes or
delimiters, of ASCII `diagnostic-c4`, ASCII
`g3.philox4x32-10.categorical-f32.v1`, the 96-byte baseline T1 fingerprint,
the following 59-byte generation library identity and 71-byte compiler identity:

```text
eshkol-transformer\0
0.1.0-draft\0
eshkol-generation-state:1.0\0
Eshkol Compiler v1.3.4-evolve\0
90cbd7130f47b8184bcc77b8d5c1b0026da980de\0
```

The first three lines form the library identity; the last two form the compiler
identity example. Freeze the exact compiler bytes and all lane-record values
against the accepted upstream pin before implementation evidence; `90cbd...` is
not mandated as the final runtime. Preserve the71-byte field grammar/width; do
not truncate, pad or silently change schema if the accepted identity cannot fit.
Displayed line breaks only separate concatenated string fragments: they
are not bytes. Each `\0` is exactly one NUL, including the final fragment's NUL;
there is no extra C-string terminator. Metadata length is exactly 274 bytes;
the nested C1 begins at offset530. Identities are inert fixed-length bytes,
never code/module/library/provider selection. The generation identity is separate
from C2's57-byte `eshkol-training-state:1.0` identity; C2 bytes remain unchanged.

The execution-lane digest is SHA-256 over the UTF-8 domain
`eshkol-generation-lane-v1\0` followed by the trusted artifact's canonical
71-byte compiler identity, then 15 length-prefixed ASCII records in this exact
order: target triple, LLVM identity, CPU execution/ISA lane, operating-system ABI,
libc identity, libm identity, floating-point control contract, N2 identity, N3K
identity, A2-attention identity, A2-cache identity, G3-N identity, G3-C4 numerical
identity, G3-S identity, and G3-C4 model/transport/composition/replay identity.
The final record binds the reviewed native and Eshkol schedule/replay source
identity, not merely the numerical providers. Each record is `u32 byte-length || bytes`, length
1..256, with printable ASCII bytes 0x20..0x7e, no padding or terminator. Provider
records must include the reviewed algorithm/version and immutable evidence/source
identity; FP controls must identify rounding, contraction and denormal behavior.
These exact record values are a missing deterministic-lane manifest to be reviewed
and frozen by the provider owners before implementation; callers and files cannot
supply them. An artifact lacking that manifest must
reject SAVE/LOAD as `determinism-unavailable`. This field cannot confer authority
on a host; LOAD compares it with its independently admitted current lane.

Nested C1 is canonical C1 1.0, P1 provider interface 2.0,
`i2-dense-cpu-f32-v1`, CPU dense f32 parameters only, with zero buffers. Paths are
exactly the 14 canonical rows in `m3t-paths`, plus `token_embedding/weight` tied
to `head/weight`; only `position_embedding/weight` changes to `[4,4]`. There is
exactly one two-member alias group, and all other allocations are distinct.
All value bits must be finite, preserving signed zero. Unique values total
1,192 f32 / 4,768 bytes; C1 stores the tied entry twice, hence exactly 8,864
physical tensor-payload bytes. No gradients, optimizer, initializer RNG,
training config, cache, logits, graph, pointer or lease appears in this format.

History contains exactly `H` raw byte IDs. `P0` records the last successful
prefill length; later single-token appends preserve it, and a replacement prefill
changes it atomically with the new cache/history. It is the complete committed cache
prefix, including any emitted EOS; it is not merely the most recent output.
SAVE requires a populated cache and rejects an idle never-prefilled generator.
A stale value binding or mismatched history/cache length rejects before I/O.
Terminal latch semantics are those of the separately proposed C4 continuation
contract: a terminal latch requires `H > P0`, configured EOS and a matching
last history byte (prefill itself always clears the latch). The converse is not valid because a prompt/manual token may equal EOS.
A zero-budget continuation remains allowed at capacity or terminal stop; a
nonzero continuation at terminal stop rejects. Replay itself neither samples nor
turns a manual/prompt EOS byte into a terminal stop. The saved normalized policy
is admitted independently of current history length:
`H + configured max-new` need not fit when loading a completed request. A future
new generation/continuation request performs its own context check.

The final digest is SHA-256 over domain `eshkol-generation-state-container-v1\0`
followed by every preceding file byte, including all embedded C1 digests and
history. It ends exactly at EOF. C1's own container/tensor checksums are also
mandatory. These unkeyed digests detect accidental corruption and bind fields;
they do not authenticate a writer or make a malicious rewritten file trusted.

Fixed G3-R admission ceilings, lowered by caller policy, are 65,536 file bytes,
8,192 artifact-wide metadata bytes (outer plus nested C1 metadata), 4,096 bytes
per tensor and 15 tensor entries. Profile/schema checks further require the
exact payload/counts above. All section sums, shape products and policy
comparisons use checked arithmetic before allocation. These are proposed parser
limits for a tiny profile, not measured process memory or latency bounds.

## Validation, save and restore transactions

LOAD order is:

1. Validate current C1 policy and path, authenticate tokenizer kind/liveness; bounded same-descriptor
   regular non-symlink read, probe/retained-header agreement, hard/caller bounds,
   checked arithmetic, reserved bytes, exact physical size and EOF.
2. Verify outer digest, fixed metadata/versions, full C1 parser and all tensor
   checksums, exact C4 paths/shapes/tie/value schema, policy/RNG/history/latch
   semantics, and tokenizer fingerprint/raw-byte/no-specials policy. Serialized
   provider text is compared only to trusted fixed I2 selection.
3. Obtain and authenticate the genuine current private report and every exact
   storage-copy request; admit all generation providers and exact deterministic
   lane. No decoder callback, model construction or typed RNG publication occurs
   before all byte/semantic/capability checks finish.
4. Build one unpublished detached reconstruction transaction. Decode/copy all
   saved value bits and tie identity into fresh C4 storage; no initializer draw
   substitutes for any saved value. Construct RNG kind from validated version,
   seed and full counter bits, under the private registry only.
5. Replay history through the accepted no-grad schedule without sampling or RNG
   advancement. Rebuild cache, last logits and exact value binding. Preallocate
   result pair and all publications, establish eval mode while unpublished,
   finish fallible P1/I2 finalization, and release temporary decoded state.
6. Jointly seal/publish model and generator in a no-allocation/no-recoverable-error
   tail. Return the already rooted pair. No caller-visible partial model,
   generator or RNG exists on a recoverable error before this point.

Replay first prefills exactly the first `P0` history tokens, then appends
each remaining token using the same single-token manual-decode schedule used
originally, and finally restores the terminal latch. It must not substitute a
full-`H` prefill or assume tolerance-equal alternative schedules yield bitwise
continuation. Full-length history replays even if the saved max-new policy
would reject a new generation request. The exhausted RNG sentinel is valid on
load and no-draw replay; only a later requested categorical draw rejects it.
LOAD never calls `generator-generate!` to reconstruct history.

SAVE authenticates generator, current model, exact cache binding and tokenizer,
then requires the genuine current I2/K2 `tensor.f32/storage.copy` successor for
each exact tensor span after separate C4 profile/schema authentication. Take a
coherent independent snapshot under the shared invocation guard and C4 pins.
The closed snapshot seam copies exactly the 14 pinned value spans into independent
bounded byte storage, alongside policy/RNG/history/P0/stop metadata. It cannot
call ordinary P1/I2 parameter cloning while their values carry pin sentinels or
export borrowed descriptors. End pins after the coherent copy, then construct
and encode detached C1 state from those copied bits through reviewed typed
construction. This snapshot-copy/typed-state bridge is part of the exact R seam
freeze, not a new generic state-import authority. It changes no model/gradient/cache/RNG state. Build and strictly
self-parse complete canonical bytes before publication; release runtime borrows
before the atomic writer. No file is created on admission/encoding failure.
Use C1's same-directory temporary, checked writes, file fsync, rename commit and
directory fsync. Pre-rename failure preserves the old destination; post-rename
failure may report `published? = #t`, durability unknown. Preserve those fields;
never report that every I/O error means no publication. No network-filesystem,
concurrent-writer or power-loss guarantee is added.

Wrong/forged/cross-owner carriers are `invalid-argument`; dead/busy/no-cache/stale
binding owners are `invalid-state`; malformed bytes/checksums/schema are
`corrupt-data`; well-formed unsupported versions are `version-mismatch`; absent
capability/profile/tokenizer policy is `unsupported`; unavailable deterministic
lane is `determinism-unavailable`; allocation or trusted invariant defects are
`internal`; filesystem failures are `io` with publication details. Authentication
precedes payload access. Wire-invalid RNG/config values are `corrupt-data`, not
an opportunity to repair or clamp them. Errors use the invoked public operation
and bounded inert details, never raw pointers or executable diagnostics.

## Required evidence before full G3 acceptance

All checks below are future requirements, not executed evidence:

- Independently produced canonical fixtures and parser comparison; exact/one-over
  limits, truncation/trailing bytes, reserved/version/feature errors, overflow,
  recomputed-checksum semantic corruption, duplicate/missing/path/alias/shape
  defects, nonfinite f32, malformed RNG/history/latch (including recomputed-checksum latch1 with H=P0)
  and tokenizer mismatch.
  Invalid semantic input must invoke zero tensor constructors/codecs.
- Genuine/stale/forged/cross-aggregate private report witnesses and malformed/mutated/foreign-
  tag C1 policy checks, every exact new
  rank-two capability row, missing-row injection before reconstruction, and
  C2's genuine newly admitted exact rank-two rows plus unchanged rejection
  of unlisted rank-two shapes before decoding.
- Supported-lane fresh-process split versus uninterrupted C4 continuation with
  non-initialized model values and nonzero full RNG counter: every vocabulary
  logit bit, committed IDs, raw output bytes, generated/cache lengths, terminal
  state and final RNG words agree. Exercise split after different committed
  tokens, carry, exhausted no-draw restore, final consumable draw and EOS. Sampler
  evidence must come from its accepted provider after its restriction is lifted;
  do not reproduce restricted review or replace it with a mock implementation.
- Show manual/replay paths consume no draws and preserve gradients; full-capacity
  history restores; partial history continues through at least two appended
  tokens. Verify exact cache-binding identity/value checks after restore and
  rejection/recovery after model mutation/exact restoration.
- Fail every allocation/codec/finalization/replay cut before final publication.
  Existing model/tokenizer/generators remain unchanged; unpublished payloads
  are reclaimed and retained identity/control costs are counted separately; ambiguous corrupted invariants are fail-stop, not invented
  recoverable rollback. Verify no allocator work in valid final publication and
  prove same-parent-region identity promotion.
- Repeat SAVE and failed LOAD at 1,024/8,192 horizons, measuring live storage,
  cumulative controls/tombstones, arena bytes and RSS separately. Successful LOAD
  intentionally retains a fresh model per call; quantify growth and the inherited
  I2 module-provenance limit instead of promising flat memory or hiding a smaller
  arbitrary iteration cap. Do not claim an 8,192-successful-model-load gate when
  the existing provenance ceiling prevents it.
- Two independent fresh source-composed AOT artifacts with exact source/native/
  symbol/facade closure, private-link negatives, supported sanitizers and the
  orchestrator's accepted exact-tree CI. One P1/I2/T1/K2/E1 registry lineage;
  never stitch separately owning C2 and M3 archives together.

C4 numerical rows, shared C4 pin admission, generator continuation, deterministic
lane manifest, shared I2/K2 storage successor and staged P1/I2 construction
are explicit dependencies. This document authorizes none of their implementations
by itself. G3 completion remains blocked until the orchestrator accepts the
contracts, merges those dependencies and accepts the complete supported evidence.

## Source basis

- [Accepted G3 proposal](../G3_GENERATION_PROPOSAL.md): separate persistence gate,
  typed RNG and manual no-draw replay; [G3-T](G3_T_PRIVATE_CONTRACT.md): authentic
  owners, exact model binding and unchanged C2 geometry.
- [C1 policy validator](../../internal/c1/lib/transformer/persistence_policy_internal.esk):
  current tagged data-policy admission, distinct from
  [C2 closure policy](../../native/c2_persistence_policy_extension.esk).
- [C1 format](../CHECKPOINT_FORMAT.md): non-executable nested model, checksum and
  ownership ledger, alias checks and atomic I/O; [C2](../C2_TRAINING_STATE.md):
  detached training owner and current public rank-two capability limitation.
- [Canonical C2 policy](../../native/c2_persistence_policy_extension.esk) and
  [public compositor](../../native/c2_public_extension.esk): one authenticated
  policy and private C1/T2 projection, authentic report
  requests before reconstruction; [K2 native audit](../../native/k2_capabilities.c):
  exact eleven-row runtime and report generation admission.
- [M3T model composition](../../native/m3t_transport_extension.esk): exact paths,
  14 handles/15 entries, one tie, current C2-only construction and abort/seal.
- [P1 construction](../../internal/p1/lib/transformer/module.esk) and
  [I2 construction/codec](../../native/i2_wave2_root.esk): staged registration,
  abortable ownership, current fallible seal and higher-rank codec construction;
  these are source capabilities, not proof of the proposed composed transaction.
