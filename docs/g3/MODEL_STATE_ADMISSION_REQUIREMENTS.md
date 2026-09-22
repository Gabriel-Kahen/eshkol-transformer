# Shared C2/C4 model-state admission requirements

**Consumer requirement record for root's dispatched
[SHARED-R2 contract #114](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/114);
implementation remains pending acceptance/merge.** Source baseline `ba0e37d`. This is the G3 consumer requirement for one shared prerequisite, not a parallel provider or
owner assignment. C1/C2 bytes and the twelve public K2 operations stay unchanged.

## Concrete shared successor

Extend the actual I2 `tensor.f32` / `storage.copy` provider and its matching K2
provenance audit. Preserve existing rank-zero and rank-one support, and add only
these exact rank-two min=max alternatives:

| Shape | Exact bytes | C2 model role | C4 model role |
|---|---:|---|---|
| [2,4] | 32 | learned position | not a C4 position shape |
| [4,4] | 64 | attention weights | attention and learned position |
| [4,8] | 128 | FFN down | same |
| [8,4] | 128 | FFN up | same |
| [256,4] | 4096 | tied head/token | same |

Rank1[4] already covers all six norm vectors. These rows cover every C4 model
parameter and C2 parameter plus corresponding optimizer moments. No new
`generation.persistence` capability, composite provider, resolver registration or
public report route is needed. Root froze provider/verified-entry name and implementation
`eshkol-transformer-f32`, version `1.1`, evidence
`I2:bounded-exact-f32-storage.copy-v2`; consumers require the accepted genuine
successor artifact, never copied metadata. Existing ABI symbols stay unchanged.

Preserve storage semantics: one read-only input and one disjoint caller-owned
output with identical admitted rank/extents and exact byte spans; true f32/cpu,
dense-row-major/offset0, natural alignment and checked spans. Validate complete
K1 request/table/descriptor prefixes, shapes, bytes and aliases before any write.
Full K1 generic admission and provider validation precede nonfailing synchronous
bit-copy invoke. There is no allocation, retained view, new owner, arithmetic,
shape flattening or metadata relabel. Preserve every bit, including signed zero
and NaN payloads; numerical model finite-value admission is a separate consumer
check. Reject all unlisted rank-two shapes and every rank>=3; preserve existing
rank0/1 behavior and error categories/precedence. Deterministic=false may select
the same deterministic copy.

K2 must audit the actual successor provider's exact metadata and callback
provenance, authentic runtime/process/generation, report/request closures and
K1 require result. The runtime still has its original 11 capability entries:
one entry gains five rows (seven shape alternatives total), not another named
capability. Verify its new exact report bytes/hash, frozen provider version/evidence,
source/symbol manifests and all malformed-metadata negatives. Preserve all other
entries and all twelve public operations. Report contents or file `verified`
flags alone never confer authority. Old runtime/report identities cannot be
accepted as the new verified evidence.

This **intentionally enables** C2's existing genuine per-tensor
`tensor.f32/storage.copy` LOAD gate for these rank-two shapes. It does not change
C2 format bytes, bypass that gate or admit arbitrary rank-two tensors. G3-R uses
private current K2 report/request/require operations in the same owning aggregate;
no new public K2 facade is requested by G3.

## Dispatch boundary

| Work | Available base / actual dependency |
|---|---|
| I2 provider and K2 audit successor | Dispatched by root using merged K1/I2/K2; no G3 model/sampler, P1 mode-restoration or C4 numerical runtime needed |
| Detached C1/C2 state load/save | Existing higher-rank codecs, P1 states and ownership already support it; verify real rank-two copies and new capability admission rather than inventing another lifetime ABI |
| Genuine C2 model/state/tie and 28 O2 moments | Existing M3/P1/O2/C2 APIs; exact15-path/14-unique model plus per-parameter moment shapes; root's shared acceptance gate |
| C4 schema/storage witness | Change only position descriptor to[4,4] in an inert/reference state; proves storage/schema coverage, not an authenticated live C4 model |
| Fresh C4 model+generator replay/publication | Later G3-C4/R consumer work: distinct owner, closed construction eligibility/revocation and fallible-finalization/nonfailing-seal split; not a blocker for detached state/provider work |
| Full generation continuation | Later accepted C4 numerics/transport, sampler, exact same-partition no-draw replay and supported-lane proof |

No generic upstream mutation or duplicate implementation is authorized here.
Root has assigned one independent shared owner through#114; the G3 task only
consumes its accepted merged artifact. Shared TR3 state semantics remain separate
from the generation capsule.

## Exact model provenance and tie schemas

Both profiles have15 logical paths/14 unique parameters, in canonical
[M3T order](../M3T_TRANSPORT_PROPOSAL.md#profile-and-initialization-identity):
key/output/query/value[4,4]; down[4,8]; up[8,4]; norm1/norm2 bias/weight[4];
head[256,4]; final bias/weight[4]; position[C,4]; token_embedding[256,4].
Only head/weight and token_embedding/weight alias; every other unique parameter
has distinct genuine identity/storage. Reject missing/extra/duplicate paths,
unexpected ties, partial aliases, inconsistent alias bytes and profile spoofing.
A six-shape storage set is not an authenticated model schema.

C2 position[2,4] gives1,184 unique values/4,736 bytes; C4 position[4,4] gives
1,192/4,768. C1's per-entry payload includes the tied4,096 bytes twice:8,832
physical tensor bytes for C2 and 8,864 for C4. C4 rejects position[2,4] despite
that storage row being verified. Check the exact profile/path schema before
numerical model admission; ordinary detached C2 state loading retains its own
schema contract and does not thereby become a live model.

The trusted consumer chooses its compiled provider and profile table. File
strings remain inert comparisons. Authenticate a live receiver through its own
registry before payload/parameter access, then verify actual path/tie/native
handle bindings. No P1 slot, generic extension registration or source callback
authority follows from the storage capability.

## Transactions, lifetime and evidence

Validate the complete inert container, checksums, bounds and declared aliases,
then require each staged tensor's genuine capability before codec construction.
Each decoded carrier/state has one cleanup owner. Preserve declared alias
storage; do not deduplicate equal independent tensors. Failed preparation
reclaims unpublished payloads, preserves first error and accounts for retained
identity/control costs. No borrowed pointer may escape its synchronous region.

A model receiver's all 14 preflight and allocation must finish before its existing
atomic value commit, retaining parameter identities/topology/tie. Trusted commit
invariant defects are fail-stop. Joint optimizer/gradient/mode/cursor/RNG restore
belongs to TR3 and is not implied by bit-copy or detached checkpoint success.

G3 SAVE additionally needs a closed exact-span copy while C4 pin sentinels are
held, followed by detached typed-state encoding after unpinning. Fresh G3 LOAD
needs the unpublished C4 construction/replay transaction in
[G3-R](G3_R_PERSISTENCE_PROPOSAL.md). Current I2 rollback admits pending M3T OWNER/
staged_owner, not a distinct C4 model; current P1/I2 seal includes fallible work.
Those are **consumer-specific later seams**, not new blockers for the shared
provider or existing detached C1/C2 LOAD.

Required shared evidence: actual disjoint K1 copies for all five new rank-two
rows and rank0/1 regressions, genuine I2 carrier/borrow ownership, every invalid
neighbor/alias/metadata path and unchanged destination, exact authentic K2 report
and require calls, stale/forged/foreign reports, genuine15-path/14-unique M3/C2
tie with 28 O2 moments, and unsupported shape rejection before any codec callback.
Use reviewed fresh AOT/package manifests and supported sanitizer/CI evidence;
no numerical model, trajectory or sampler claim follows from this gate.

Existing P1 models have process-local lifetime and I2 caps module provenance at
4,096. Fixed-model/storage reuse can measure1,024/8,192 loops; fresh published
model LOADs accumulate storage and eventually hit that ceiling. Report live
payload, cumulative controls, arena/error growth and RSS separately. No destructor,
flat-memory claim or hidden shorter test loop is introduced.

Sources: [I2 provider](../../native/f32_tensor.c),
[K2 audit](../../native/k2_capabilities.c),
[C2 capability-before-codec route](../../native/c2_public_extension.esk),
[C1 format](../CHECKPOINT_FORMAT.md),
[I2 construction bridge](../../native/i2_wave2_package_bridge.c),
[M3T pending-owner checks](../../src/eshkol_transformer/m3t_transport.c).
