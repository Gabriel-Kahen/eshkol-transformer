# M3 preflight: merged decoder integration audit

Status: **audit complete; prerequisite proposal awaiting integration decision**.
Tracking: [#65](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/65),
parent [#1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1).
Audited base: `231f9354f14389db15faac7820ef23dc038ae941`.

This is a fresh audit of merged source and executable evidence. It does not accept
an implementation plan, freeze public APIs/formats/versions, or complete M3.
Deleted earlier code, issues, and proposals were not inputs. O2/D2 unmerged APIs
were not consumed; C2 and the trainer remain downstream. No numerical provider,
K1 metadata, upstream Eshkol, or production package was modified.

## Decision requested

Accept or revise the bounded profile and prerequisite boundaries below before
implementation is dispatched. Current merged code cannot execute a byte decoder:
N2's largest verified embedding vocabulary is 5, and public Eshkol model transport,
head permutation, initialization, VJP orchestration, and transient ownership are
absent. Changing dimensions cannot remove the byte-vocabulary blocker.

The proposed profile deliberately uses an already verified A2 attention row.
There is no demonstrated need to change A2 numerical semantics, K1 ABI, I2 provider
identity, or upstream compiler semantics for this profile. Those would require
separate evidence and approval through integration, not inference from this audit.

## Minimal integration profile (proposal)

This is an exact **teacher-forced forward/analytic-VJP acceptance profile**, not a
useful language-model quality target, arbitrary-shape API, generator, or trainer.

| Property | Proposed value |
|---|---|
| Inputs and targets | exact `i64[1,2]`, values 0 through 255 |
| Vocabulary/tokenizer | 256 bytes, T1 with no configured specials |
| Batch/sequence/context | N=1, T=2, context capacity 2; only exact T=2 admitted initially |
| Layers/hidden/heads | L=1, D=4, Hq=Hkv=2, Dh=2 |
| Block | pre-LayerNorm, separate bias-free Q/K/V and output projections |
| FFN | bias-free 4→8, exact N2 GELU, bias-free 8→4 |
| Positions | learned `f32[2,4]`, positions `i64[1,2]=[0,1]` |
| Attention mask | owned canonical `bool[1,2,2]` all true; A2 enforces causal positions |
| Normalization | three affine LayerNorms including final norm; proposed epsilon `1e-5` rounded once to f32 |
| Head | bias-free, same canonical `f32[256,4]` parameter as token embedding |
| Precision/device | true dense zero-offset row-major CPU f32; serial deterministic admitted FP environment |
| Randomness | explicit seed 1729 for diagnostic fixture; no dropout or stochastic forward |
| Initialization | proposed bounded uniform random matrices/embeddings in [-1/32,1/32); gamma=1, beta=+0; exact algorithm/draw mapping requires review |
| Loss extension | L2 per-token `f32[1,2]`; scalar mean and mask/seed composition require the reduction prerequisite below |
| Outputs | logits `f32[1,2,256]`; explicit model-owned saved-primal/VJP workspace |

There are 14 unique parameter tensors and 1,184 f32 values: token embedding 1,024,
position embedding 8, Q/K/V/output matrices 64, FFN matrices 64, and three gamma/beta
pairs 24. The head is an alias, not a fifteenth parameter or second initialization.

The proposed forward schedule is Eshkol-authored control flow over reviewed tensor
operations/native kernels:

```text
x = token_embedding(ids) + position_embedding(positions)
u = LN1(x)
q,k,v = split_heads(Wq(u)), split_heads(Wk(u)), split_heads(Wv(u))
a = merge_heads(causal_attention(q,k,v,positions,positions,keep))
r = x + Wo(a)
s = LN2(r)
y = r + Wdown(GELU(Wup(s)))
logits = tied_embedding_weight(LNfinal(y))
```

T=1, N>1, larger contexts, MQA/GQA variants, dropout, RoPE, KV cache, incremental
sampling, optimizer updates, corpus streaming, and checkpoint resume are outside
this exact profile. A context capacity of 2 does not prove every shorter decoder
shape. Fixed byte triples can provide shifted inputs/targets for development
without consuming D2. Model quality/overfit/resume gates remain later work.

## Required operations versus merged evidence

All numerical rows below request `cpu`, compute `f32`, deterministic=true. “Pass”
means actual K1 capability admission by an explicitly discovered production native
provider; it does not mean public Eshkol reachability or a composed model run.
The row probe checks both forward/backward, with 31 assertions including biased
linear alternatives. The exact operation tables remain authoritative in
[N2](N2_PRIMITIVES.md), [A2](A2_ATTENTION.md), and [L2](L2_INDEXED_CROSS_ENTROPY.md).

| Required operation | Semantic request / operand shape | Actual merged coverage and probe result | Production public Eshkol reachability |
|---|---|---|---|
| Token embedding fwd/VJP | `[1,2,256,4]` | N2 exact rows `[1,1,1,1]`, `[1,4,3,2]`, `[2,3,5,6]`; **unsupported** | No numerical facade |
| Learned position embedding fwd/VJP | `[1,2,2,4]` | Same embedding rows; **unsupported** | No numerical facade |
| Q/K/V and attention output linear fwd/VJP | `[1,2,4,4]`, no bias | N2 exact rows `[1,1,1,1]`, `[1,2,3,4]`, `[2,3,4,5]`; **unsupported** | No numerical facade |
| FFN up/down fwd/VJP | `[1,2,4,8]`, `[1,2,8,4]`, no bias | Same linear rows; both **unsupported** | No numerical facade |
| Tied head fwd/VJP | `[1,2,4,256]`, weight `[256,4]` | Same linear rows; **unsupported** | No numerical facade |
| Three LayerNorms fwd/VJP | `[1,2,4]`, gamma/beta `[4]`, scalar epsilon | Exact N2 row **passes** (other rows `[1,1,1]`, `[1,1,3]`, `[2,2,4]`) | No numerical facade |
| GELU fwd/VJP | `[1,2,8]` | N2 `[1,1,1]`, `[1,1,8]`, `[1,2,5]`, `[2,2,4]`; **unsupported** | No numerical facade |
| Embedding addition, residuals, activation gradient sums | `[1,2,4]` | N2 residual `[1,1,1]`, `[1,3,4]`, `[2,2,4]`; **unsupported** | No numerical facade |
| Split/merge heads and inverse VJP | dense `[1,2,4]` ↔ `[1,2,2,2]`, token-major ↔ head-major | No accepted permutation operation; marker probe proves reshape/relabel is wrong | No carrier reshape/permutation facade |
| Causal attention fwd/VJP | `[N,Hq,Hkv,Tq,Tk,Dh]=[1,2,2,2,2,2]` | Exact A2 row **passes** | Private test transport only |
| Tied parameter gradient sum | two `[256,4]` → one `[256,4]` | No general add row; hypothetical residual `[1,256,4]` explicitly **unsupported**; requires reviewed rank adaptation or bounded add | I2 contribution seam private; duplicate destinations reject |
| Indexed loss fwd/VJP | `[1,2,256]`; targets/upstream `[1,2]` | L2 rank 3, each extent 1..1,664,510; **passes** | Private test transport only |
| Scalar masked mean / upstream seed | loss/mask `[1,2]` → `f32[]`, finite positive denominator | L2 explicitly has no mask/reduction; no accepted general reduction/seed operation | Absent |
| Random matrix fill and constant initialization | unique shapes `[256,4]`, `[2,4]`, `[4,4]`, `[8,4]`, `[4,8]`, `[4]` | N2 Philox supports dropout only; no random-fill contract; I2 exact-bit ingress can supply constants | Absent |
| Owned f32 intermediates/parameters | rank 0..64; model shapes above | I2 C carrier admits storage/borrows; K1 copy evidence is only rank 0/bounded rank 1, not these numerical shapes | I2 construction/gradient seams localized |
| Exact IDs/positions and bool keep mask | `i64[1,2]`, `bool[1,2,2]` | I1 exact native storage exists; T1 sealed outputs are rank 1; no general owned bool carrier | No public T1→model borrow/reshape adapter |

I2 storage ownership must not be confused with its narrow K1 `storage.copy`
capability. A higher-rank allocation/borrow is allowed by the carrier contract and
already exercised in N2/I2 native integration; it does not require advertising
higher-rank K1 copy or `tensor.contiguous`. Likewise, analytic provider VJPs do not
establish `autodiff.reverse` through the Eshkol compiler.

## Concrete composition and ownership requirements

### Layout, dtypes, masks, and input storage

N2 linear output indexes `[n,t,h*Dh+d]`; A2 indexes `[n,h,t,d]`. For two tokens and
two heads, a marked tensor has 10 at flat index 2 while the head-major destination
requires 100. Equal byte counts do not make these layouts interchangeable. A
reviewed materializing permutation and its inverse VJP are needed; generic K1
allows only dense zero-offset views and I2 borrows cannot be silently relabeled.
Separate Q/K/V projections avoid adding a fused split schema.

A2 requires exact mask bytes 0/1 shaped `[N,Tq,Tk]`, even for an all-keep policy.
It performs causal exclusion using exact i64 positions. No f32 mask cast,
additive mask, or implicit broadcasting is available. A bounded model-owned mask
buffer and exact-I1 input adapter are sufficient candidates; a general bool tensor
subsystem is an alternative, not a demonstrated prerequisite. Do not reinterpret generic numeric vectors as exact-i64 storage or route IDs
through floating conversion; a separately reviewed exact-integer copy bridge remains
an alternative.
T1's encoded tensor identity has a process-retained active borrow and no public
model-oriented read/reshape/release seam. A synchronous trusted validated copy into
model-owned I1 storage can be reviewed without changing T1 bytes or exposing raw
pointers. Fixed diagnostic inputs may be created once; repeatedly encoding every
training batch would accumulate T1's documented process-lifetime storage.

### Analytic backward and canonical ties

Reverse the explicit forward schedule using the providers' unnormalized analytic
VJPs. Each primitive overwrites disjoint caller-owned output buffers and retains
no forward state. Numerical finite differences are development oracles only.
The model must combine Q/K/V input gradients in fixed edge order, add residual
skip/branch contributions, reverse layout permutations, and combine position/token
embedding paths. All intermediate sums must use reviewed f32 tensor operations or
native kernels, not recursive scalar Eshkol loops.

Embedding weight `[V,D]` already matches linear head weight `[Dout,Din]=[V,D]`.
No weight transpose or duplicated parameter is required. Sum head and embedding
VJPs in a declared order into one `[256,4]` contribution, then stage exactly one
entry for that canonical P1/I2 handle. Repeated IDs are handled by embedding VJP's
scatter accumulation. Submitting tied destinations twice in one I2 batch rejects;
submitting them as two microbatch contributions incorrectly advances contribution
count/normalization. Prepare the complete finite unique-handle batch and commit
once only after all VJPs succeed.

L2 returns per-token loss. A scalar A0 loss requires
`sum(mask*loss)/sum(mask)` with f32 accumulation and positive finite denominator.
For the all-ones two-token training numerator, seed L2 with upstream `[1,1]` and
submit I2 normalization increment 2; do not seed `[1/2,1/2]` and also normalize by
2 later. Scalar-mean gradient parity compares the accumulated numerator divided by
2. A separately callable mathematical VJP of the mean can use `[1/2,1/2]`, but its
contract must distinguish that result from the accumulation numerator. Integration
must choose that boundary before M3 backward is exposed. Logits-only VJP parity
can precede the loss extension; full `model-output-loss` acceptance cannot.

### Random initialization and reproducibility

No merged initializer produces all model tensors from `run.seed`. Dropout's private
Philox implementation/state layout is evidence for dropout, not a public RNG or
random-fill contract. Review a bounded initializer using explicit immutable
seed/state, canonical unique-parameter path order, an exact integer-to-f32 mapping,
versioned algorithm identity, and explicit successor state. A concrete candidate
reuses the documented N2
Philox4x32-10 arithmetic with key=seed, counter=0, little-word lane order,
`u=(float)(lane>>8)*0x1p-24f`, and `w=(u-0.5f)*0x1p-4f`, consuming complete
blocks per canonical unique matrix and discarding tail lanes. This is an initializer
proposal requiring independent bit/overflow/continuation tests, not permission to
call a private dropout helper or reuse its version identity. Tied storage consumes
one draw sequence. Constants need a reviewed production path; existing I2 exact-bit
ingress can supply gamma/beta and tiny upstream constants without a new fill kernel.
Uniform weights
avoid making a normal-distribution transform necessary for this minimal profile;
normal/Xavier initialization is an alternative with additional numerical evidence.

No-dropout forward returns the unchanged supplied RNG (or `#f` when absent), per
A0. Initialization RNG and future stochastic-forward RNG need explicit separation
and recording for later C2; no checkpoint schema is selected here. Python-produced
weights are valid test fixtures but cannot establish runtime initialization.

### Output and tape lifetime

A0 requires owned logits and retained forward graphs, but provides no model
constructor, backward invocation, or output/tape release operation. The installed
model module is absent. These are model/A0 design decisions, not authority to
export private I2 pointers or reuse P1 state release as a generic destructor.

The native probe demonstrates that ending a parameter borrow and applying a valid
I2 update changes primal bytes while preserving parameter identity and storage
address. Retaining a handle alone cannot validate a delayed tape. Keeping the borrow
open also blocks a second borrow and mutation. Two viable choices require review:

1. Save detached primal tensors plus needed exact IDs/positions/targets/masks/RNG in a
   model-owned bounded frame, with borrows opened/ended inside each dispatch. This
   gives stable forward values at explicit copy/memory cost. Before gradient commit,
   a reviewed synchronous bit/shape comparison against all current canonical
   parameters, relevant mutable model buffers, profile, and topology can reject
   changed primals under the serialized nonreentrant compare→prepare→commit rule.
   Saved input IDs/targets/masks describe the historical call and need not equal
   subsequently changed caller buffers.
   This avoids assuming new mutation epochs, while making the comparison cost explicit.
2. Enforce one active training frame and exclude every parameter mutation path
   (including public state load and future optimizer writes), or add proved mutation
   generations and reject stale frames. A model-only flag cannot constrain existing
   P1/I2 mutation paths without coordinated integration.

Recommended starting point is one reusable bounded frame with explicit consume or
release, saved primals, and synchronous exact comparison before committing a
contribution; integration must review that unchanged-primal policy or choose
mutation exclusion/generations. This recommendation does not
freeze API names or add epochs to I2. Returned logits must preserve A0 ownership and
graph identity without granting mutation of saved internals. A reusable frame
cannot overwrite saved primals while any live output retains that graph; consume,
release, and invalidation require an explicit A0 decision. Define repeated accessor,
second backward, eval/no-grad, release-before-backward, active-borrow, stale/cross-model,
and partial-failure behavior before implementation.

I2 destructors free native payloads but published opaque handles leave inert shells.
Allocating/destroying fresh tensors for every operation each iteration therefore
does not by itself meet flat-RSS training gates. Reuse model/workspace carriers and
bound concurrent outputs/tapes, test both payload baselines and long-run resident
memory. P1 module destruction is absent and I2's module ledger is capped at 4096;
reuse one model. General module teardown/GC is not necessary to this exact profile,
while safe transient lifetime is necessary. No output or tape may borrow authority
from `state-dict-release!`.

## Capability routing, public transport, configuration, and packaging

The inventory probe independently discovers five provider descriptors (I1, I2,
N2, A2, L2) containing 12 distinct verified capability entries. Linking all objects
still leaves the provider-free K1 baseline at 11 unverified entries and zero
verified entries. No canonical provider symbol is defined. K1 accepts one resolver
provider per runtime; no production multi-provider composer exists in this base.
Architecture's present-tense composition language describes the required rule,
not an implemented aggregate. The I2 contract explicitly calls composition future
work.

Two candidate designs need no inherent K1 ABI change: fixed per-provider runtimes
plus model-owned union validation/report/routing, or one private composite provider
descriptor. Both must reject duplicate **whole-capability** ownership, preserve
unchanged request/view/storage identities, and preflight all required operation rows
before model allocation/publication. Neither may concatenate providers sharing a
capability into broader rectangles, advertise compiler AD, or infer availability
from linking. Preserve native categories through E1; malformed public shapes and
well-formed unsupported profiles need deliberate distinct mappings. No K1 evidence
row is widened by this proposal.

The I2 aggregate still exports exactly 47 E1/P1/D1/X1/C1/T1 globals. Actual public
object and AOT probes reject `transformer.model`, the private I2 constructor, and
T1's private tensor read. A positive public P1 AOT call succeeds and returns the
expected E1 structured error. This proves the tested boundary is real, not an
unusable compiler environment. Existing N2/A2/L2 AOT tests execute test-only native
bridges; they cannot be promoted to production model transport.

A production model needs fixed-purpose reviewed Eshkol/native adapters and one
successor aggregate source-composed from the merged trusted registry-owning inputs,
then localized once. Never link prelocalized I2, T1, T2, P1, or X1 registry-owning
archives side by side. The successor needs exact root/bridge/renames/exports/native
source/undefined manifests and repository-tuple admission, public caller closure,
archive-index/private-symbol/hostile-include/duplicate-owner negatives, and fresh
AOT artifact/output reproducibility. Tests and Python oracles cannot enter its
source closure. Independent native provider archives can be inputs to a reviewed
link strategy; they do not create E1/P1 registries by themselves.

X1 resolves the proposed existing dimension fields successfully and derives
head-size=2, kv-head-count=2. Its 14-field v1 schema rejects `model.ffn-size` and
`model.profile`. It has no activation, position policy, tie policy, normalization
kind/epsilon, dropout probability, initializer identity, or model-family field.
Consequently the current X1 fingerprint alone cannot identify those model semantics.
A fixed profile constructor with a separately versioned model identity is an
alternative to extending X1's schema; variable choices require explicit canonical
representation and a schema/format/version decision through #1. Do not insert
unknown X1 fields, change defaults silently, or use runtime capability reports as
configuration identity. This audit changes no X1 bytes or versions.

## Bounded prerequisite proposal and order

These are proposed review boundaries, **not created tasks or accepted APIs**.
Integration can combine or split them after reviewing evidence.

| Proposed prerequisite | Required change and alternative | Contract/version/lifetime implications | Depends on / exit evidence |
|---|---|---|---|
| Model profile and lifecycle decision | Choose exact profile, constructor/VJP/gradient-normalization boundary, output release, stale-gradient policy; fixed profile identity vs X1 extension | A0/public SemVer decision; profile/config identity version; no inferred checkpoint migration | Integration decision first; shape/owner transition specification |
| Exact N2 shape evidence | Verify byte/position embedding, four linear, GELU and residual rows above; alternative larger tested ranges only with complete evidence | Existing operation schemas can remain; evidence/provider version decision required before metadata changes | Profile; every new admitted operation-row pair forward/VJP/oracle/negative/FP/failure-atomic tests |
| Tensor composition and initialization | Bounded head permutations/inverse VJP, activation/tied adds, exact i64/bool ingress, random fill and exact constant ingress; optional loss reduction/seed | Separate reviewed native operation/adaptor schemas; owned disjoint outputs, deterministic order/RNG identity; no K1 ABI change assumed | Profile/carrier decisions; marked layouts, central differences, exact RNG bits, malformed masks/IDs and allocation rollback |
| Production transport and one aggregate | Reuse private I2/P1L/T1 seams inside a new reviewed source aggregate; fixed per-provider routing or private composite | Fixed-arity error/owner boundaries, exact exports/manifests; no duplicate registries or generic privileged dispatcher | Merged seams plus accepted profile; public AOT positives and all boundary negatives |
| Eshkol model composition | Build canonical tied module once; execute forward and explicit analytic reverse schedule in reusable owned frames; stage one I2 contribution batch | Output/tape consume/release and mutation contract; no compiler reverse-AD assertion | All required rows/transport/lifecycle above merged or explicitly accepted; full tiny-model oracle and gradient/lifetime evidence |

A2 numerical changes are unnecessary for this profile. RoPE would instead require
new `[1,2,2,2]` RoPE evidence and a frequency initialization/buffer policy. Removing
positions avoids a row but changes the proposed GPT profile; it is not a silent
substitute. A scalar/tiny-vocabulary model cannot satisfy byte-level acceptance.
General tensor/autodiff engines, broad K1 capability extensions, accelerator work,
cache adaptation, T2/D2, O2, and C2 are not prerequisites to this bounded model
forward/VJP diagnostic. They remain dependencies for their actual downstream goals.

## Executed evidence and reproduction

All shell execution used `/usr/bin/bash` with login disabled. The fresh isolated
checkout had no toolchain/build artifacts. A fresh compiler was bootstrapped from
locked Eshkol `90cbd7130f47b8184bcc77b8d5c1b0026da980de`, version `1.3.4-evolve`.
Host: x86-64 CachyOS, Clang/LLVM 22.1.6, **unsupported compatibility lane**.
The standard host guard rejects this host; the explicit override records
`host_supported=false`. These results do not satisfy Ubuntu 22.04/LLVM 21.1.8 CI.

From the repository root (environment assignments are explicit diagnostic choices):

```sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  CC=/usr/bin/clang CXX=/usr/bin/clang++ ESHKOL_JOBS=4 \
  /usr/bin/bash scripts/bootstrap-eshkol.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/configure.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash scripts/build-i2.sh
/usr/bin/bash tests/probes/m3_preflight/numerical_probe.sh
/usr/bin/bash tests/probes/m3_preflight/packaging_probe.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash tests/probes/m3_preflight/api_run.sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
  /usr/bin/bash tests/probes/m3_preflight/config_run.sh
```

| Executed gate | Result and boundary |
|---|---|
| Fresh bootstrap/configure/I2 build | Pass; pinned source/hash provenance and I2 artifact manifest guards; unsupported local host |
| Numerical row/layout probe | 31 expected K1 admissions/rejections pass; missing rows return category 6/code 15; marked layout mismatch detected |
| Existing native N2/A2/L2 suites | 1,892 / 1,041 / 4,336 checks pass; independent primitive numerical evidence, not composed decoder parity |
| Provider inventory and repeated output | Five independently discovered providers, 12 verified entries; unchanged baseline; no canonical provider symbol; deterministic inventory bytes |
| Existing N2/I1/I2 and K1 tests under all-provider link | 449 and 596 checks pass |
| Public object/AOT boundary | Six expected compile failures; public P1 positive runs; exact 47-global I2 manifest matches |
| Native parameter lifetime probe | Stable identity with changed primal verified; snapshots stable; active-borrow mutation rejected; tracked resource counts return to baseline |
| Lifetime ASan/UBSan/LSan | Probe passes with `ASAN_OPTIONS=detect_leaks=1`; scoped to this native diagnostic, not all model lifetimes |
| Public X1 profile probe | Existing profile fields resolve; two unknown keys reject; fingerprint deterministic across two fresh AOT executions |

Probes live only under `tests/probes/m3_preflight/`; generated binaries/logs live
under ignored build/temp directories. The compiler-backed runners revalidate full pinned toolchain provenance,
use fresh caches, and consume the existing I2 artifact; bootstrap/configure above
establish the local build provenance. Diagnostic native runners print their compiler and explicitly
disclaim supported-lane and public model evidence. No performance, flat-RSS, general
AD, JIT parity, mixed precision, GPU, training, resume, or end-to-end model numerical
claim follows from these tests. Full project gates/CI were not rerun for this
planning-only change.

## Required later acceptance and independent reconciliation

Before accepting production M3, run a frozen independent forward/analytic-VJP oracle
and central differences for all 1,184 canonical f32 parameters. Perturb tied weights
once so both embedding and head uses change. Include repeated IDs `[7,7]`, boundary
IDs `[0,255]`, distinguishable head/token values, causal future-token noninfluence
and zero VJP to future-token activations, and gradients through every LN/FFN/residual branch.
Test scalar-loss versus numerator normalization separately. Test malformed shapes,
IDs, dtype, device, masks, aliases, NaN/Inf, stale/forged/cross-model outputs, repeated
release/backward, every allocation failure, and failure before I2 batch commit.
Measure a long bounded frame-reuse loop for flat RSS and native payload baselines.
Require fresh public AOT, source/package isolation, deterministic seed/state and
supported-lane gates before claiming the implementation usable.

Three independent Astra-high reviewers audited numerical composition/shapes, public
API/ownership, and tests/packaging. Their findings agree after these distinctions:

- Provider admission, native numerical execution, private AOT transport, and public
  Eshkol model execution are four different evidence levels.
- Head weight tying needs no transpose; activation head layout needs a permutation.
- A2 attention already fits; learned positions remove a RoPE prerequisite.
- I2 stable storage identity does not preserve saved primal values or supply epochs.
  Saved-primal exact comparison is a candidate commit policy that avoids making
  new epochs or global mutation exclusion mandatory.
- Private model composition can reuse merged carriers without requiring a new I2
  provider; public lifetime/API decisions and package review remain necessary.
- L2 already admits the byte logits; missing scalar reduction is separate from its
  accepted per-token operation and from I2's normalization-weight accounting.

No disagreements about these blockers remain. The next action belongs to integration:
accept/revise this proposal and dispatch separately reviewed prerequisites. M3 stays
planned and incomplete; issue #65 records audit readiness only.
