# M3T production transport proposal

Status: **proposal for integration decision; no implementation authorized**.
Tracking [#68](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/68), parents
[#65](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/65) and
[#1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1).
Base: merged main `15ab04d4be6df11ca279c3eeb50b582e8163d535`.
Inputs are the merged contracts and fresh preflight `a97c109`; deleted proposals
and unmerged D2/O2/C2 are not inputs.

## Decision requested

Accept the bounded diagnostic public surface below, its separate profile identity,
and three narrowly necessary lifetime decisions before production changes:

1. Explicitly mutable reusable diagnostic workspace and caller-owned logits
   receivers, rather than reusing an immutable A0 model-output identity.
2. Private unpublished I2 scoped views, preserving public ABI/provider identity
   and existing borrow exclusion, because ordinary borrow release retains shells.
3. Private P1/I2 unpublished-constructor abort/seal handling, because partial
   registration currently has no safe destruction boundary.

The public primitive transport is deliberately real production functionality:
ordinary Eshkol application code can call the bounded operations in an order it
authors. M3T does not supply an automatic complete forward/reverse schedule,
`model-forward`, loss, scalar reduction, loader, optimizer, checkpoint, trainer,
evaluation, or generation. Its operations are fixed profile roles, not caller-
selected native capabilities, arbitrary tensors, or a global resolver.

## Profile and initialization identity

Profile symbol: `eshkol-diagnostic-byte-decoder-v1`. It means exactly
N=1,T=2,V=256,D=4,Hq=Hkv=2,Dh=2,L=1,F=8, CPU f32, dense zero-offset row-major,
learned positions, separate bias-free projections, tied head, exact-erf GELU,
three affine LayerNorms with epsilon bits `0x3727c5ac`, no dropout or cache,
and the merged deterministic numerical contracts. It is independent of X1's
fingerprint. A constructor stores both identities; neither substitutes for the
other. X1 bytes, versions, keys and defaults remain unchanged.

The constructor accepts only a same-aggregate resolved X1 value with those exact
dimensions/dtype/device and deterministic=true. Well-formed other dimensions are
`unsupported`. `run.seed` is the initialization key and must equal the typed
initializer state's seed. `training.accumulation-steps` is retained X1 data and
does not select M3T behavior.

Typed initializer values are immutable, same-aggregate registered identities with
the private algorithm `n3k.philox4x32-10.uniform-f32.v1` and exact i64 words
`[1,seed,counter-low,counter-high]`. Public construction starts at counter zero,
with an exact nonnegative signed-i64 seed matching X1. A successful model stores
one newly registered successor identity; repeated access returns that cached
immutable identity. State words/algorithm inspection is detached data and never
grants native authority. A copied vector, same-message condition, foreign RNG, or
N2 dropout state cannot impersonate this kind. No serialization is added.

Canonical paths below are lists of the displayed slash-separated ASCII segments.
Ordering is P1's unsigned UTF-8 segment ordering; there are 15 logical paths,
14 unique parameters, 1,184 f32 values, and one tie group.

| Canonical unique path | Shape | Initialization / counter interval |
|---|---|---|
| `blocks/0/attention/key/weight` | [4,4] | random [0,4) |
| `blocks/0/attention/output/weight` | [4,4] | random [4,8) |
| `blocks/0/attention/query/weight` | [4,4] | random [8,12) |
| `blocks/0/attention/value/weight` | [4,4] | random [12,16) |
| `blocks/0/ffn/down/weight` | [4,8] | random [16,24) |
| `blocks/0/ffn/up/weight` | [8,4] | random [24,32) |
| `blocks/0/norm1/bias` | [4] | exact +0; no draws |
| `blocks/0/norm1/weight` | [4] | exact 1; no draws |
| `blocks/0/norm2/bias` | [4] | exact +0; no draws |
| `blocks/0/norm2/weight` | [4] | exact 1; no draws |
| `head/weight` | [256,4] | random [32,288); tied to `token_embedding/weight` |
| `norm_final/bias` | [4] | exact +0; no draws |
| `norm_final/weight` | [4] | exact 1; no draws |
| `position_embedding/weight` | [2,4] | random [288,290) |

The implementation first validates the full shape/tie collection, groups actual
unique storage identities, chooses each group's least path, sorts representatives,
then calls N3K once per random group threading the successor. It does not merely
trust registration order or initialize the head twice. All initialization, required
capability admission, and allocation must succeed before public model publication.
Intervals in the table are offsets from the supplied state's counter. Any live
registered state of this initializer algorithm, including a previous constructor's
successor, is admitted when its seed matches X1 and its counter has capacity for
all 290 blocks under N3K's exhausted-sentinel rule. Whole-sequence exhaustion is
checked before initialization; failures preserve the supplied immutable state.
The constructor retains both original and successor states. Model/successor registry
records and every fallible check/allocation are staged before the final seal; no
fallible registration or allocation may follow publication.

The constructor returns a genuine P1 module compatible with existing parameter,
snapshot/load, mode, and zero-gradient operations. Successful model storage retains
the existing process-local module lifetime; there is no model destructor.

## Proposed public module and exact arities

All new names belong to `transformer.diagnostic_transport`. The installed source
requires only `transformer.error_consumer`, and exports exactly the names listed
in this section. Each has one same-capability safe closure alias for the pinned
compiler, following E1B; aliases expose no additional authority.

| Operation | Arity | Result / purpose |
|---|---:|---|
| `diagnostic-initializer-state` | 1 | seed → typed initial state |
| `diagnostic-initializer-algorithm` | 1 | state → algorithm symbol |
| `diagnostic-initializer-words` | 1 | state → detached four-exact-integer list |
| `diagnostic-model-create` | 2 | resolved config, typed initializer state → P1 module |
| `diagnostic-model-profile` | 1 | model → fixed profile symbol |
| `diagnostic-model-initializer` | 1 | model → cached successor state |
| `diagnostic-input-create` | 0 | newly owned reusable exact i64[1,2] input |
| `diagnostic-input-copy!` | 2 | input, exact two-integer list → #t |
| `diagnostic-input-copy-tokenizer!` | 2 | input, same-aggregate sealed T1 i64[2] → #t |
| `diagnostic-input-release!` | 1 | permanently close input; idempotent → #t |
| `diagnostic-logits-create` | 0 | newly owned reusable f32[1,2,256] receiver |
| `diagnostic-logits-copy-bits!` | 2 | logits, exactly 2048-byte bytevector → #t |
| `diagnostic-logits-bits` | 1 | logits → detached exact 2048-byte copy |
| `diagnostic-logits-release!` | 1 | permanently close logits; idempotent → #t |
| `diagnostic-workspace-create` | 1 | model → reusable model-bound workspace |
| `diagnostic-workspace-begin!` | 2 | idle workspace, input → #t; snapshot primals/inputs |
| `diagnostic-workspace-vjp-begin!` | 2 | workspace, logits upstream → #t; snapshot upstream |
| `diagnostic-workspace-copy-logits!` | 2 | workspace, logits receiver → #t; explicit copy |
| `diagnostic-workspace-check-primals` | 1 | active workspace → #t or stale `invalid-state` |
| `diagnostic-workspace-reset!` | 1 | end/discard active frame; idempotent while idle → #t |
| `diagnostic-workspace-release!` | 1 | permanently close workspace; idempotent → #t |

The input and logits values are genuine opaque owners of I1/I2 tensors, not raw
pointers, ordinary numeric vectors, or test carriers. Their descriptor is fixed by
their public type. Values are recognized by exact registered identity before native
access. Copy/release authority is specific to the owning type and cannot consume
P1 states, parameter handles, initializer metadata, another workspace, or T1 storage.
New input storage contains exact IDs [0,0]; new logits storage contains exact
positive-zero bits in all 512 elements.

Exact ingress is explicit and bounded: input copy validates a proper acyclic list
of length exactly two and representation-real exact signed-i64 values before any
native conversion, then validates both IDs in [0,256), stages, and commits both.
The tokenizer adapter validates the sealed T1 owner and exact rank-one length two,
then synchronously copies its two exact words into distinct owned rank-two I1
storage. It retains no T1 pointer. These are declared conversions, not implicit
reshapes or a general integer tensor adapter. Positions are owned exact I1[1,2]
constants [0,1]; attention keep storage is exactly four owned canonical bool bytes
[1,1,1,1] with shape [1,2,2]. Neither is an application-selected numeric mask.

Logits byte ingress/egress uses four little-endian bytes per row-major IEEE-754
binary32 element. It preserves every f32 bit pattern without numerical conversion; provider
dispatch rejects nonfinite operands under its unchanged contract. Byte egress is
explicit observational copying, not a differentiable scalar loop. Both calls are
outside the steady-state allocation claim. Constants use reviewed I2 exact-bit
ingress: norm weight `0x3f800000`, bias `0x00000000`, epsilon `0x3727c5ac`.

### Public primitive transport

The following operations return #t, mutate only the receiver's owned destination
slots, and have matching `-vjp!` counterparts with the same arity. For example,
`diagnostic-linear-vjp!` takes workspace and projection. No operation accepts a
tensor pointer, shape, provider, callback, capability name, or arbitrary opcode.

| Forward name | Arity | Exact role selectors |
|---|---:|---|
| `diagnostic-embedding!` | 2 | token, position |
| `diagnostic-linear!` | 2 | q, k, v, attention-output, ffn-up, ffn-down, tied-head |
| `diagnostic-layer-norm!` | 2 | block-input, ffn-input, final |
| `diagnostic-gelu!` | 1 | — |
| `diagnostic-residual!` | 2 | embeddings, attention, ffn |
| `diagnostic-heads-split!` | 2 | q, k, v |
| `diagnostic-heads-merge!` | 1 | — |
| `diagnostic-attention!` | 1 | — |

One additional name, `diagnostic-sum!` (arity 2), accepts exactly qkv-input,
attention-residual-input, ffn-residual-input, or tied-weight. It dispatches N3K
sum3 for Q/K/V in that order, sum2 for skip then branch, or sum2 for head then
embedding. These selectors describe arithmetic edge order, not capabilities.

Each role has fixed source/destination slots matching the merged provider tensor
tables. Embeddings feed embedding residual; norm1 reads that residual; Q/K/V read
norm1; their materialized head layouts feed A2; merged attention feeds the output
projection and attention residual; norm2 and FFN up/GELU/down feed FFN residual;
final norm feeds the tied head. Reverse roles read the saved forward operands and
their corresponding upstream slots and produce distinct per-edge VJP slots. The
four named sum roles combine those edges before downstream consumption. Embedding
residual VJP copies its upstream to the token and position branches; it needs no
extra sum. Parameter VJP slots have the exact 14 unique parameter shapes, with
separate head/embedding edge slots before the one tied sum.

This fixed wiring is a transport contract only. M3T implements no loop/function
that executes the complete model schedule. Every destination has one producer per
frame; repeated writes, missing source readiness, wrong phase and unavailable
selectors reject before dispatch. Thus callers cannot overwrite an ancestor while
retaining silently stale descendants. Future M3 writes the forward and reverse
ordering in Eshkol and proves full-model numerical parity.

The following internal slot labels make the role wiring exact. They are not public
tensor names or a lookup API. Forward values are `et,ep,x,n1,qt,kt,vt,qh,kh,vh,ah,at,
ao,r,n2,fu,fg,fd,y,nf,z`. Every f32 slot is an independent I2 owner; intermediate
token-major slots have [1,2,4], heads have [1,2,2,2], fu/fg have [1,2,8], z has
[1,2,256], and parameter-gradient slots have their parameter's exact shape.

| Role | Forward sources → destination | VJP upstream → destinations |
|---|---|---|
| embedding token | ids, token weight → et | det → gtoken |
| embedding position | positions, position weight → ep | dep → gposition |
| residual embeddings | et, ep → x | dx → det, dep |
| norm block-input | x, norm1 weight/bias, epsilon → n1 | dn1 → dxattention, gnorm1weight, gnorm1bias |
| linear q/k/v | n1, respective weight → qt/kt/vt | dqt/dkt/dvt → dn1q/dn1k/dn1v, respective gweight |
| heads split q/k/v | qt/kt/vt → qh/kh/vh | dqh/dkh/dvh → dqt/dkt/dvt |
| attention | qh,kh,vh,positions,positions,keep → ah | dah → dqh,dkh,dvh |
| heads merge | ah → at | dat → dah |
| linear attention-output | at, output weight → ao | dao → dat,goutput |
| residual attention | x,ao → r | dr → dxskip,dao |
| norm ffn-input | r,norm2 weight/bias,epsilon → n2 | dn2 → drffn,gnorm2weight,gnorm2bias |
| linear ffn-up | n2,up weight → fu | dfu → dn2,gup |
| GELU | fu → fg | dfg → dfu |
| linear ffn-down | fg,down weight → fd | dfd → dfg,gdown |
| residual ffn | r,fd → y | dy → drskip,dfd |
| norm final | y,final weight/bias,epsilon → nf | dnf → dy,gfinalweight,gfinalbias |
| linear tied-head | nf,tied token weight → z | dz → dnf,ghead |
| sum qkv-input | reverse only | dn1q,dn1k,dn1v → dn1 |
| sum attention-residual-input | reverse only | dxskip,dxattention → dx |
| sum ffn-residual-input | reverse only | drskip,drffn → dr |
| sum tied-weight | reverse only | ghead,gtoken → gtied |

VJPs additionally read exactly the saved primal operands required by the merged
provider table; they never read current mutable parameter bytes. All 14 final
unique contributions include gtied once, not ghead and gtoken separately.

## Workspace, output and stale-primal lifetime

A model permits one active workspace frame at a time, including across separately
allocated workspaces. Workspace states are idle → forward → reverse → idle, with
permanent released terminal state. Reset discards a frame from either active phase.
Release preflights borrow/reentrancy guards, ends an active frame, invalidates the
receiver, then destroys every owned carrier exactly once. All storage, readiness
bits, fixed operand tables and cleanup slots are allocated once at workspace create.

Begin snapshots all 14 canonical live parameter tensors into separate reusable I2
storage, exact IDs, fixed positions/mask, model mode, profile and topology identity.
Every forward/VJP reads these saved primals. VJP begin requires ready logits and
snapshots an independent owned upstream f32[1,2,256]; it neither normalizes nor
consumes RNG. Copy-logits copies ready logits to an independently owned receiver;
future writes to either owner cannot alter the other's bytes. Repeated copy calls
are allowed; no graph is attached to that mutable readout.
VJP begin makes the forward→reverse transition exactly once. Analytic VJPs are
allowed in either captured train or eval mode; M3T supplies no no-grad mode.

The durable workspace and logits receiver deliberately denote their current mutable
contents. An old alias is another alias to that same mutable receiver, not an
immutable old frame. Reset makes active-frame operations invalid until begin.
Release is permanent: no dead public identity is recycled into live authority.
Copied metadata never authorizes frame, tensor, gradient or release operations.

**A0 decision:** these are additive diagnostic operations with explicit mutable
receivers and explicit copies. They do not implement or reinterpret
`model-forward`, `model-output-logits`, `model-output-loss`, or `model-output-rng`.
A0's ordinary owned output/retained-graph contract remains outstanding for later M3;
integration must accept this diagnostic receiver exception before these names freeze.
It is not permissible later to return this workspace as an immutable A0 output.

Check-primals synchronously compares exact shape/bytes for all 14 current canonical
parameters plus mode, profile, topology and relevant immutable constants against
the saved frame. Changed bytes reject `invalid-state`; mutation followed by exact
restoration may pass. This is a value comparison, not epochs or mutation-history
detection. The public check is observational and grants no later commit authority.
Future M3's trusted contribution boundary must repeat the comparison and perform
I2 prepare/commit in the same serialized, nonreentrant call; checking now and
committing later is forbidden. That future contribution uses unnormalized VJP
numerators, one unique handle each and a separately supplied normalization weight.
M3T exposes no scalar-loss, normalization or gradient-commit operation.

The bounded success loop creates model/input/workspace/logits once, executes
begin/primitive/copy/reset calls in reclaimed Eshkol regions, and reuses native
storage and scoped leases. No public value escapes an expiring region. Public
accessor snapshots, fresh model/workspace/input/output construction, P1 snapshot/
load/zero-grad plans, I2 gradient plans, T1 encoding and E1 errors retain their
existing cumulative cost. Flat transport memory does not prove flat training
memory. The future contribution/reset plan lifecycle is an explicit M3 dependency.

## Necessary private lifetime seams

No new installed C header, public native ABI, I2 provider ID, K1 version, f32
representation, tensor metadata or numerical operation is introduced.

I2 private scoped-view pair (internal header only):
`et_f32_tensor_scoped_begin_internal(tensor, guard, error)` and
`et_f32_tensor_scoped_end_internal(guard)` have arities 3 and 1. A caller-owned
stack guard contains the unchanged K1 view and an unpublished exclusion sentinel.
Begin validates a registered live tensor and all current borrow/plan exclusions,
then pins it through the ordinary active-borrow guard without publishing a public
lease or adding a retired shell. Guards start zero/inactive; begin rejects an
already-active guard. Both functions return an i32 native status. End is idempotent
on an inactive exact guard, otherwise clears only the registered owner's exact
installed sentinel; copied guards cannot release it. Misuse is invalid-state;
properly tracked cleanup is proved nonallocating and infallible.
Guard/storage never cross the native synchronous call; no application supplies one.
Fixed M3T dispatch code acquires all guards, calls unchanged K1 dispatch, snapshots
status, and releases all acquired guards on every return. Public I2 borrow APIs
must reject while a private guard is active and must never accept its sentinel as
a public lease. I1 existing begin/end already frees its leases and needs no change.
Repeated equal input owners in admitted sum/residual calls acquire one guard;
ordered K1 operand tables retain the exact semantic views. Unwind is reverse
acquisition order. The M3T-only implementation can source-include the unchanged I2
implementation in a private integration translation unit and add these scoped
helpers there. That leaves the standalone I2 public object/header/symbol manifest
unchanged; the new private symbols are declared in M3T's exact manifests and
localized. It does not alter the standalone borrow ABI or retired-shell rules.

P1 proposed trusted-only construction boundary: `module-construction-begin-internal`
(arity 1, admitted provider → construction), `module-construction-root-internal`
(arity 1), `module-construction-seal-internal!` (arity 1 → finalized root), and
`module-construction-abort-internal!` (arity 1 → #t). A construction owns only a
fresh unpublished root and every newly attached descendant/handle created through
the same construction scope. Existing registration remains private and recognizes
that exact scope. There is one serialized nonreentrant active construction; nested
begin rejects. Every fresh descendant and handle is enrolled before exposure and
all registration/tie/child operands must belong to that exact construction.
Existing live trees/states cannot be adopted, tied or attached into it; public
observation is forbidden before seal. Ledger storage precedes each publication.
Abort preflights the exact ledger for outstanding guards/plans, then revokes the
whole partial module/handle authority graph, clears shell metadata and I2 membership,
and lets the constructor owner destroy its carriers in a nonallocating, nonraising
tail. It is idempotent and rejects
sealed constructions. Seal validates/finalizes the full tree before its infallible
publication tail. Seal/abort clear the active scope. This is not a general module
destructor or public builder.

The I2 wrapper operations are `i2-construction-begin-internal` (arity 0 → owner
ledger plus P1 construction), `i2-construction-seal-internal!` (arity 1), and
`i2-construction-abort-internal!` (arity 1). They hard-code the existing I2 provider,
track exact newly created module membership and parameter ownership, and call only
the exact bound P1 construction. Abort completes preflight before P1 revocation,
then removes I2 membership and destroys each owned carrier once without a fallible
callback. Metadata/provider IDs cannot select a destructor. Constructor failures
may leave authority-free identity
tombstones, but no live parameter/tensor payload or dangling actionable carrier.
These are proposed changes to P1/I2 private lifetime assumptions and require
explicit integration acceptance; source inspection alone does not authorize them.

### Private Eshkol/native transport names

Trusted Eshkol handlers have the exact public suffixes prefixed `m3t-public-` and
the same arities. The source owns configuration checks, opaque identity admission,
canonical P1 construction/tie traversal and call ordering. The native boundary
below only transports fixed owners and calls reviewed kernels. Names in the table
are suffixes of `et_m3t_private_<suffix>_v1`; native pointers never escape trusted
source. Native status is i64 at Eshkol externs; native i32 statuses widen exactly.

| Private native suffix | Arguments, in order | Result |
|---|---|---|
| initializer_create | exact i64 seed | private initializer pointer |
| initializer_word | initializer, i64 index 0..3 | exact i64 word |
| initializer_seal / initializer_abort | unpublished initializer | status |
| owner_create | initializer | staged fixed-shape owner pointer |
| owner_parameter | owner, i64 unique index 0..13 | private I2 parameter pointer |
| owner_bind | owner, i64 index, exact P1 handle | status |
| owner_initialize | owner, i64 unique index | status; N3K random or exact constant ingress |
| owner_successor | owner | cached private successor pointer |
| owner_seal / owner_abort | owner | status |
| input_create | no arguments | owned input pointer |
| input_copy | input, exact i64 first, exact i64 second | status |
| input_copy_tokenizer | input, validated private T1 sealed owner | status |
| input_release | input | status |
| logits_create | no arguments | owned logits pointer |
| logits_copy_from / logits_copy_to | logits, byte span, exact i64 count 2048 | status |
| logits_release | logits | status |
| workspace_create | bound model owner | workspace pointer |
| workspace_begin | workspace, input, captured bool mode encoded 0/1 | status |
| workspace_vjp_begin / workspace_copy_logits | workspace, logits | status |
| workspace_check_primals | workspace, current bool mode encoded 0/1 | status |
| workspace_reset / workspace_release | workspace | status |
| embedding / linear / layer_norm / residual / heads_split | workspace, closed role index, reverse bool encoded 0/1 | status |
| gelu / heads_merge / attention | workspace, reverse bool encoded 0/1 | status |
| sum | workspace, closed edge-group index | status |
| workspace_gradient | workspace, canonical unique index 0..13 | private ready contribution tensor pointer |
| last_error_category / last_error_code | no arguments | exact bounded i64 |

The role indices follow each public selector list in table order, starting at zero;
unique indices follow the canonical unique-path table. Native code independently
validates every index, state and owner. No argument can name another provider or
arbitrary native operation. Native creation output is null on failure; status calls
preserve their documented data outputs on rejection and use a fixed private error
record. No error-message pointer crosses Eshkol; source maps numeric diagnostics to
bounded fixed strings. Constructor abort consumes unpublished ownership only.

The only future contribution handoff is trusted Eshkol
`m3t-workspace-contributions-internal` (arity 1): require all 14 final VJP slots
ready and return a preallocated canonical vector of the model's exact P1 handles
and their ready private I2 contribution carriers. It exposes no public authority,
owns no storage and is valid only in the active reverse frame. Future M3 must
repeat the primal check in its synchronous contribution transaction; an extracted
vector is not a commit token. Native `workspace_gradient` cannot mutate parameters
or gradients and performs the same owner/readiness validation.

## Dispatch, errors and package boundary

Use three fixed runtimes discovered once through the merged N3K, N2 and A2
accessors. Validate descriptors, reject duplicate whole-capability ownership and
preflight every consumed exact row before model publication. N2 is used for
LayerNorm only; A2 for exact [1,2,2,2,2,2] attention; N3K for its declared rows.
Keep requests/view identities unchanged through K1 generic validation and provider
validate/invoke. No canonical resolver, capability rectangles or fallback exists.

Public validation order: receiver identity/kind/owner; recognized dead/conflicting
lifecycle; selector/control type and range; tensor shape/dtype/device/layout;
well-formed unsupported profile/capability; unchanged K1/provider validation.
Allocation occurs only after the necessary admission for its stage; K1 discovery
itself allocates a validated metadata snapshot, and constructor staging precedes
later validations. There is no impossible global allocation-last guarantee.
Wrong/forged/copied/cross-owner values are `invalid-argument`; recognized
dead/stale/conflicting phase is `invalid-state`. Wrong input length or token range
is `shape-mismatch` at this explicit public tensor ingress; malformed scalar/list
representation is `invalid-argument`. Native categories are preserved thereafter.
Allocation failure is `internal` with bounded source code, never hidden as unsupported.
Unknown/impossible native category/code is a bounded internal defect.

Every failure completes deterministic borrow/unpublished-owner cleanup before the
fixed E1B raise seam. Snapshot native status before cleanup. Public operation is
the invoked public name; data-only details record source-domain, source-code,
source-message and native-category, cause #f. No raw pointers or caller graphs enter
details. A failed primitive preserves all already-ready data and destination bytes;
readiness changes only on success. A failed begin leaves the workspace idle and
the prior logits receiver/parameters/RNG untouched; unpublished scratch has no
read authority and can be overwritten on retry.

Source proposal: `native/m3t_package_root.esk` source-loads merged I2/Wave1 trusted
roots; `native/m3t_package_bridge.c` adds reviewed boxed wrappers;
`src/eshkol_transformer/m3t_transport.c` owns fixed native transport; installed
`lib/transformer/diagnostic_transport.esk` contains only public stubs.
Output: one `m3t_package.o` in `build/m3t/libeshkol_transformer_m3t.a`.
This replaces earlier registry-owning aggregates; none is linked beside it.

The exact export set is the existing 47 globals plus 38 new wrappers: 21 lifecycle/
identity/copy operations, 8 forward, 8 VJP and 1 sum. C names are
`et_e1b_public_m3t_<suffix>_v1`, with the logical `diagnostic-` prefix removed,
hyphens replaced by underscores and terminal ! removed (e.g.
`et_e1b_public_m3t_linear_vjp_v1`). Each wrapper has one input box per logical
argument and one result box. All constructor/registry/lifetime/dispatch/native
provider/error seams and generated companions are localized.

Add an exact E1B repository root/bridge/renames/exports/include/native-input policy
tuple and checked-in Eshkol/native source, defined/undefined, public-string and
archive manifests. Preserve numerical compile flags and unchanged provider-owned
source/artifact contracts. Integration source lives under src so it does not widen
the N2/N3K provider-owned native/include inventory. No tests, oracle, Python, private
bitcode or unlocalized object enters the public artifact.

## Evidence and acceptance gates

Independent Astra-high reviews covered compiled reachability, ownership/RNG and
aggregation/errors. Existing preflight already proves public model/private bridge
unreachability and a positive public P1/E1 AOT caller; no broad compiler audit is
needed. A narrow fresh local development probe measured I2 retired borrow shells:
1,000/10,000/30,000 borrow cycles leave exactly that many 96-byte shells and zero
live borrows (96,000/960,000/2,880,000 bytes). Current process RSS was observed as
2240/3224/5388 KiB; inherited peak RSS was unsuitable and is not evidence.
A pinned AOT region probe with poisoning ran 30,000 transient-vector iterations
and returned the expected count. These are local compatibility diagnostics only;
neither is public aggregate flat-memory evidence.
The development probe sources are checked in at
`tests/probes/m3t_lifetime/borrow_retention.c` and `region_reuse.esk`.
The native probe is reproducible with
`clang -std=c11 -Iinclude -Inative tests/probes/m3t_lifetime/borrow_retention.c native/kernel_abi.c -lm -o /tmp/m3t-borrow-retention`
followed by `/tmp/m3t-borrow-retention 30000`. Record compiler/host separately;
this command alone never establishes supported provenance.

After acceptance, required gates are:

- Two fresh pinned public AOT builds using only installed facades and the completed
  artifact: real constructor, typed RNG continuation, canonical P1 ties, exact input
  adapters, real N3K/N2/A2 transport fragments and VJP calls, owned output copies,
  E1 import order/identity, deterministic binaries and output.
- Every exact selector and readiness rule; wrong tagged types, complex/inexact/
  out-of-i64 ingress, improper/cyclic lists, length/range negatives, forged/copied/
  foreign/wrong-owner/dead identities, active borrow, double release, second frame,
  mutation/restoration of saved parameters, no mutation on failure.
- Every constructor/ingress/frame allocation failpoint and partial P1 registration:
  revoked partial authority before destruction, exact live payload baseline,
  no outstanding guards, unchanged outputs/parameters/RNG and retryable state.
- Scoped I2 guards versus ordinary borrow/copy/destroy/plan paths; unchanged K1
  malformed shape/dtype/device/alias/finite/fenv behavior and status precedence.
- Long public AOT successful begin/fragment/copy/reset loop with one set of owners,
  region poisoning, exact stable native addresses, zero growing I2 lease shells,
  payload/guard counters, and current-RSS plateau across increasing iteration
  counts after warmup. Report measured budgets; do not extrapolate to training,
  retained errors, fresh output creation, snapshots or gradient plans.
- Exact source/symbol/archive/undefined manifests, private name/object/link negatives,
  hostile include/environment and copied/partial tuple rejection, duplicate registry
  owner rejection, production Python isolation and repeatable fresh artifacts.
- Focused ASan/UBSan/LSan and supported Ubuntu22/LLVM21.1.8 CI at final head;
  local CachyOS/LLVM22 results remain explicitly compatibility evidence.

Future M3 still owns complete Eshkol scheduling, all-parameter forward/VJP oracle
and gradient checks, atomic numerator contribution, its plan-memory boundary, and
ordinary model-output/A0 graph integration. Integration owns acceptance, review and
merge. This proposal does not mark M3T or M3 complete.
