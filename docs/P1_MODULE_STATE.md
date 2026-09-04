# P1 module and state-tree contract

## Boundary

P1 implements the existing A0 `transformer.module` names and arities and adds the
public unary `state-dict-release!` lifetime operation. It adds no public constructor,
registration operation, tensor descriptor, state-version accessor, serialized
format, or generic privileged dispatcher. Library construction and malformed-state helpers
remain in the explicitly unstable trusted replacement root at
`internal/p1/lib/transformer/module.esk`. The two Eshkol roots are mutually exclusive generated products of one
canonical template. The public root provides exactly 18 bindings and never
imports or references the trusted root; the trusted root independently provides the
same 18 bindings plus fixed-arity construction, provider, scoped-borrow, and
state-adoption helpers.

Pinned Eshkol flattening cannot protect a provider seam inside one compiled source
closure. P1 therefore uses a narrow native identity bridge, version 1.1, and a
single prelocalized E1B package object. That object owns the process's sole E1
registry, the trusted P1 root, the private identity implementation, and 18 fixed-
arity transport wrappers. Its exact global surface is those 18 operations plus
the six A0 error accessors. Every E1B seam, generated P1 thunk, and public/private
identity definition inside it is local. The separately built four-symbol read-only
identity archive remains an alternative inspection product and is never linked
beside the registry-owning package. The private replacement identity archive is
non-installed and exists only for native ABI tests and future reviewed integration.
The wider P1 undefined-symbol manifest is selected before any build mutation only
when canonical paths match the exact repository-owned trusted root, C bridge,
compiler rename map, and public export manifest as one tuple. The exact bridge binds
the repository identity source/header through its fixed quoted include. Copied roots,
copied/spoofed bridges and native inputs, partial tuple substitutions, and environment
policy overrides cannot opt into the P1 manifest.
The compiler depfile's complete repository-relative trusted Eshkol closure is
byte-deterministic and must exactly match the checked-in
`native/p1_package_source_closure.txt`; adding, removing, or reordering any
transitive source fails the focused P1 gate.
The exact layout, ownership, status, and symbol contract is frozen in
[P1_IDENTITY_ABI.md](P1_IDENTITY_ABI.md).

The never-installed trusted root imports only `e1b_error_consumer_private` and calls
the fixed five-value, nonreturning `et-e1b-private-raise`; it never imports
`transformer.error_internal` or `transformer.error_core`, returns an error object, or
uses the generic foreign wrapper. Ordinary failures pass an empty details alist so
caller-controlled paths, cycles, and payloads never enter E1. Native failures map
the bounded status/code to a fixed primary message and canonical data-only
`native-category`, `source-domain`, `source-code`, and `source-message` details with
cause `#f`, snapshotting status before cleanup can overwrite it. Exact valid
status/code pairs are admitted; unknown or impossible pairs map to bounded
`internal` diagnostics. The installed root
imports only `transformer.error_consumer` and declares narrow boxed calls. It
provides exactly the 18 P1 bindings; importing the error facade in either order
uses the same artifact registry and unchanged six accessors.

## Paths, nesting, and identity

A logical path is a nonempty list of valid UTF-8 segments. Each segment contains
1..65536 encoded UTF-8 bytes, inclusive; this is a per-segment bound, not an
aggregate path-byte bound. Ordering is lexicographic by the encoded UTF-8 bytes of
each segment, segment by segment; if all shared segments compare equal, the shorter
path sorts first. Registration order, host locale, and display spelling do not affect
enumeration.

Parameter, buffer, and child names share one local namespace. Duplicate names,
shared children, cycles, and cross-root parameter ties are rejected. Enumeration
flattens the complete nested module tree and then applies the global path ordering.
Registration validates the same 1..65536 encoded-byte domain before provider lookup
or callback, ownership allocation, identity/device work, or topology mutation, then
copies caller-owned name strings. Prebuilt-subtree attachment applies that gate to
its new child edge; every existing subtree segment was already copied through the
same admission gate. A root tree is finalized on its first public enumeration or
state access, after which every registration operation fails with `invalid-state`;
attached child modules are not independent public roots.

One root tree contains at most 4096 modules and 4096 logical parameter/buffer
entries. Module depth is at most 64 child edges, and every generated leaf path is at
most 64 segments; consequently, a module at child depth 64 may not accept a leaf.
These are admission-time invariants, not deferred accessor checks. Leaf registration
and prebuilt-subtree attachment validate the prospective root totals, shifted subtree
height, and shifted maximum leaf span before provider callbacks, identity admission,
device claiming, parent/name mutation, or cached-count mutation. Exact-bound trees
remain valid inputs to path lookup and state operations; one-over attempts leave both
roots and all live/tombstone identity counts unchanged.

Private module records cache authoritative subtree entry count, node count, height,
and maximum generated leaf-path span. Each successful registration applies a
precomputed bounded ancestor-update vector; failed registration applies none.
Observation and mode changes likewise build and validate a bounded, iterative
finalization plan before committing any public shell or tensor mutation. Their
post-commit finalization tail performs only the precomputed finite vector writes and
contains no allocation, callback, validation, recursive traversal, or recoverable
raise. Storage-disjointness checks deduplicate tied bindings by authoritative handle
identity, then compare every unique storage; aliases therefore cannot amplify
subtree-admission work without weakening storage isolation.

Every leaf in one root tree carries the same logical device metadata. Mixed-device
leaf registration or child attachment is rejected with `device-mismatch`. Provider
identities must also agree across an attached tree. Devices may be `cpu` or finite,
acyclic, data-only opaque descriptors. Providers perform exact device comparison;
P1 deep-copies descriptors at metadata and public-accessor boundaries.

Each unique parameter has one live handle. Reusing that exact handle under another
logical path creates a tie; value equality never creates one. Repeated
`module-parameters` calls return fresh tree containers but preserve handle identity.
The first lexical path is canonical. Tie paths and groups use the same ordering and
are disjoint.

## Logical state dictionary 1.0

The opaque in-memory format identity is `transformer-state-dict`, version `(1 0)`.
Version fields are exact nonnegative integers. Version 1.0 permits an empty
required-feature list only. Unknown major versions,
nonzero minor versions, and unknown required features are rejected with
`version-mismatch`; readers never guess forward compatibility.

The logical value contains:

- the format and exact major/minor version;
- a canonical required-feature list;
- the inert provider identity
  `(transformer-tensor-provider 2 0 provider-id)`;
- a raw ordered entry list, preserving duplicates for validation;
- a canonical parameter-only alias graph.

Each entry contains its logical path, kind (`parameter` or `buffer`), redundant
shape/dtype/device/layout metadata, and one owned tensor snapshot. With a conforming
admitted provider, every owned clone has exactly one ledger owner and is either
transferred into one live state or released exactly once. A provider that violates
the physical-storage identity obligation below can make an ambiguous publication
unsafe to destroy; P1 fails closed and makes no exact-cleanup claim for that trusted-
provider defect. Metadata never
overrides the tensor. Before mutation, the loader independently validates tensor
rank/extents, dtype, device, and dense row-major contiguity, then requires exact
agreement with entry metadata and the destination leaf.

Tied logical paths receive value-equal independent snapshots. Alias groups contain
parameters only; conflicting tied payloads are `corrupt-data`. `module-buffers`
contains only buffer entries and has no alias groups.

This schema is a P1/C1 logical contract only. It commits to no magic bytes, field
numbers, encoding, checksum, checkpoint/container version, file path, temporary
file, filesystem operation, or atomic replacement behavior.

The logical value never contains provider callbacks, registration handles,
admission results, load plans, or capability evidence. The trusted root represents
each runtime state with an unforgeable native identity shell; a native process-local
registry records only the exact state-token/provider-token binding. Raw Eshkol
vectors, copied visible metadata, and transplanted tokens cannot authorize a bind.
Bindings are omitted from logical/serialized state and never reconstructed from the
inert provider identity.

`state-dict-bind-provider-internal!` is the trusted handoff for preconstructed test
states. A nonempty unsafe state is bindable only when construction captured exact
registered source-entry and source-state shells from live owned states. First bind
resolves that provenance before any tensor callback, pins every unique source state,
clones every borrowed carrier into a preallocated exact-once ownership ledger, and
unpins only after the last source dereference. Native binding then atomically adopts
the clones and revokes only the target state's replaced entry identities; the target becomes an independently owned state requiring explicit
`state-dict-release!`. Published entry shells for the target's pre-bind borrowed
entries become `invalid-state` tombstones when the entry list is replaced. Released
or mismatched provenance fails before a tensor callback, direct unproven nonempty
construction is unsupported, and clone or native-bind failure releases every
published clone while leaving the target unbound and retryable.

The private pre-state `state-entry-internal` constructor returns an unregistered raw
construction value to its trusted caller. P1 stores no hidden registry edge to that
value or its borrowed carrier, and the value cannot establish ownership or bindable
provenance. Only `state-dict-entries-internal` creates opaque entry identities, and
those identities are scoped to the exact owning state.

The release-aware C1 decoder instead uses one fixed-arity adoption seam that
validates a caller ledger before transferring every decoded carrier into one live
state. Its reviewed trusted-workstream caller must explicitly name an already
admitted provider chosen through trusted policy and capability validation. Before
adding a binding, it validates the exact state and provider-interface versions,
provider ID, every tensor/metadata entry, storage independence, and aliases using
that trusted-workstream-selected provider, never a provider selected by application
input or checkpoint bytes. Adoption is rejected before transfer if the ledger
repeats an envelope identity or exact carrier identity; the caller retains the
complete ledger on those rejections. Ownership confusion is handled separately:
P1 clears every envelope that names storage already owned by a live P1 state or
borrowed by a registered module. It first applies the infallible exact-wrapper
guard, then, after exact provider preflight, applies that provider's
`storage-identical?` callback to distinct wrappers before transfer or any destructive
release callback. This prevents caller rollback from releasing protected storage;
every nonambiguous sibling remains in the caller ledger for exact-once cleanup under
a conforming provider. If the comparator itself violates its contract, P1 disarms
the ambiguous carrier and makes no exact-cleanup claim for it.
Provider-reported aliases among newly claimed owners are rejected by complete state
validation and rolled back through the transferred P1 ledger. Rebinding the same
state/provider is identity-preserving and idempotent. Conflicting, unknown,
unadmitted, malformed, or incompatible bindings fail through E1 without changing
the existing binding. Unbound states cannot be inspected or loaded and fail
`unsupported`. State metadata is never used to select or load executable code.

`state-dict-tensor` returns one cached opaque, read-only state-backed identity per
entry. It never creates another native tensor clone. A trusted synchronous consumer
must resolve the exact `(state, handle)` pair through P1, which validates native
identity, state ownership, entry membership, provider binding, and liveness before
returning the carrier. The consumer ends that borrow in the same call and retains no
raw pointer or storage view. The state is serialized and nonreentrant while such a
borrow is active. The private begin/end pair is authority available only to reviewed
trusted aggregate consumers: such a consumer must not reenter `borrow-end` or public
release from one of its own callbacks. End authenticates the exact state/handle pair,
not an application-visible per-borrow capability, so defect containment for a
malicious trusted consumer is not claimed; I2/C1 executable evidence must prove the
one-call begin/use/end discipline.

`state-dict-release!` is public, arity 1, and P1-specific. It rejects before mutation
when a scoped borrow is active. Otherwise its first transition invalidates the state
and all dependent state-entry and state-tensor identities before provider destruction begins, clears
every entry carrier, and releases the state ownership ledger exactly once. Repeated
calls are no-provider-callback, no-native-identity-allocation success. Forged,
copied, mutated, foreign,
unregistered, and wrong-kind values are `invalid-argument`; an exact registered dead
state or dependent handle used by any non-release accessor, load, borrow, or
serialization path is `invalid-state` before carrier dereference. Cross-state handle
pairs are `invalid-argument`. Detached copies
returned by `state-dict-paths` and `state-dict-alias-groups` remain ordinary data and
may outlive release. Small native identity tombstones may remain for process-lifetime
stale-authority rejection, but they retain no tensor carrier or storage.
The Eshkol shell registry likewise replaces every released state, dependent tensor,
and registered entry edge with a shared compact dead marker on successful release,
callback-defect cleanup, and post-shell construction rollback. An entry tombstone
overwrites both its raw-entry slot and owner-state slot with shared dead markers, so
no edge back to the released lifecycle graph remains. It does not retain the released
state, entry list, paths, shapes, devices, aliases, or carrier graph. The focused
gate includes a bounded probe that retains eight stale entry shells while dropping
their released states, checks every shell remains `invalid-state`, checks detached
path/alias copies remain usable, and requires native/provider live counts to stay at
the pre-loop baseline.
The pinned runtime supplies no proved finalizer, so callers must explicitly release
every successfully returned state. P1 neither hides abandoned states in a strong
snapshot registry nor treats GC reachability as native-storage reclamation.

## Strict atomic load

Loading never constructs a module and never performs a partial load. The loader:

1. validates format/version/features and every raw entry;
2. rejects duplicate, missing, unexpected, malformed, or incompatible paths;
3. validates the complete alias graph and exact tied values;
4. asks the selected provider to preflight and stage the complete batch of unique
   destination assignments without mutating a destination;
5. invokes exactly one provider commit for the complete prepared batch.

The internal provider 2.0 boundary stores eight compile-checked unary callbacks:
metadata description, independent clone, storage-identity comparison, exact-value
comparison, exact opaque-device comparison, whole-batch preparation, and whole-batch
commit, plus `release-owned!`. Requests and result envelopes are vectors. Registration is explicit,
process-local, append-only, and rejects provider-ID collisions. It performs no
environment lookup, dynamic load, or selection from untrusted state metadata.
The trusted Eshkol root deep-copies a private immutable descriptor and retains the
actual callbacks; native code stores seven legacy callback-identity slots plus one
release identity in the compatible 1.1 binding extension and never receives an
Eshkol closure, tensor, payload, plan, or descriptor pointer.
Modules retain the admitted private snapshot while logical state dictionaries store
only the inert versioned identity above.

The provider-level `storage-identical?` callback is the authoritative same-provider
physical-storage identity mechanism. After P1 prevalidates both carriers as live
values from the same exact provider, it must be total, deterministic, nonallocating,
nonraising, request-nonretaining, side-effect-free, and semantically infallible. It
returns true if and only if both carriers reach the same destructively releasable
allocation—release through either would invalidate the other—and false if and only
if their allocations are disjoint. The relation is reflexive, symmetric, and stable
for the serialized call. This definition includes distinct wrapper identities and
does not use raw data-pointer equality, which is insufficient for separately owned
zero-element tensors. `eq?` is only a fast exact-wrapper guard and is never proof of
storage independence. Malformed, foreign, dead, wrong-kind, or wrong-provider
carriers reject before the callback or native dereference. The boolean is solely an
ownership-safety predicate and grants no release authority.

P1 allocates one process-local five-slot comparison request before any callback can
publish an owner and reuses it only under the serialized, nonreentrant rule. It
validates that the callback preserved every request field, then clears all five
slots on success, raise, or malformed result. Every provider storage-identity
invocation, including ordinary post-publication and precommit alias checks, passes
through this lifetime pin. Clone publication also preallocates its two-slot
comparison outcome before callback entry. Before clone publication becomes
actionable, state adoption transfers ownership, or rollback/private release invokes
`release-owned!`, P1 applies exact and physical identity checks to every earlier
actionable owner in the newest-first ledger, borrowed inputs, and every carrier
protected by a live P1 state or registered module under that provider. Newer
candidates are sacrificed in favor of older genuine owners. Adoption and private
release pessimistically clear the candidate envelope before comparison and restore
it only after a valid false result proves independence. A true result leaves a protected alias disarmed
and rejected before destructive callback use, with its payload unchanged. A raise,
non-boolean result, request mutation, or reentrancy is a provider defect: P1 leaves
the ambiguous carrier disarmed, continues scanning later candidates, and reports the
defect without granting release authority. Proved-independent siblings are still
cleaned exactly once when their comparisons and release callbacks conform. P1 cannot
safely reclaim an ambiguous carrier and therefore makes no exact-cleanup or baseline-
restoration claim for a provider that violates this semantically infallible
obligation.

While that protected comparison is active, every externally callable public or
trusted P1 boundary that could observe, mutate, retain, transfer, borrow, or destroy
tensor/state authority rejects `invalid-state` before shell-registry, native, or
provider work. This includes state release and module parameter/buffer registration;
the callback cannot tombstone a compared live owner or retain the disarmed candidate
and then return a misleading result. Internal helpers belonging to the already
admitted outer operation remain available, so cleanup can still settle the candidate
after the callback exits.

A provider ID has stable cross-process format meaning: it permanently identifies
the tensor carrier representation, metadata/device equality, exact-value and clone
semantics, and prepare/commit protocol. Any incompatible carrier, encoding, device,
or copy semantic requires a new provider ID. Process-local collision rejection is a
defense, not proof of cross-process compatibility.

Registration requires two through sixteen nontrivial, storage-disjoint scratch
assignments. It proves that preparation leaves every source and destination
unchanged, commit mutates every destination in one batch while preserving exact
destination identity, sources remain unchanged, the exact plan-carrier vector is
returned, a second commit is rejected, and every admission clone is released. Clone
uses a mutable four-slot request envelope and may publish ownership in slot 3 exactly
once. That slot is single-assignment: a callback may never clear or replace a
publication, including before raising. Any owned value overwritten before P1 can
observe it would be an unrepairable provider leak, so each integrating provider must
prove this rule in executable evidence. Each successful publication must have a new exact carrier
identity: it cannot be any borrowed carrier in the complete precomputed state or
admission-evidence input set, the request or its borrowed fields, or an earlier
owned publication. P1 rejects those ownership-confusion cases before
making the new envelope actionable, so it never releases a live input and releases
each prior physical owner once under a conforming provider. The check includes
carriers in every other live P1 state and every registered module parameter/buffer,
not only the current operation.
Generic provider admission first validates every evidence carrier, then checks each
exact wrapper against itself twice and every distinct evidence pair in both
directions twice through the preallocated protected request. This rejects a raise,
request mutation, non-reflexivity, asymmetry, instability, or false disjointness
before clone publication, preparation, or commit. Each real integrating provider,
including I2, must additionally prove same-allocation distinct-wrapper truth,
request nonretention, no allocation/write/raise, invalid-input rejection, and native
owner identity—including distinct zero-element owners—in carrier-native executable
evidence before use; P1's inert fixture does not prove another provider's owner
model.

I2 has confirmed that its implementation can canonicalize each prevalidated wrapper
to the live owning `et_f32_tensor` resource. It will not compare Eshkol wrapper
identity or raw data pointers: two distinct zero-element tensor owners remain
disjoint even when both data pointers are null. That compatibility disposition is a
design commitment, not P1L acceptance or executable I2 integration evidence. I2 has
not rebased or changed its runtime and remains blocked until P1L is independently
approved and merged. See the [frozen P1L equivalence decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5502247288)
and [I2 confirmation](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/49#issuecomment-5502269486).

Release consumes one P1-owned carrier, returns the exact same
preallocated request envelope as acknowledgment, and is admitted as nonallocating,
nonraising, nonreentrant, and semantically infallible after P1 prevalidation.
Substitution/allocation of a replacement,
a raised error, a partial result, or reentrant registration rejects admission
atomically. Failed admission revokes every newly created callback identity and the
unpublished provider identity, restoring the exact native live-entry baseline;
tombstones are permanent and cannot regain authority.

Every potentially failing operation belongs to preparation. A conforming commit is
one-shot, infallible, nonallocating, and preserves destination storage identity; a
violation is an internal provider defect rather than a recoverable state load. P1
does not claim that the compiler proves those semantic properties. Each production
provider must supply its own reviewed executable evidence before registration is
permitted by its integrating workstream.

P1 clears an ownership ledger before invoking release, continues through all
remaining owners if a defective callback raises or returns malformed acknowledgment,
and reports the first defect without retrying a consumed carrier. This applies to
clone validation, partial state creation, provider-admission scratch, failed or
successful strict-load scratch, C1 decode/adoption rollback, and explicit state
release. A provider release callback is never called for live module parameters,
live gradients, live optimizer storage, or caller/provider-owned module inputs.
Module registration borrows its already-live carrier for the module lifetime because
P1 has no module-release API; the integrating constructor/provider remains its owner.

The trusted aggregate exposes only the fixed-arity provider preflight, owned-carrier
admission, and single-envelope release mechanisms needed by reviewed consumers such
as C1. Admission disarms a newly published candidate, validates it with the exact
provider, compares its physical storage with every earlier actionable owner and all
protected live P1/module storage, and rearms only a unique carrier; it never releases.
A later
O2-specific wrapper may reuse the preflight and release mechanisms only after validating its
own optimizer-owner ledger and hard-coding its reviewed provider identity; it may not
accept an application-selected provider or an envelope for P1 state, live parameter,
gradient, or optimizer-moment storage owned elsewhere. O2 still needs its own
versioned optimizer-state receiver, liveness/handle checks, and public release
operation. P1 does not add that sixth optimizer operation or a generic dispatcher.
As a last-chance invariant, the private release mechanism first clears and rejects
an envelope naming the exact wrapper of any live P1-state or registered-module
carrier. After provider preflight it disarms the envelope, performs the authoritative
`storage-identical?` scan over those protected carriers, and rearms only a proved-
independent owner before calling `release-owned!`. This protects a decoder that fails
before adoption as well as a future O2 wrapper; optimizer moments and gradients
remain O2's own ledger responsibility.

After the first destination mutation begins, P1 invokes no callback other than the
single admitted commit and takes no recoverable branch. It checks only that commit
returned the already validated exact plan identity; mismatch is a trusted-provider
`internal` defect, not a recoverable state branch. Before commit, P1 also rejects
clone/source aliasing, aliases between independently
registered leaves, aliases between state snapshots (including tied logical paths),
and aliases between distinct batch destinations.

## Private identity bridge 1.1

The bridge requires 64-bit pointers and fixed-width `i64` calls. Compile-time
assertions freeze the caller token, private registry record, context, and error-record
sizes at 264, 256, 344, and 272 bytes. The public archive defines only four read-only functions: ABI major,
ABI minor, token kind, and token liveness. It contains no context, constructor,
provider, callback, bind, revoke, result, error, or generic-dispatch symbol. All 31
fixed-arity private functions have hidden ELF visibility and exist only in the
trusted replacement archive.

Native scope is limited to opaque identity/capability storage: token creation and
recognition, bridge-owned immutable provider identity bytes and callback-identity
snapshots, exact state/provider binding, cleanup/revocation, and deterministic
live/tombstone counts. C performs no schema validation, list traversal, tensor
operation, load preparation/commit, serialization, model logic, or capability
inference.

Provider IDs arrive only after Eshkol validates their semantic `symbol` contract.
The ABI stores an exact unsigned length plus exact bytes, with
`ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES` fixed at 127. Zero length is allowed because
the existing P1 symbol contract does not exclude an empty symbol. Nonzero lengths
require a nonnull pointer; negative/oversized spans fail without truncation. The
bridge never calls `strlen`, interprets NUL, applies a locale/normalization/UTF-8
rule, or exposes a mutable pointer to its owned bytes.

Tokens carry nonzero 128-bit `getrandom` identities, exact process ownership, kind,
and duplicated integrity data. Allocation or entropy failure is explicit and has no
fallback. Foreign, copied, mutated, wrong-kind, cross-context, stale, transplanted,
or post-fork tokens reject before mutation. Unpublished-provider abort and callback
revoke are cleanup-only operations: they reject sealed/published/referenced or stale
tokens and leave permanent tombstones, so allocator address reuse cannot restore
authority. Arbitrary malicious native object injection is outside the Eshkol-module
threat model; arbitrary compiled Eshkol linked only with the public package cannot
reach the private authority.

## Error mapping

| Condition | Category |
|---|---|
| Wrong opaque receiver; malformed or unknown accessor path | `invalid-argument` |
| Missing/unexpected path, entry kind, tree or alias-topology mismatch | `shape-mismatch` |
| Tensor/entry/destination shape mismatch | `shape-mismatch` |
| Tensor/entry/destination dtype mismatch | `dtype-mismatch` |
| Tensor/entry/destination device mismatch | `device-mismatch` |
| Non-dense, offset, strided, or unsupported layout | `noncontiguous` |
| Unbound/unknown/unverified provider, dtype, copy mode, gradient slots, or runtime capability | `unsupported` |
| Malformed format/version/provider fields, entry metadata/features, duplicate path, invalid alias group, conflicting tie | `corrupt-data` |
| Well-formed unsupported format, major/minor version, or required feature | `version-mismatch` |

Validation precedes mutation for every category.

## Measured capability limit

Merged R0 evidence does not verify canonical Eshkol f32/i64/bool storage, device
identity, contiguity, tensor cloning, tensor equality, gradient slots, or in-place
tensor copying. Production P1 therefore registers no provider. The default internal
construction path has no tensor provider, and every tensor-dependent public state
operation fails explicitly with a same-registry structured `unsupported` error.
Malformed public receivers fail with structured `invalid-argument`; both paths use
the unchanged A0 accessors and taxonomy above.

Structural tests explicitly add the separate `tests/p1/providers` include root and
compose production modules with an inert `fixture-v1` carrier from the qualified
`p1_test.tensor_provider` module. The implementation, constructors, symbols, and
fault injection live entirely below that test root. Canonical production builds use
only the `lib` include root; depfile, object-symbol, object-string, and negative
compile checks prove that the fixture module is absent from the production graph and
cannot be selected through `transformer.module`. The fixture payload contains only
exact inert atoms. It is not f32/i64/bool tensor storage, a device or backend, a
production constructor, capability-discovery evidence, or a fallback.

The injected later-entry failure proves the P1 validation/staging/commit control
flow and logical ownership semantics only, not physical runtime tensor storage,
copying, lifetime, mutation, or atomicity. Real tensor-backed module construction,
snapshot ownership, storage mutation, and state operations remain explicitly
unsupported until a reviewed provider supplies and proves the required metadata,
clone, exact-value comparison, whole-batch preflight, and infallible commit contract.
P1 must be retested against merged I1/K1 and later f32 tensor support before any
numerical or module-capability claim. There is no scalar, CPU, dtype, Python,
finite-difference, device, or numerical fallback.

The native registry has no verified concurrent-mutation behavior. P1 therefore
requires serialized, nonreentrant state use and proves the scoped begin/end rule at
the trusted boundary. State, state-scoped entry, and state-tensor shells become inert
tombstones on explicit release; no finalizer or GC reachability is assumed. The registry never
retains an Eshkol state vector, entry, tensor snapshot, carrier, raw storage pointer,
or closure. Strict load constructs no temporary expected state or native binding, so
rejected and repeated loads add no registry entry. Later I2, O2, and C2 work must
retain this explicit lifetime discipline and independently prove their own receiver
ownership contracts.

## Coordination records

The proposed contract is tracked in [P1L issue #51](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51).
Provider 2.0 and the release envelope were coordinated in the
[I2/P1L compatibility decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5488110508).
The exact storage equivalence and fail-closed rule are frozen by
[integration comment 5502247288](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5502247288),
with downstream compatibility recorded on
[I2 issue #49](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/49#issuecomment-5502269486)
and the matching [issue #51 disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/51#issuecomment-5502293490),
then acknowledged on [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5502293552).
The lifecycle taxonomy is fixed by
[integration comment 5493574404](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493574404),
and the narrow O2 reuse boundary by
[integration comment 5493412398](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493412398).
PR #55 remains subject to the
[P1L-R request changes](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/55#issuecomment-5501171044).
