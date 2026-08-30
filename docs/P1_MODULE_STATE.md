# P1 module and state-tree contract

## Boundary

P1 implements the existing A0 `transformer.module` names and arities. It adds no
public constructor, registration operation, tensor descriptor, state-version
accessor, or serialized format. Library construction and malformed-state helpers
remain in the explicitly unstable trusted replacement root at
`internal/p1/lib/transformer/module.esk`. The two Eshkol roots are mutually exclusive generated products of one
canonical template. The public root provides exactly the 17 A0 bindings and never
imports or references the trusted root; the trusted root independently provides the
same 17 bindings plus the pre-existing construction/provider helpers.

Pinned Eshkol flattening cannot protect a provider seam inside one compiled source
closure. P1 therefore uses a narrow native identity bridge, version 1.0, and two
alternative archives with the same basename. The normal build emits only the public
archive. Trusted P1 tests and future reviewed C1/I1 integration explicitly build and
link the private replacement archive instead; the archives are never linked
together, and the private archive is not an installed public library.
The exact layout, ownership, status, and symbol contract is frozen in
[P1_IDENTITY_ABI.md](P1_IDENTITY_ABI.md).

The Eshkol-facing error dependency is `transformer.error_internal` from E1/#23.
P1 constructs no subsystem-local error type. Every P1 error carries a data-only
proper alist in `details`, uses `#f` when no cause exists, and is inspected through
the unchanged A0 accessors. P1 never forwards an untrusted offending value into E1;
its current errors use the empty details alist and place the diagnostic in the
operation and message fields.

## Paths, nesting, and identity

A logical path is a nonempty list of nonempty valid UTF-8 strings. Ordering is
lexicographic by the encoded UTF-8 bytes of each segment, segment by segment; if all
shared segments compare equal, the shorter path sorts first. Registration order,
host locale, and display spelling do not affect enumeration.

Parameter, buffer, and child names share one local namespace. Duplicate names,
shared children, cycles, and cross-root parameter ties are rejected. Enumeration
flattens the complete nested module tree and then applies the global path ordering.
Registration copies caller-owned name strings. A root tree is finalized on its first
public enumeration or state access, after which every registration operation fails
with `invalid-state`; attached child modules are not independent public roots.

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
  `(transformer-tensor-provider 1 0 provider-id)`;
- a raw ordered entry list, preserving duplicates for validation;
- a canonical parameter-only alias graph.

Each entry contains its logical path, kind (`parameter` or `buffer`), redundant
shape/dtype/device/layout metadata, and one owned tensor snapshot. Metadata never
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

`state-dict-bind-provider-internal!` is the trusted C1/I1 handoff. Its caller must
explicitly name an already admitted provider chosen through trusted policy and
capability validation. Before adding a binding, it validates the exact state and
provider-interface versions, provider ID, every tensor/metadata entry, storage
independence, and aliases using that caller-selected provider. Rebinding the same
state/provider is identity-preserving and idempotent. Conflicting, unknown,
unadmitted, malformed, or incompatible bindings fail through E1 without changing
the existing binding. Unbound states cannot be inspected or loaded and fail
`unsupported`. State metadata is never used to select or load executable code.

## Strict atomic load

Loading never constructs a module and never performs a partial load. The loader:

1. validates format/version/features and every raw entry;
2. rejects duplicate, missing, unexpected, malformed, or incompatible paths;
3. validates the complete alias graph and exact tied values;
4. asks the selected provider to preflight and stage the complete batch of unique
   destination assignments without mutating a destination;
5. invokes exactly one provider commit for the complete prepared batch.

The internal provider-v1 boundary stores seven compile-checked unary callbacks:
metadata description, independent clone, storage-identity comparison, exact-value
comparison, exact opaque-device comparison, whole-batch preparation, and whole-batch
commit. Requests and result envelopes are vectors. Registration is explicit,
process-local, append-only, and rejects provider-ID collisions. It performs no
environment lookup, dynamic load, or selection from untrusted state metadata.
The trusted Eshkol root deep-copies a private immutable descriptor and retains the
actual callbacks; native code stores only seven inert callback-identity tokens and
never receives an Eshkol closure, tensor, payload, plan, or descriptor pointer.
Modules retain the admitted private snapshot while logical state dictionaries store
only the inert versioned identity above.

A provider ID has stable cross-process format meaning: it permanently identifies
the tensor carrier representation, metadata/device equality, exact-value and clone
semantics, and prepare/commit protocol. Any incompatible carrier, encoding, device,
or copy semantic requires a new provider ID. Process-local collision rejection is a
defense, not proof of cross-process compatibility.

Registration requires two through sixteen nontrivial, storage-disjoint scratch
assignments. It proves that preparation leaves every source and destination
unchanged, commit mutates every destination in one batch while preserving exact
destination identity, sources remain unchanged, the exact plan-carrier vector is
returned, and a second commit is rejected. Substitution/allocation of a replacement,
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

After the first destination mutation begins, P1 invokes no callback other than the
single admitted commit and takes no recoverable branch. It checks only that commit
returned the already validated exact plan identity; mismatch is a trusted-provider
`internal` defect, not a recoverable state branch. Before commit, P1 also rejects
clone/source aliasing, aliases between independently
registered leaves, aliases between state snapshots (including tied logical paths),
and aliases between distinct batch destinations.

## Private identity bridge 1.0

The bridge requires 64-bit pointers and fixed-width `i64` calls. Compile-time
assertions freeze the private token, context, and error-record sizes at 264, 344,
and 272 bytes. The public archive defines only four read-only functions: ABI major,
ABI minor, token kind, and token liveness. It contains no context, constructor,
provider, callback, bind, revoke, result, error, or generic-dispatch symbol. All 25
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
operation fails explicitly with `unsupported`.

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

The native registry has no verified concurrency behavior and makes no concurrency
claim. Successful opaque shells remain process-local until explicit trusted cleanup
or process exit; it never retains an Eshkol state vector, entry, tensor snapshot, or
closure. Strict load constructs no temporary expected state or native binding, so
rejected and repeated loads add no registry entry. I1/C1 must retest lifetime and
synchronization if the runtime later supplies weak references, finalizers, or
concurrent mutation.
