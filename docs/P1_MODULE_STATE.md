# P1 module and state-tree contract

## Boundary

P1 implements the existing A0 `transformer.module` names and arities. It adds no
public constructor, registration operation, tensor descriptor, state-version
accessor, or serialized format. Library construction and malformed-state helpers
remain in the explicitly unstable `transformer.module_internal` implementation
surface.

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
admission results, load plans, or capability evidence. A runtime state object has a
private lightweight binding token outside these logical fields. A process-local
side table maps that token—not the state or any tensor—to an admitted provider.
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
Modules store the registered provider descriptor while logical state dictionaries
store only the inert versioned identity above.

A provider ID has stable cross-process format meaning: it permanently identifies
the tensor carrier representation, metadata/device equality, exact-value and clone
semantics, and prepare/commit protocol. Any incompatible carrier, encoding, device,
or copy semantic requires a new provider ID. Process-local collision rejection is a
defense, not proof of cross-process compatibility.

Registration performs a structural empty-batch admission probe. Preparation must
return one plan-carrier vector and commit must return that exact vector identity;
substitution/allocation of a replacement, a raised error, or reentrant registration
rejects admission atomically. This probe does not prove nonempty behavior,
nonallocation, nonraising behavior, or physical tensor atomicity. Before I1/K1/C1
registers a production provider, that workstream must supply reviewed executable
evidence that nonempty commit is one-shot, nonraising, nonallocating, nonreentrant,
infallible, preserves destination storage, and returns the same validated plan.

Every potentially failing operation belongs to preparation. A conforming commit is
one-shot, infallible, nonallocating, and preserves destination storage identity; a
violation is an internal provider defect rather than a recoverable state load. P1
does not claim that the compiler proves those semantic properties. Each production
provider must supply its own reviewed executable evidence before registration is
permitted by its integrating workstream.

After the first destination mutation begins, P1 invokes no callback other than the
single admitted commit and takes no recoverable branch. The commit result is not
interpreted after mutation; exact carrier return is an admission invariant. Before
commit, P1 also rejects clone/source aliasing, aliases between independently
registered leaves, aliases between state snapshots (including tied logical paths),
and aliases between distinct batch destinations.

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

The process-local binding side table has no verified concurrency behavior and keeps
one lightweight token/provider pair for each bound state until process exit. It does
not retain the state vector, entries, or tensor snapshots, but its small metadata
cost is monotonic. I1/C1 must retest binding lifetime and synchronization if the
runtime later supplies weak references, finalizers, or concurrent mutation.
