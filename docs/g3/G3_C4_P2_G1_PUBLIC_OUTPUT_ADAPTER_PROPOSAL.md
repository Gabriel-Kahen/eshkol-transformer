# G3-C4 P2/G1 exact-owner public output adapter

**Contract for independent review; no public C4 output adapter is implemented.**
Base: accepted public C4 model/input `9581b8a` and reviewed, supported-gated
private coordinator `9f41d1a`/tree `b31e5fb`. This leaf adapts only an already
published C4 `(P,G,N)=(2,1,1)` shell. It does not enable public
`generator-create` or `generator-generate!`; those require a later composed
profile/producer dispatch. The public diagnostic-C2 admission and the
existing `lib/transformer/generation.esk` names and arities stay unchanged.

## Existing read and missing typed owner

`g3c4-p2g1-output-entry-live` authenticates the 12-slot C4 shell and its
protected record, including the exact native owner and owned raw `[1]`
bytevector. `g3c4-p2g1-output-copy-internal` then returns a detached fresh
five-vector `(raw-copy, selected-id, 1, 3, rng-snapshot)` from the published
native I1 `[1]`. Its seven-word native read authenticates the output,
requires an unborrowed live I1, checks raw/ID parity, and retains no borrow.
The private result is mutable data, **not** a public tensor or RNG owner.
`et_g3t_private_output_*_clone_v1` cannot be used: those stems authenticate
G3-T records and enforce the C2 cache bound. C4's input transport owns I1
`[1,1]` or `[1,2]`, so relabeling it as a result would also violate shape.

The adapter uses the already accepted public arity-one accessors:

| Public call | Exact C4 result |
|---|---|
| `generation-output-ids` | Fresh singleton list of a new owned CPU I1 `i64[1]` holding the selected ID `[0,255]` |
| `generation-output-lengths` | New owned CPU I1 `i64[1]` holding `1` |
| `generation-output-cache-lengths` | New owned CPU I1 `i64[1]` holding `3` |
| `generation-output-text` | Fresh singleton list of the copy-out's detached raw bytevector `[1]` |
| `generation-output-rng` | New opaque, immutable, explicitly releasable C4 RNG owner with the four copied successor words |

Each invocation calls the reviewed copy-out before constructing its public
result. Independent calls return distinct tensor/RNG shells and storage, or
distinct raw bytevectors. The output, generator, each clone and each raw
copy have independent lifetimes. A prior clone remains readable/releasable
after output release or generator close; release of one clone does not alter
another, the output, cache or generator RNG. Returned text is caller-owned
bytes and may be mutated without changing a later read.

## Bounded native clone prerequisite

Under a new `ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE` feature, requiring C4
publication and copy-out, add only source-private native constructors and
release. `et_g3c4_private_result_tensor_create_v1(kind,value)` accepts exact
selectors `ids`, `lengths`, `cache-lengths` (numeric selectors 5, 6, 7 in
this private ABI), respectively requiring `[0,255]`, `1`, `3`. It allocates a
registered C4 transport control of that kind plus a fresh rank-one I1
`i64[1]` through `et_i64_tensor_create_v1` and `copy_from_v1`; no C2 owner or
source output pointer is retained. `et_g3c4_private_result_tensor_release_v1`
authenticates that exact registered kind, rejects an active I1 borrow before
mutation, destroys the I1, then tombstones the retained control. Exact dead
release is idempotent. Any allocation/copy failure destroys unexposed I1
storage and enrolls no live clone; cleanup failure is fail-stop.

`et_g3c4_private_result_rng_create_v1(w0,w1,w2,w3)` admits the copied
snapshot schema (`w0=1`, `w1>=0`, four exact signed i64 words) and allocates
a fresh registered C4 RNG through the existing C4 RNG allocator/enrollment
path. Its words are never mutable through a public operation. The existing
`et_g3c4_private_rng_release_v1` releases that native owner. There is no
generic tensor constructor, C2 relabeling, second C4 registry-owning archive,
or new installed **public** ABI. The Eshkol C4 transport registry records exact
clone shells; an auxiliary protected clone record pins their native identity,
kind and original scalar/words so changing a mutable shell slot cannot
redirect release to another clone. Public tensor/RNG release selects only
these exact clone records and existing authentic C4 input owners; a C4 model,
generator or output is a wrong tensor/RNG kind.

## Retained-device prerequisite and error order

A0 requires `device-mismatch` when retained output storage has an
inconsistent device. Today the C4 native copy-out reports malformed I1
invariants as generic `internal`, and it does not expose a device-specific
result. The adapter therefore needs one separately reviewed, feature-gated,
nonmutating native inspection, provisionally
`et_g3c4_private_output_device_state_v1(output)`: `0` means exact live
published output with an unborrowed owned I1 and CPU view, `1` means that
same authenticated I1 view reports a non-CPU device, and `-1` means an
ordinary typed native rejection with the existing last-error triple. It
must borrow/end I1 within the call, preserve the first error on cleanup,
and never alter output, cache, RNG or carrier bytes. The Eshkol adapter maps
only `1` to public `device-mismatch`; it never guesses from generic
`internal`. Rank/dtype/layout/shape corruption that prevents reliable
inspection follows the existing typed native invariant error unless a later
separately reviewed discriminator is added. A held I1 borrow remains
`invalid-state`, not `device-mismatch`. The normal I1 constructor is CPU-only;
test-only descriptor corruption is required to prove the distinct branch.

Each public accessor first tests exact C4 transport **and separate C4 model**
membership (including dead entries), then kind, then liveness and protected
output identity. An
authentic dead C4 output is `invalid-state`; authentic wrong kind and forged
or copied shells are `invalid-argument`. C4 model membership is always wrong
accessor kind, even when its transport registry has no entry. Only values
absent from both C4 registries take the
unchanged G3-T/C2 path. The C4 branch checks retained device, calls the
private copy-out, validates `ID=raw[0]`, lengths `1/3` and RNG schema, then
allocates its detached result. Other violated native owner invariants keep
their bounded `internal` error; malformed caller/carrier values retain the
accepted copy-out category. All public errors use the invoked accessor or
release name, bounded data-only details and `cause #f` through the existing
E1 boundary. The C2 branch's validation order and `P+G<=2` errors do not
change.

## Publication, rollback and release

The public E1 boundary is **not** an outer `m3-call`. A first guarded
read-only scope authenticates the output and checks device; the reviewed
`g3c4-p2g1-output-copy-internal` then runs in its own existing `m3-call`;
finally a sibling `m3-call` reauthenticates the output and constructs the
public value. No `m3-call` is nested around copy-out. These scopes remain
one synchronous externally serialized public invocation, with no caller
callback between them. The final scope preallocates its shell/entry/guard
ledger and return container before native clone creation, and enrolls a
pending C4 clone only with cleanup armed. It checks the exact
protected output/clone links before marking the clone live, and returns no
partially owned list or shell. Allocation, copy-out, device inspection,
native clone or final recheck failure releases any newly created native
clone, scrubs private raw/staging copies where applicable, tombstones the
pending Eshkol entry and leaves the published output unchanged. A caught
foreign exception maps to `internal`; a cleanup failure is fail-stop. Native
clone construction is the only new ownership publication and occurs after
successful detached read. No callback or fallible operation follows the
final Eshkol live-mark/return cut.

`generation-output-release!` routes exact C4 output membership to the
accepted `g3c4-p2g1-output-release-internal!`: native release first, then
shell/record tombstone; exact dead release succeeds, while an external I1
borrow rejects recoverably before mutation. `generation-tensor-release!`
routes exact C4 result-tensor clones to the new native-first release and
keeps existing C4 input release; `generation-rng-release!` routes exact C4
output RNG clones to native-first release. Each checks the protected native
identity before invoking native release. Cross-output or cross-clone owner
swaps cannot release an innocent owner. Non-release access to dead owners is
`invalid-state`; wrong release kind is `invalid-argument`.

## Acceptance gate and limits

Use a same-archive private P2/G1 producer to supply two distinct authentic
outputs to the **public accessor wrappers** without enabling public
`generator-generate!`. Greedy and nontrivial categorical cases must prove
fresh CPU I1 shapes/values, ID/raw parity, lengths `1/3`, all Philox words
against the independent oracle, two-call isolation, clones after output
release/generator close, and exact repeated release. Test forged/copied,
wrong-kind, dead, cross-profile, cross-output native/raw/clone swaps, held
output and clone I1 borrows, device-only corruption, other malformed I1
descriptors, raw/ID mismatch, native/Eshkol allocation and copy/decode
failures, no leaked lease/partial public result, output unchanged and
same-output retry. Source-only feature-off must omit all new private clone
and device symbols. Re-run C2 P1/G1, G0 and manual public behavior unchanged.
Verify one registry-owning package archive, exact source/export/undefined/
string manifests, no added public names, private-link negatives, fresh AOT
caller, and pinned fe9 Clang21 normal/repeat/ASan-UBSan evidence with exact
head/tree and sealed logs before claiming the adapter accepted.

Public C4 generator-create/generate dispatch and return are a later joint
integration gate. EOS, repeated `G>1` decode, `N>1`, cache/no-cache parity,
save/reload and CLI3 `generate` are outside this contract.
