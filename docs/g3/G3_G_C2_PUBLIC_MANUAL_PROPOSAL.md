# G3-G public manual facade prerequisite proposal

**Prerequisite only; no public API, package revision, export or implementation is
accepted here.** The [A0 generator contract](../PUBLIC_API_CONTRACT.md#13-generator)
requires `generator-prefill!` and `generator-decode-step!` to return newly
owned readable floating `[N,V]` tensors. The accepted source-private
[P1 prefill](G3_M_PREFILL_P1_LEAF.md),
[P2 prefill](G3_M_PREFILL_P2_LEAF.md) and
[one-token decode composer contract](G3_M_MANUAL_DECODE_COMPOSER_PROPOSAL.md)
return detached G3-T kind-3 logits shells, which do not yet meet that public
result contract. The private decode [candidate](G3_M_MANUAL_DECODE_LEAF.md)
has been independently source-reviewed by root; its prior head passed all
23 hosted checks. Refreshed hosted CI and merge remain pending. Revision-2
G3-G continues to expose only its thirteen existing names; this proposal
does **not** install the two A0 manual names or widen
`generation-tensor-release!`.

## Observed ownership and missing seam

`src/eshkol_transformer/g3t_transport.c` stores an `et_f32_tensor *` inside
an authenticated G3-T logits record and creates it with I2 shape `[1,256]`.
The private composers publish an Eshkol shell registered only as G3-T kind
`logits`; the accepted native typed release can destroy it. Production
`g3t_transport.h` has reserve/release and frame operations but no authenticated
logits read, clone, transfer, or public I2 tensor handle. Its logits-word and
borrow observers are guarded by `ET_G3T_TESTING`, so private numerical tests
cannot establish a readable installed result.

The existing M3T diagnostic logits owner is a different registry and an I2
rank-three `[1,2,256]` tensor. Its `diagnostic-logits-bits` copies 2,048
bytes after M3T identity admission; it cannot accept a G3-T shell or satisfy
A0's `[1,256]` result by casting, sharing an internal pointer, or changing
shape silently. The lower I2 `et_f32_tensor_copy_bits_to_v1` and clone/borrow
primitives provide storage mechanisms, not a G3-T authorization or an
installed floating-tensor interface. The revision-2 generation output
accessors accept only opaque generation outputs and provide no manual logits
route.

Before either A0 manual name may be proposed as a public export, implement
and independently review the separate [source-private G3-T exact-bit
materialization proposal](G3_T_MANUAL_LOGITS_MATERIALIZATION_PROPOSAL.md).
It admits only a detached live kind-3 result by exact G3-T registry identity,
checks CPU f32 `[1,256]` storage and no active borrow, and copies 1,024 exact
bit bytes to a caller-owned bytevector. Forged, foreign, wrong-kind, pending,
dead, busy, borrowed and malformed owners reject before destination writes.
Failure preserves the source and cache and leaves no lease; a previous result
retains its ownership and release semantics. This private bytevector is still
not a public floating tensor.

A separate public tensor contract must then say how the result is represented,
read, cloned or borrowed, released, and interchanged with other tensor APIs.
It must give callers the promised newly owned CPU f32 `[1,256]` value with
exact bits and independent lifetime after generator close or cache
replacement. Returning a G3-T kind-3 shell alone, or only a diagnostic
bytevector, does not satisfy A0. No future public signature, boxed symbol or
export count is frozen until these two contracts have passing implementation
evidence.

## Conditional manual facade after the result gate

Once materialization and public tensor ownership are accepted, a bounded
successor to the [revision-2 G0/G1 package](G3_G_C2_PUBLIC_G0_PROPOSAL.md)
may add the A0 arity-two `generator-prefill!` and
`generator-decode-step!` names. It needs a **new fixed package tuple** and
distinct artifact directory; revision 2's root does not load
`native/g3t_manual_p1_extension.esk`,
`native/g3m_prefill_p1_extension.esk`,
`native/g3m_prefill_p2_extension.esk`, or
`native/g3m_manual_decode_extension.esk`. A future root must source-compose
those exact accepted files and its materialization source once, retain one
G3-T/M3 registry owner, and pin all source/native/object/export/undefined/
string manifests and feature flags. No v1/v2/v3 archives may be linked
together. Sequential default builds must preserve earlier installed
artifacts byte-identically. Package stem, public boxed symbols, counts and
revision number will be fixed in that later contract.

The future prefill dispatch can use revision-2 exact-shell input provenance
only to choose the accepted P1 or P2 composer; native still authenticates
owned I1 shape, byte values, same-aggregate identity and unborrowed state.
Scalar input is P1; authentic sealed T1-backed two-byte input is P2. Manual
calls admit budget 0 or 1 without a draw or generate-only full-request
preflight. Prefill replaces an existing cache/binding without advancing RNG.
Decode admits only one committed P1 prefix, after manual P1 or P1/G0, and
appends exactly position one; a missing/full/stale/busy prefix rejects. A
subsequent manual prefill may replace the cache. Neither call creates a
generation output, and `generator-generate!` still rejects an existing
prefix. P2 decode, repeated append and generation continuation remain open.

A public wrapper cannot allocate or convert the returned tensor after cache
commit unless an accepted transaction contract proves failure atomicity.
The required future design must reserve/root the readable result before the
private commit or otherwise prove an atomic cache-and-result publication.
On every precommit failure it must abort the acquired call, drain fourteen
pins, preserve entry cache/binding/RNG and older detached outputs, release
any pending materialization, and rethrow the first bounded E1 source
category/code with the invoked public operation and `cause #f`. Native
commit/finish and impossible postcommit failure remain fail-stop. Manual
result release must remain separate from generation-output accessors.

## Completion gates and deferrals

First finish refreshed hosted CI and merge for the reviewed private decode
candidate. The materialization gate
must test exact 256-word bits against independent M3T after P1, P2 and decode,
I2 shape/dtype/device/layout, disjoint ownership and survival after generator
close/cache replacement, plus wrong-kind/foreign/dead/pending/borrowed owner,
allocation/copy/cleanup cuts and normal/repeat/sanitizer runs. The public
tensor gate must prove a caller can read those same 256 values through the
installed API and release the tensor independently. Only then design and
review the new package: fresh-cache AOT public callers, old-package isolation,
exact manifests, feature-off/private-symbol negatives, Q0, supported CI,
P1/P2/G0/G1 transition and rollback tests, and bitwise K/V/mask parity.

Until these gates pass, no public manual facade, A0 conformance, output
interop, continuation, capacity above two, N>1, persistence or CLI is
claimed.
