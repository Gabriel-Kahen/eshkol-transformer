# G3-G diagnostic-C2 public manual tensor and facade proposal

**Conditional revision-3 contract; no public implementation or package is
accepted here.** The [A0 generator contract](../PUBLIC_API_CONTRACT.md#13-generator)
requires `generator-prefill!` and `generator-decode-step!` to return newly
owned floating `[N,V]` values. This proposal defines a bounded, readable CPU
f32 `[1,256]` tensor result for those exact names. It does not claim general
A0 shape/device support or generic tensor interop. Revision 2 still installs
only thirteen generation names. The private decode
[candidate](G3_M_MANUAL_DECODE_LEAF.md) was independently source-reviewed by
root and merged as `bf6c284` after 23 exact-head hosted checks.

## Fixed-profile public tensor decision

`src/eshkol_transformer/g3t_transport.c` stores an `et_f32_tensor *` inside
an authenticated G3-T logits record and creates it with I2 shape `[1,256]`.
The private P1/P2 composers and source-reviewed decode candidate publish a
newly owned Eshkol shell registered as G3-T
kind `logits`; the accepted native typed release destroys it. It is not yet an
installed result. The accepted M3 bounded implementation treats its
authenticated, owned f32 `[1,2,256]` `model-output-logits` shell as a tensor,
with separate `diagnostic-model-logits-bits` readout and typed release
([M3 packaging design](../m3/packaging-design.md),
[A0 bounded M3 coverage](../PUBLIC_API_CONTRACT.md#bounded-m3-implementation-coverage)).
That precedent permits a fixed-profile tensor contract, not an unreadable
shell or a bytevector substituted for the tensor.

M3T's owner is a different registry and rank-three tensor; it cannot accept a
G3-T shell. G3-T now has a source-private `(g3t-logits-bits logits)` path that
authenticates detached live kind-3 identity, verifies CPU f32 `[1,256]`,
dense strides `[1024,4]` and no active borrow, then returns a fresh 1,024-byte
exact-bit bytevector. Its pinned Ubuntu 22.04/LLVM 21 full private runner
passed 47,358 identical normal/repeat/sanitizer checks, Q0 4/4 and closed
source checks at `a22572e` / tree `2834a68`; the focused native gate passed
47 identical checks. Exact commands and logs are sealed under
`/home/gabe/.codex/evidence/eshkol-transformer/g3t-logits-materialize-pinned-a22572e-20260928/`
(`SEAL.sha256`: `980c7e46...`). This is private execution evidence, not a
public facade. #143 materialization hosted integration remains pending.

The [private materialization proposal](G3_T_MANUAL_LOGITS_MATERIALIZATION_PROPOSAL.md)
supplies the authenticated read seam. Revision 3 can conditionally define
the exact kind-3 shell itself as its bounded newly owned floating tensor and
install a distinct diagnostic bit accessor. A bytevector alone is never the
manual result. Generic I2 carrier construction, clone/borrow and interop are
deferred; no unreviewed cross-registry cast supplies authority.

## Conditional revision-3 surface and ownership

| Installed operation | Arity | Bounded result |
|---|---:|---|
| `generator-prefill! generator input-ids` | 2 | Fresh detached kind-3 shell owning immutable contiguous I2 CPU f32 `[1,256]` last-position logits. |
| `generator-decode-step! generator token-ids` | 2 | The same new owner kind for the appended position. |
| `diagnostic-generation-logits-bits logits` | 1 | Fresh detached 1,024-byte little-endian bytevector of all 256 binary32 words in row-major order, without conversion or normalization. |
| `generation-tensor-release! tensor` | 1 | Existing typed release widened only to kind-3 manual logits alongside revision-2 input and output-clone kinds. |

Each successful manual call returns the **tensor shell**, never the diagnostic
bytevector or a generation output. The shell's rank, shape, dtype, device,
layout, strides, immutable value and sole ownership are fixed by this
profile. The installed bit accessor is the observational read path and may
be called repeatedly while the shell is live. Its bytes are independent of
later cache replacement, generator close, source release and other
snapshots. The shell also survives these operations and unrelated generation
output release. An exact dead shell's typed release is idempotent; its other
operations reject `invalid-state`. Active I2 borrow blocks release and read
without changing owner or cache. No graph, gradient or implicit finalizer
exists. Generation output accessors remain output-only and cannot read
manual logits. Neither the accessor nor typed release accepts an M3T shell,
generic I2 pointer or another aggregate's owner.

## Fixed package tuple and boxed signatures

Use a **new** tuple `g3g_manual_package_*`, built by
`scripts/build-g3g-manual.sh` into default `build/g3g-manual/`, with one
registry-owning `g3g_manual_package.o` member in
`libeshkol_transformer_g3g_manual.a`. Keep eight installed facades, copying
the generation facade for this artifact. New root
`native/g3g_manual_package_root.esk` source-composes the revision-2 lower
closure once, replaces its public adapter with a revision-3 adapter and
loads exactly `native/g3t_manual_p1_extension.esk`,
`native/g3m_prefill_p1_extension.esk`,
`native/g3m_prefill_p2_extension.esk`,
`native/g3m_manual_decode_extension.esk` and
`native/g3t_manual_logits_materialize_extension.esk` once each. Compile the
one G3-T transport with accepted manual logits/P1/P2/decode/materialization
private feature flags. Add no second G3-T/M3 registry owner or provider.
The planned closure has 40 Eshkol sources (revision 2's 35, replacing root
and adapter and adding five specified private sources), 53 native
source/header entries with its bridge substituted, five ordinary provider
objects, eight facades and one archive member. Freeze exact source/native/
object/defined/undefined/export/string/SHA manifests before acceptance.

The two A0 arity-two wrappers target boxed C signatures
`void et_e1b_public_g3_generator_prefill_v1(void *generator, void *input,
void *output)` and
`void et_e1b_public_g3_generator_decode_step_v1(void *generator, void *input,
void *output)`. The additive arity-one readout targets
`void et_e1b_public_g3_diagnostic_generation_logits_bits_v1(void *logits,
void *output)`. Existing
`et_e1b_public_g3_generation_tensor_release_v1(void *, void *)` keeps its
signature and gains only kind-3 admission. Three added names give targets
of **103 boxed exports, 109 global definitions and 109 public-name strings**,
versus revision 2's 100/106/106; measure the candidate manifests. The
source-private `et_g3t_private_logits_copy_bits_v1` is never a public boxed
export. Revisions 1, 2 and 3 remain separate installable artifacts and may
not be linked together; sequential default builds must preserve earlier
archives, objects and facades byte-identically.

## Admission, transaction and error boundary

Authenticate generator identity/liveness first, then exact same-aggregate
owned input identity/liveness, prompt provenance, native I1 rank/extent,
contiguity, CPU i64 byte values, and model/cache binding. Revision-2
exact-shell provenance chooses only the accepted P1 or P2 composer; native
admission remains authoritative. Scalar input is P1; authentic sealed
T1-backed two-byte input is P2. Manual calls admit budget 0 or 1 without a
draw or generate-only full-request preflight. Prefill atomically replaces
cache/binding without advancing RNG. Decode admits only a committed
length-one P1 prefix after manual P1 or P1/G0, appends position one, and
returns a new detached result; missing/full/stale/busy prefixes reject. A
subsequent manual prefill may replace the cache. Neither call creates a
generation output; `generator-generate!` still rejects an existing prefix.
P2 decode, repeated append and generation continuation remain deferred.

The private composers reserve and root the kind-3 shell and pending
registry entry before frame work. A revision-3 wrapper must complete every
additional Eshkol/boxed result allocation and fallible status preparation
**before** frame commit. No clone, materialization, conversion, allocation or
recoverable check may occur between successful native frame commit, call
finish and returning that same shell. On every precommit failure, abort the
acquired call, drain fourteen pins, destroy the pending result, and preserve
entry cache, binding, RNG and older detached logits/outputs. Preserve the
first source error across cleanup. An impossible postcommit failure is
fail-stop, not a recoverable E1 error with mutated cache and no result.

Each facade reports the **invoked public operation**, bounded data-only
source domain/category/code and `cause #f`, without leaking private names
or pointers. Forged, copied, foreign and wrong-kind shells map to
`invalid-argument`; exact dead, busy, pending, borrowed, stale-binding or
missing-prefix state maps to `invalid-state`. Malformed typed I1 shape and
token-range failures retain native `shape-mismatch`; wrong input metadata
length, non-byte IDs and inline/stored ID divergence retain native
`invalid-state`. Context/profile and admitted dtype/device/layout failures
retain their native categories. A live authenticated kind-3 record with
corrupt I2 shape, strides or storage is an internal invariant failure. The
diagnostic read allocates its bytevector before native copy; any recoverable
failure leaves destination bytes, source tensor, cache, binding and RNG
unchanged and no lease held. Typed release authenticates first, accepts an
exact dead identity before borrow checks, and destroys a live tensor only
after unborrowed preflight. It must not widen authority to a generic I2
pointer.

## Completion gates and deferrals

First resolve #143 private materialization hosted integration. Then
independently review this fixed-profile public
tensor contract and implement the revision-3 tuple. A fresh-cache installed
AOT caller must read all 256 words through the diagnostic accessor after
P1, T1-backed P2 and one-token decode; compare exact bits and K/V/mask with
independent M3T; and prove independent shell release and survival across
generator close, cache replacement and older output release. Cover both
budgets, input provenance, P1/G0/P1/G1 transitions and rejected
continuation. Negative gates include forged/foreign/wrong-kind/dead/pending/
borrowed owners, malformed storage, shape/device/layout and allocation/
copy/cleanup cuts, no orphan owner, old cache/result/RNG preservation on
failure and same-generator retry. Require E1 operation/details parity,
normal/repeat/ASan+UBSan+LSan, Q0, exact feature-on/off and private-symbol/
string inventories, old-package isolation, sequential byte-identical
predecessor artifacts and supported hosted CI. This adds no differentiable
operation; bitwise numerical parity, not a gradient check, is the relevant
gate.

Until that gate passes, revision 2's thirteen names remain installed. This
bounded contract claims neither full A0 generality nor generic tensor
interop. N>1, context above two, P2/repeated decode, generated-token
continuation, persistence and CLI remain deferred.
