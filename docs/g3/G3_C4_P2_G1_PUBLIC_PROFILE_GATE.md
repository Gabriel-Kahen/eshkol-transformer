# G3-C4 P2/G1 public-profile and output-accessor gate

**Dependency contract for independent review; no public C4 implementation.**
The reviewed private `(P,G,N)=(2,1,1)` output is copyable, but it is not an
output of the installed public diagnostic-C2 generator. This note fixes the
boundary before any package adapter is implemented.

## Present public and private authority

The accepted [G3 generation proposal](../G3_GENERATION_PROPOSAL.md) admits
public diagnostic-C2 requests only when `P+G<=2`. Its model has two learned
positions, and the public cache length cannot exceed two. The current
`lib/transformer/generation.esk` already exports the five output accessors,
the three typed releases, and the constructor/generate names. The merged
public G3-G P1/G1, G0 and manual packages (#140, #142, #146) route through
the G3-T diagnostic-C2 owner in
`native/g3g_manual_package_root.esk` and
`native/g3g_manual_public_extension.esk`. Their existing public tests cover
fresh IDs, text, lengths, cache lengths and RNG owners, release/lifetime,
and the C2 budget boundary. The older
[C2 public proposal](G3_G_C2_PUBLIC_CONTRACT_PROPOSAL.md) still says
"proposed" in its header; that status text does not remove the merged
capability or authorize C4 admission.

The separate C4 model uses `diagnostic-c4`. Native/shell P2/G1 publication
from #155 yields an authentic C4 output with generated length one and cache
length **three**. The I2/f32 context-free overlap prerequisite from #157
and the reviewed [private copy-out](G3_C4_P2_G1_COPYOUT_PROPOSAL.md) at
`6100465`/tree `344dd58` permit detached reads of raw byte `[1]`, selected
ID, lengths `1/3`, and four successor RNG words after native-first
publication and even after generator close. This private five-vector is
mutable caller-owned data, not an opaque public output, tensor or RNG
owner. Its exact native/shell identity belongs to the C4 registry, not to
G3-T. `et_g3t_private_output_*_clone_v1` admits G3-T records and, for
cache lengths, enforces C2's maximum two; no C4 output may be relabeled to
pass that check.

## Required decision before a public adapter

No public accessor dispatch for a C4 output may be added merely because
the existing names accept an opaque argument. A separate accepted public
C4 profile must first define the model/context and learned-position
authority for capacity at least three, how `diagnostic-c4` is represented
and validated in public configuration, the exact public constructor and
input provenance, and a public `generator-generate!` route that returns
this authentic C4 output. That contract must specify how C4 coexists with
the merged C2 package and preserves the latter's `P+G<=2` rejection and
error order. It must also settle whether C4 P2/G1 output access is included
in that same public-profile leaf or follows as a separately reviewed adapter.
Until then, a trusted private fixture may read/release the C4 output through
the accepted private operations; a public caller cannot produce it.

Once that profile is accepted, an accessor adapter may reuse the **existing
public names and arities** with these fixed C4 results: a fresh list with
one owned CPU I1 `i64[1]` ID, fresh owned CPU I1 `i64[1]` tensors holding
length `1` and cache length `3`, a fresh list with one detached raw
bytevector `[1]`, and a fresh authenticated immutable/releasable G3 RNG
owner with the four copied words. Exact-owner output release must be
idempotent, native-first and independent of earlier clones; clones survive
output release and generator close. Every accessor must authenticate exact
C4 shell/record identity, call the reviewed copy-out before constructing a
public result, and publish no partial clone after allocation/decoding
failure. The G3-T native clone stems cannot implement these C4 clones; the
adapter must contract its own typed owner construction from the existing I1
create/copy/destroy ABI and protected RNG storage. These are **requirements
for a later contract**, not assumed upstream functions or an authorization
to implement them now.

The later contract must decide exact C4/G3-T registry dispatch in one
source-composed owning archive, clone/release ownership and rollback, public
E1 error precedence, and A0's `device-mismatch` for inconsistent retained
storage. Current C4 copy-out maps several malformed native I1 invariants to
`internal`; an adapter must not guess `device-mismatch` without a reviewed
source discriminator or proof that the public state cannot reach it. Public
accessors must continue rejecting wrong-kind/forged owners as
`invalid-argument` and authentic dead owners as `invalid-state`, with
invoked-operation, bounded data-only details and `cause #f`.

## Evidence gate after profile approval

First integrate and recheck #155, #157 and the private copy-out in the
intended package closure; no second registry-owning archive may be linked
beside the merged C2 package. A later public-profile/accessor candidate
needs a genuine public C4 generator/output producer, then fresh-value and
release tests for all five accessors across greedy and categorical P2/G1.
Prove ID/raw parity, lengths `1/3`, all RNG words against the independent
G3-S/Philox oracle, two-call isolation, output-close/release survival,
wrong-kind/forged/dead/two-output swaps, borrowed I1, malformed storage,
allocation/decoder rollback and same-output retry. Re-run existing C2
P1/G1, G0 and manual public tests unchanged. Verify one package owner,
source/export/undefined/string manifests, private-link negatives, fresh
AOT public callers, feature-off behavior, and supported pinned fe9 normal,
repeat and sanitizer evidence before claiming public access.

This gate adds no EOS policy, repeated `G>1` decode, `N>1`, cache/no-cache
parity, persistence or CLI3 `generate`; each still needs its own contract
and end-to-end evidence.
