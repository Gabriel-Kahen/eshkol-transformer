# G3-C4 private P2/G1 generation copy-out proposal

**Contract for independent review; no implementation or public API is
accepted.** Base: reviewed shell PR #155 head `cf050385`/tree `a7167a4`.
This leaf reads one already-published `(P,G,N)=(2,1,1)` result after call
finish. It does not sample, decode another token, decide EOS, install a
generation accessor, or change CLI3.

## Existing owners and missing read boundary

`native/g3c4_p2g1_shell_publication_extension.esk` returns the existing
12-slot output shell. Its authenticated live entry keeps the native owner in
slot 3 and one privately owned raw-V256 bytevector `[1]` in slot 8. The
protected `g3c4-p2g1-output-records` entry pins those exact native/raw
identities after the call and staging roots are dropped. Slots 4–7 and 9–10
are false, and release calls native first, then tombstones the shell. The
published native `et_g3c4_output_internal` owns a separate I1 `[1]` ID,
`length=1`, `cache_length=3`, four successor RNG words, `parent_ctx=NULL`,
and three readiness flags. Its live validator checks shape, ownership and
unborrowed I1; native release destroys I1 and zeros the record. Both owners
survive call finish and generator close. The native owner has **no** raw byte.

`et_g3c4_private_output_copy_decode_ids_v1` belongs to the pending active
call: it requires the linked pending output and copies before `text_ready`.
It cannot read the detached published output. The Eshkol shell's raw byte is
mutable even though its identity is protected; text acceptance compared it
with I1 only before publication. A later read must compare them again. No
existing native function copies the published ID/length/cache/RNG tuple.

## Required I2 storage-inspection prerequisite

The detached output has `parent_ctx=NULL`, so the pending-copy helper's
`context->pins.views[0..13]` cannot protect model f32 storage. I1 and A2
already expose source-private, context-free storage-overlap queries;
**I2/f32 does not**. The C4 copy entrypoint below must remain unimplemented
until a separately reviewed, conditional I2 helper such as
`et_f32_tensor_private_storage_overlap_v1(pointer, bytes)` is accepted in
`native/f32_tensor.c` and declared under a private macro in
`include/eshkol_transformer/f32_tensor.h`. This is an inspection query, not
new tensor/model ownership, borrowing or a public ABI.

For a non-null, representable nonempty span it returns `1` on overlap with
any process-local live f32 tensor control, shape, strides or data; live
parameter/borrow/plan control and their owned arrays or prepared buffers;
and retained retired controls. It returns `0` only when disjoint from all
such allocations. Null, zero-length or overflowing spans return `-1`; the
C4 caller maps that to `invalid-argument/identity`, overlap to
`invalid-argument/alias`, and must never treat either as disjoint. The
helper has no output buffer, error-record write, allocation, model/context
argument, or mutation. It must use the authoritative I2 registries and
their existing retired-control index, not rely on a C4 owner or a borrowed
view; corruption that prevents a reliable scan is an internal fail-stop,
never a `0` result. The C4 call remains serialized as required by I2.

An active f32 borrow does not suppress inspection: its owner data and
borrow record remain protected and are reported as overlaps. A closed G3
generator or aborted construction owner supplies no pin requirement: live
f32 allocations belonging to any model remain protected; freed payload is
no longer owned, while retained retired controls still overlap. A sealed
model's parameter storage likewise remains protected after generator close.
C4 has no sealed model-owner close operation: generator close leaves that
owner sealed, so the contract does not infer model destruction from it. If
an I2 owner is actually destroyed, only its retained retired control and
any other still-live f32 allocation remain protected; freed payload is not
an owned span. The prerequisite gate must prove data, shape, stride,
live/retired control, parameter and plan overlaps, external active borrow,
two independent models,
closed-generator and aborted-owner cases, a disjoint carrier, and null/end
overflow. For every rejection, carrier bytes and I2 ownership remain
unchanged. Compare the query against I2's existing
`storage_aliases_live_reference` behavior without weakening feature-off I2
symbols or public behavior.

## One conditional native copy boundary

Add one **new, reviewed C4-private** entrypoint, provisionally
`et_g3c4_private_output_copy_snapshot_v1(output, raw_header, dest_header)`.
Gate its definition under `ET_G3C4_P2_G1_COPYOUT_PRIVATE`, requiring the
accepted publication feature; feature-off object symbols and behavior stay
unchanged. The entrypoint takes the exact registered published output, the
current one-byte Eshkol raw carrier, and a caller-owned bytevector carrier
with declared payload length **56**. It requires no generator/context and
changes no native record size or installed ABI. Its seven signed little-endian
`i64` payload words are, in order: selected ID, generated length, cache
length, then RNG words 0–3. There is no pointer, tensor handle, raw storage
alias, or executable object in the result. As with the existing C4 carrier
ABI, native pointer-range checks assume accessible caller-owned storage;
the Eshkol seam supplies actual bytevectors, and no arbitrary-address
mapping probe is claimed.

Only after that I2 prerequisite is reviewed and gated, before writing,
authenticate the native output through the existing registered-output path
and require `P=2`, `G=1`, published, detached,
`text_ready=ids_copied=numeric_ready=1`, I1 live/unborrowed with exact dense
`i64[1]` shape/stride/bytes, length one, cache length three, and valid
four-word RNG. Validate both bytevector headers/ranges, raw length one,
destination length 56, and their nonoverlap with each other, registered
transport/model owner records, and all I1, A2 and I2-owned storage using
the three context-free private overlap queries. Unlike the pending-copy
helper, this check cannot depend on an active context or its pins. Begin a
scoped I1 borrow, read its exact view, require ID `[0,255]` and `raw[0]=ID`,
then end that borrow on every path. Stage all seven words locally and copy
them to the destination only after every fallible check and
borrow end succeeds. A rejected request leaves the destination header and
all 56 payload bytes, output, raw owner, cache and RNG unchanged; cleanup
preserves the first error. A borrowed I1 rejects before writing and succeeds
after its owner ends the borrow. A malformed native owner/descriptor is an
internal invariant error; malformed caller carriers and stale/forged identities use
the established typed native error mapping. No live native pointer or I1
lease escapes.

## Source-private detached value and lifetime

A new extension loaded after the shell extension may expose exactly one
source-private copy operation. Under an `m3-call`, it authenticates the live
shell **and** the protected native/raw identity record before allocating a
fresh raw bytevector `[1]`, exact native staging `[56]`, and result/RNG
containers. The result carrier is a fresh five-slot vector and no slot
aliases the live shell or native I1. It passes the existing private raw
carrier and new staging to the native copy boundary. It then decodes the
seven signed little-endian words with the existing C2-style
unsigned-load/sign-conversion pattern, checks ID `[0,255]`, lengths `1/3`,
RNG schema
`g3.philox4x32-10.categorical-f32.v1`, and rechecks shell/record identities
and `raw[0]=ID` before filling the preallocated result. Its private result
is exactly `(raw-copy, selected-id, length, cache-length, rng-snapshot)`,
where `raw-copy` is a new `[1]` bytevector and `rng-snapshot` is a new
five-slot G3-C4 snapshot with its algorithm tag and four copied words.
The result is caller-owned, detached and mutable only as a copy; changing it
cannot change the shell, native I1, or a later copy. It is **not** a public
generation output or a public tensor/list/text accessor.

Allocation or decoding failure publishes no result and leaves the live
output unchanged. The operation discards/scrubs its private staging and
partial copy on failure, propagates a typed error, and permits same-output
retry after a recoverable borrow or raw-content repair. It never calls
native output release or alters the generator/cache/RNG. An altered slot-3
native owner, slot-8 raw owner, protected-record link, raw byte, or I1 ID
rejects before returning a value. An authenticated dead output rejects
copy-out as `invalid-state` even though repeated **release** remains
idempotent; forged shells reject as `invalid-argument`. Copies returned
before release remain valid after native-first release and generator close.
Neither the shell's raw bytevector nor its native I1 pointer is returned.
The operation is synchronous under the existing aggregate guard; this leaf
adds no concurrent release/read guarantee.

## Acceptance and next gate

The focused native and Eshkol witness must use genuine seeded P2 prefill,
position-two G3-S sampling and token-forward, followed by the reviewed
joint frame/shell publication for both greedy and categorical policies.
It compares copied ID/raw byte, `1/3` lengths and all RNG words against the
owned native output and independent Philox oracle, including reads after
generator close. Two reads must return equal but separately owned carriers;
mutating the first raw/RNG copy must not affect the output or second read.
Release must invalidate later reads while preserving earlier copies.

Negative cuts cover pending/dead/wrong-kind/forged/cross-call shells,
two-output native/raw owner swaps, mutated raw content and native ID
`-1/256`, malformed I1 shape/stride/storage, malformed/short/overlapping
destination and raw carriers, including aliases into borrowed and closed-
generator I2 storage, an external I1 borrow, and each reachable
allocation/borrow/view failure. For every rejection, prove no partial
destination write or returned result, no leaked lease, unchanged output,
cache and RNG, then repair/end the lease and retry or release. Keep the old
pending-ID-copy route, shell release, feature-off symbols, diagnostic C2,
and public package behavior unchanged. First gate the I2 inspection helper
with its own focused source/negative/feature-off tests. Then run focused
source/symbol closure and normal/repeat/sanitizer linked native-plus-Eshkol
gates on pinned fe9; seal exact head/tree evidence before integration.

Only after this private read boundary is reviewed may a separate package
adapter define fresh public IDs/text/lengths/RNG ownership. EOS comparison
and stop policy, repeated `G>1` decode, `N>1`, cache/no-cache parity and
CLI3 `generate` each require their own upstream contract and gate.
