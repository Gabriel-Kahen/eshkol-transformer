# G3-C4 T1/I1 exact-pair validation prerequisite

**Proposed source-private contract; no implementation or public C4 admission is
approved by this document.** This is the missing validation dependency of the
[public C4 model/input leaf](G3_C4_PUBLIC_MODEL_INPUT_AUTHORITY_PROPOSAL.md).
It preserves the merged diagnostic-C2 path and the single G3-G owning archive.

## Observed boundary

The sealed [supported probe](/home/gabe/.codex/evidence/eshkol-transformer/g3c4-public-model-input-descriptor-blocker-8f50929-20260929-a/SHA256SUMS)
shows that one authentic, sealed, registered T1 shell still passes
`et_t1_i64_shell_length_v1`/`read_v1` after its borrowed view's rank, dtype,
device, layout or data pointer changes. The forged-data case reads from the
replacement buffer. `native/t1_i64_shell.c` checks the descriptor only at
creation; `src/eshkol_transformer/g3c4_model_owner.c` currently reads length
and words before allocating I1. Eshkol T1 membership proves shell identity,
not current descriptor or storage identity. Neither the existing I1 borrow-view
accessor nor length/read is an authoritative exact-pair validator.

## Closed private interface and owner

Add `et_i64_tensor_private_t1_pair_validate_v1(tensor, borrow, view,
expected_length)` in the **existing I1 owner** `native/i64_tensor.c`, declared
in a non-installed `native/i64_t1_pair_private.h`. Compile its definition and
declaration only under one linked-package feature macro,
`ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE`; the installed I1 ABI is unchanged.
`expected_length` is exactly 1 or 2 so existing private C4 P1/P2 reads can
share this guard; public `generation-c4-input-create` still admits only 2.
The helper returns a small private reason code: OK, invalid handle/lease,
shape, dtype, device, layout or storage. It allocates nothing, acquires no
borrow, changes no owner and never dereferences a candidate pointer before
pointer-value admission against **this owner instance's** `live_tensors` and
`live_borrows` registries. A retired/foreign tensor or borrow rejects even if
its bytes resemble a live object.

After admission, require the borrow's owner to be the admitted tensor, the
tensor's active borrow to be that exact borrow, and `view` to be the address of
that borrow's embedded view. Require live I1 rank 1, element count equal to
`expected_length`, byte length `8 * expected_length`, nonnull private shape,
stride and data allocations, shape `[expected_length]`, and stride `[8]`.
Require the view's exact ABI struct size, rank 1, same shape pointer and
extent, byte length, zero offset, dense row-major layout, and **data pointer
equal to this tensor's data allocation**. Require exact I1-owned CPU/i64
descriptor identity. Compare dtype/device pointer identity with canonical
I1-owned static strings installed when the borrow is built; do not `strcmp`
an untrusted, possibly invalid replacement pointer. Check pointer identity
before reading the shape extent or token storage. I1's private control fields
are the storage authority; callers cannot supply replacement allocations.
This contract does not claim detection of arbitrary writes into private I1
owner control memory outside the test-only seam.

The I1 helper alone does **not** bind a valid pair to its original T1 shell:
another live T1's tensor, borrow and view could satisfy every I1 check. At
successful T1 construction, after the borrow/view checks and before registry
publication, record a one-time **birth anchor in that same T1 shell**:
`birth_tensor`, `birth_borrow`, `birth_view`, `birth_data`, `birth_shape`, and
`birth_length`. These are independent of the operational
`tensor`/`borrow`/`view`/`length` fields; production never rewrites them and
the test corruption hook cannot target them. They require no second registry
or allocation. Before calling I1 validation, compare the operational triple
and length with the birth anchor by pointer/value only. After I1 validation,
require the validated view's data and shape pointers to equal the anchored
allocation pointers. Thus even a coherent replacement by another **live,
valid** T1 pair rejects without reading either token buffer. The sealed
shell's active borrow keeps its original I1 tensor and allocations live for
the shell's lifetime. An unsealed abort removes the shell from the T1 registry
before ending the borrow/destroying I1; no anchor is dereferenced afterward.
Unpublished construction failures clean up without consulting an incomplete
anchor. No pointer in an anchor is dereferenced before current registry/lease
validation; an altered pointer is only compared as a value.

Add `et_t1_i64_shell_private_c4_read_v1(candidate, words[2])` in the
**existing T1 shell owner** `native/t1_i64_shell.c`, declared under the same
feature macro only in its non-installed private header. It first admits the
T1 shell by registry pointer value, then requires sealed, birth-anchored
length 1 or 2, exact original pair identity, and the I1 exact-pair check.
Only after all checks may it read
the now-validated I1 storage into local words, reject IDs outside `[0,255]`,
and copy the accepted prefix into the caller's two-word native stack buffer.
Null output and unsupported length reject without a write. The output is a
trusted internal stack destination, never a user pointer or I1 allocation.
The returned private reason distinguishes invalid shell, unsealed/broken
lease, shape, dtype, device, layout, storage and byte-ID range. No T1/I1
borrow or pointer escapes with the words. Existing generic T1 length/read
behavior and public I1 ABI are unchanged.

`et_g3c4_private_input_from_t1_v1` must use that read as its **only**
source-word acquisition, before `et_g3c4_input_allocate` or I1 creation.
The Eshkol public wrapper keeps same-aggregate T1 authentication and its
P2-only check; native revalidation prevents private callers from bypassing
it. A second descriptor check followed by the old unchecked read is not this
contract. Rejection returns no C4 input or pending registry entry, leaves the
sealed T1 and all existing C2/C4/I1 owners, cache and RNG unchanged, and
allows a fresh attempt after fixture restoration. A successful copy owns
independent I1 `[1,2]` storage; it retains no T1 lease.

Private reason mapping at the C4 boundary must preserve the existing public
operation `generation-c4-input-create`, data-only details and `cause #f`.
Foreign shell maps to `invalid-argument`; unsealed, broken or substituted
lease maps to `invalid-state`; malformed shape/storage or out-of-range IDs to
`shape-mismatch`. Dtype/device/layout may use more precise categories only
if the existing E1/C4 bridge can carry them without a new public ABI; otherwise
map them to `shape-mismatch` and prove each rejects before copy. No fallback
or partially published shell is permitted.

## Test-only corruption and acceptance

Under `ET_T1_I64_SHELL_TESTING` and `ET_I64_TENSOR_TESTING`, expose a narrow
fixture-only mutation/restoration hook on an **authentic live sealed** shell's
borrowed view. Capture originals before mutation; change one field per case,
restore it exactly before cleanup or retry, and never release/free while a
field is altered. Cover rank, shape pointer/extent, struct size, byte length,
dtype pointer (including a non-dereferenceable value), device pointer,
layout, offset and data pointer (including a valid foreign two-word buffer).
The hook cannot mutate the birth anchor, create production authority or be
linked in feature-off packages. Test a replaced T1 tensor/borrow/view triple
with fixture-owned restoration, including the **whole valid pair** from a
second live sealed T1 of the same length and valid byte IDs. Prove the second
T1 still reads its own words while the first rejects, and both retry after
restoration. Also test invalid/retired operational pointers rejecting without
dereference. Do not use a forged shell pointer as a substitute for authentic
descriptor corruption.

Native and linked-package witnesses must show every malformed case rejected
before I1 allocation/publication, unchanged source bytes and live-owner counts,
then successful same-shell retry after restoration. Also cover genuine
unsealed/foreign/dead T1, length 0/3, IDs `-1`/`256`, two simultaneous T1
owners with crossed storage and whole-pair substitution, I1 allocation cuts,
and copied-storage independence. A sealed T1's normal borrow is required,
not a busy-state error.
A **separate held C4 input I1 borrow** must still make public release reject
recoverably and succeed after borrow end; it does not authorize skipping
source validation. Re-run unchanged C2 creation/release and one-archive,
feature-off symbol, source-closure and supported fe9 normal/repeat/sanitizer
gates. No public generator, output, EOS, loop, persistence or CLI claim follows
from this prerequisite.
