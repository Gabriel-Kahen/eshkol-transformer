# G3-T source-private manual logits bit materialization proposal

**Proposed bounded private contract; no implementation, public tensor or G3-G
export exists.** This follows the accepted [kind-3 logits owner](G3_T_MANUAL_LOGITS_LEAF.md)
and P1/P2/decode publication semantics. It supplies an authenticated exact-bit
snapshot to a private consumer. It does not satisfy A0's newly owned floating
`[N,V]` result by itself; that needs a separate public tensor and atomic
cache/result publication contract.

## Exact source-private boundary

Add one feature-gated native operation under
`ET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE` (which requires
`ET_G3T_MANUAL_LOGITS_PRIVATE`):

```c
int64_t et_g3t_private_logits_copy_bits_v1(
    void *logits, void *bytevector_header, int64_t byte_count);
```

The single result representation is a **caller-owned 1,024-byte Eshkol
bytevector**, holding 256 binary32 words in little-endian physical bit order.
`byte_count` must be exactly 1024; `bytevector_header` must be a non-null
Eshkol bytevector header with encoded length 1024 and a representable complete
payload span. This follows the reviewed M3/M3T native `byte_payload` and I2
`et_f32_tensor_copy_bits_to_v1` patterns, adjusted from 512 to 256 words.
The accepted diagnostic profile is x86-64 little-endian. The operation is
read-only: it neither creates nor releases a G3-T owner, changes a cache/RNG,
nor performs floating arithmetic or conversion. The caller owns the returned
bytes independently of the source logits and generator.

A source-private Eshkol wrapper `(g3t-logits-bits logits)` has arity one. It
roots a fresh 1,024-byte bytevector before the native call, authenticates the
G3-T `logits` shell and its live entry, passes only its exact native pointer,
and returns the bytevector on success. On failure it does not publish the
bytevector and maps the first native source domain/category/code through the
existing E1 private boundary. This wrapper is not installed in
`lib/transformer/generation.esk`; its exact local symbol belongs in the
private source closure only. The caller still owns and separately releases
the original kind-3 logits shell with the accepted typed operation.

## Admission, shape and errors

Clear G3-T status, then locate `logits` by exact native registry address
before dereference. Require kind `G3T_LOGITS`, state `G3T_LIVE`, `busy=0`,
`parent_ctx=NULL`, and non-null owned I2 storage. A pending result (including
a frame-idle pending result), dead result, generator/input/output/clone owner,
foreign/forged pointer, or mismatched parent/link state cannot be read. Exact
identity/wrong kind maps to G3-T argument/identity; pending, dead or busy maps
to state/lifecycle; a corrupt live record maps to internal/invariant. No
caller shape tag or cast supplies authority.

Before copying, prove I2 rank 2, shape `[1,256]`, dense row-major strides
`[1024,4]`, 256 elements/1,024 bytes, CPU f32 storage and no active borrow.
Use a tracked I2 borrow begin to reject an existing lease with I2
`invalid-state/active-borrow`; inspect the canonical K1 view, then copy with
`et_f32_tensor_copy_bits_to_v1(..., 256, ...)` and end the lease. The native
bridge preserves I2's domain `f32-tensor` (G3-T domain 3), category and code
on a recoverable I2 failure, snapshotting that status before lease cleanup.
A malformed bytevector header/count is
shape-mismatch before destination writes; a null/invalid destination is
invalid-argument under the accepted native pointer threat model. Destination
alias into any live I2 allocation is rejected by I2's output preflight.

All fallible checks and lease allocation precede the final copy. A
recoverable failure leaves the destination bytes, source logits, cache,
binding and RNG unchanged and ends any lease acquired by this call. The
accepted I2 copy validates the whole destination and source before its one
`memcpy`; a lease-end failure after a completed copy is an impossible
invariant breach and fail-stop. No partial result or new owner escapes.
A successful snapshot can be repeated; bytes from an earlier snapshot stay
unchanged after later manual prefill, decode, generator close or source release.
It does not turn the original logits shell into an M3T owner or public tensor.

## Package boundary and focused gate

The feature-off native object must define no new stem. Feature-on tests link
this private G3-T translation unit once with I2 and existing providers; exact
source, header, native/object and defined/undefined symbol inventories must
admit only the new private stem and wrapper. The existing revision-2 G3-G
archive and installed facade are unchanged; its public AOT caller must not
resolve this symbol. Any future package must pin a new closed tuple and
exclude the private stem from public exports and installed source strings.

Test authentic P1, T1-backed P2 and one-token decode results. Compare all
256 output words bitwise with the accepted `ET_G3T_TESTING` witness and
independent M3T reference; neither path may normalize binary32 bit patterns.
Verify repeated detached snapshots, source/result release
order, generator close, cache replacement, old-result retention and unchanged
RNG/cache. Negative tests cover forged/foreign/wrong-kind, pending (idle and
active frame), dead, busy, borrowed, null/bad-length/overflowing header,
wrong count, aliased destination, corrupted I2 shape/stride and allocator
cuts before copy. Each cut must preserve destination sentinel bytes and live
owner counts; clearing the test borrow must permit retry. Run normal/repeat/
ASan+UBSan+LSan, Q0, private source closure and production test-symbol
exclusion. No numerical gradient applies: the seam copies bit patterns and
has no differentiable operation.
