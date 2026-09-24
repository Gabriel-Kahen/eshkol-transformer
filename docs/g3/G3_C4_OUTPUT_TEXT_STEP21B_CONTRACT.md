# G3-C4 Step 21B — private raw decode and text readiness

Status: isolated implementation candidate. This is not final-frame preparation,
result publication, an output accessor, or a public generation API.

## Boundary and dependency

This leaf depends exactly on Step21A private output ID staging. It adds one
successor-only Eshkol source extension and one source-private native call:

```scheme
(t1-private-g3-decode-raw-into! tokenizer staging raw)
```

```c
int64_t et_g3c4_private_output_accept_text_v1(
    void *context, void *output, void *raw_header);
```

The Eshkol decoder is loaded only after the canonical T1 aggregate in trusted
source composition. It authenticates the tokenizer through the actual T1
registry entry and core tag, then requires raw policy, empty special/prefix/
suffix collections, and vocabulary size 256. It accepts only an exact 0- or
8-byte little-endian ID staging bytevector and an exact 0- or 1-byte raw output.
Every ID is checked before the first raw write. The valid path allocates no
bytevector, vector, cons, hash, decoded scratch, or tokenizer shell.

The native call authenticates the active context and its unique pending output,
requires `numeric_ready=1`, `ids_copied=1`, and `text_ready=0`, and revalidates
the committed parameter binding. Its complete Eshkol bytevector carrier is
checked against every G3 transport, the model owner, all parameter pins, every
live I1 storage allocation and A2 cache/transaction allocation. It then borrows
the exact output I1 synchronously, verifies G0/G1 shape and raw byte equality,
ends the borrow, and advances `text_ready=1` as the no-failure tail. Repeated,
out-of-order, mismatched, aliased, and cross-owner calls reject before readiness
mutation.

## Proof

Focused Eshkol evidence covers G0/G1 decode, actual registry/core rejection,
shape/type/range failures, mutation atomicity, no registry growth, unchanged
allocating semantic decode behavior, and allocator interception proving no
vector or cons allocation on the valid in-place path. Focused native evidence
covers G0/G1 acceptance, readiness ordering, raw mismatch, malformed carriers,
complete owned-storage aliases, stale binding, I1 borrow allocation and active
borrow cuts, cross-owner linkage, repeat rejection, and unchanged state on every
failure.

Normal, repeated, and ASan/UBSan/LSan results must be byte-identical in the
supported Ubuntu 22.04 / LLVM 21.1.8 image. Object inspection must add exactly
`et_g3c4_private_output_accept_text_v1` over the Step21A owner object and no new
undefined symbol. The predecessor Step21A gate and Python isolation test remain
required.

## Deliberate remaining dependency

The accepted two-argument `g3t-t1-decode-output!` coordinator is not introduced
here because the repository has no rooted Eshkol output entry yet: Steps 19A,
20A, and 21A expose only the pending native output. Inventing an output-entry
layout in this leaf would create a competing authority. A later output-envelope
leaf must retain the preallocated staging/raw bytevectors and exact native
output pointer; it can then validate the call/output cross-links and sequence
ID copy, this T1 decoder, and text acceptance. This leaf adds no output envelope,
frame prepare/commit, cache or RNG publication, installed symbol, or public generator.
