# G3-C4 Step 20A — private numeric output preparation

Status: isolated implementation candidate. This is not a public generation API
and does not publish a generation result.

## Boundary and dependencies

`ET_G3C4_OUTPUT_PREPARE_PRIVATE` requires the accepted Step 19A pending output
owner and Step 13A last-logit frame adapter. Step 13A already depends on the
accepted sampled-token frame and genuine token-forward route. This leaf adds
one source-private call:

```c
int64_t et_g3c4_private_output_prepare_v1(
    void *context, void *output);
```

The call authenticates an active generation context and the unique pending
output linked to it. It requires matching `P`, `G=budget`, `P+G<=2`, unchanged
false readiness flags, and the committed prompt parameter binding. It advances
`numeric_ready` exactly once.

## G0 and G1 preparation

For G0, the token frame must be idle. A scoped A2 read borrow proves that the
committed cache length equals reserved `P`. Preparation records length zero,
cache length `P`, and the current generator RNG. The exact I1 `[0]` remains
empty.

For G1, the accepted token frame must be `READY`, own its staged transaction,
and record absolute token position `P`. Preparation copies the staged byte-ID
candidate to the exact I1 `[1]`, records length one and cache length `P+1`, and
copies the staged successor RNG. Tests reach this state only through Step 13A
sampling followed by the accepted 21-dispatch token-forward route.

The I1 copy is the final recoverable operation. All admission, binding, frame,
and length checks precede it. On success the scalar lengths, RNG words, and
`numeric_ready` flag are written without another fallible call. The committed
cache, generator RNG, token-frame transaction, candidate, successor, and active
call remain unchanged.

## Failure atomicity and proof

Wrong-kind, foreign, cross-context, unlinked, repeated, pre-prefill,
pre-sample, sampled-but-not-forwarded, and stale-binding calls reject before
writes. All three G0 A2 read-borrow allocation cuts preserve the pending output,
committed cache, binding, RNG, and call. A live borrow on G1 output IDs makes
the I1 copy reject while preserving the complete ready token frame and all
output fields. Abort after either successful preparation scrubs the unpublished
output; G1 abort also discards the staged append while retaining the committed
prompt cache and old generator RNG.

Normal, repeated, and ASan/UBSan/LSan output must be identical in the supported
Ubuntu 22.04 / LLVM 21.1.8 image.

## Limits

This leaf adds no ID decode staging, text acceptance, frame preparation, cache
append or RNG publication, live-output transition, Eshkol output shell, rooted
text/raw-byte storage, accessor clone, EOS behavior, generation loop, public
operation, package export, or installed symbol. Those remain dependency-ordered
work.
