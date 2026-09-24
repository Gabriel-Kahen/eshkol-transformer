# G3-C4 Step 19A — private native output reservation

Status: isolated implementation candidate. This is not a public generation API
or a complete generation result.

## Boundary

`ET_G3C4_OUTPUT_RESERVATION_PRIVATE` requires the accepted Step 18A
`ET_G3C4_PROMPT_PREFILL_PRIVATE` leaf. It adds two source-private calls:

```c
void *et_g3c4_private_output_reserve_v1(
    void *context, int64_t prompt_length);
int64_t et_g3c4_private_output_release_v1(void *output);
```

Reservation authenticates an active generation call, repeats `P` in `{1,2}`,
`G=budget` in `{0,1}`, and `P+G<=2`, and rejects a second pending output for
the same context. It allocates the complete native owner and its exact I1 `[G]`
ID capacity before enrolling either object. Its inline RNG words and numeric,
ID-copy, and text readiness flags begin at zero. No token, length, cache length,
RNG result, or text is ready at this boundary.

The output payload fields are exactly `parent_ctx`, `P`, `G`, `ids`, `length`,
`cache_length`, `rng[4]`, `numeric_ready`, `ids_copied`, and `text_ready` after
the common transport header. Only a complete pending owner enters the closed
native registry. Failed construction destroys unpublished storage while
preserving the first error.

## Prompt and lifetime linkage

With this leaf enabled, Step 18A prefill requires exactly one pending output
linked to the same active context with matching `P` and `G`. It rejects a
caller logits span overlapping the output owner or its I1 payload before
borrowing prompt IDs or entering a numerical route. This makes native result
capacity a prerequisite for the first numerical write. The accepted
P1/P2 numerical routes remain the sole cache and binding publishers; this leaf
adds no provider dispatch, sampling, RNG mutation, or cache mutation.

A pending output prevents call prepare-end and finish. Before destroying the
pending output, call abort proves its I1 has no active borrow before discarding a token frame,
then proves the committed cache has no active view and destroys and scrubs the pending
output before draining the active call. If its I1 storage has an active borrow,
destruction fails before the output or call is mutated. After successful
prefill, abort retains the committed prompt cache, parameter binding, and old
generator RNG while destroying the still-unprepared result. Exact dead release
is idempotent, and wrong-kind or foreign identities reject before dereference.

## Proof

The focused witness covers P2/G0 and P1/G1 reservation before authentic
21-dispatch prefill, matching linkage, duplicate and shape rejection, one
native owner allocation cut, all three G0 and four G1 I1 allocation cuts, and
cache-view and output-borrow destruction cuts. It proves pending flags remain false through
prefill, covers output-owner and I1-payload logits aliases before dispatch, and
proves a non-idle token frame is unchanged when output destruction preflight
fails. Successful abort scrubs every output field while preserving committed
prompt state. Normal, repeated, and ASan/UBSan/LSan output must be identical in
the supported Ubuntu 22.04 / LLVM 21.1.8 image.

## Limits

This leaf adds no Eshkol output shell or rooted text/raw-byte storage,
no numeric output preparation, ID decode copy, text acceptance, result publication,
accessor clone, EOS behavior, generation loop, public operation, no package export,
or installed symbol. Those boundaries remain dependency-ordered work.
