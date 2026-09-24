# G3-C4 Step 18A — private prompt-owner prefill binding

Status: isolated implementation candidate. This is not a public generation API.
It joins the accepted Step 15A prompt owner with the accepted Step 16A/17A
numerical leaves without changing any predecessor ABI.

## Boundary and admission order

`ET_G3C4_PROMPT_PREFILL_PRIVATE` requires both
`ET_G3C4_PROMPT_T1_BORROW_PRIVATE` and `ET_G3C4_PREFILL2_PRIVATE`. It adds two
source-private calls:

```c
int64_t et_g3c4_private_prompt_prefill_preflight_v1(
    void *context, void *input, int64_t budget);
int64_t et_g3c4_private_prompt_prefill_v1(
    void *context, void *input, float last_logits[256]);
```

Preflight authenticates an exact idle generator and an exact live Step 15A
input owner. It then admits only `P` in `{1,2}`, `G=budget` in `{0,1}`, and
`P+G<=2`. This read-only check occurs before call acquisition, parameter
pinning, or any cache, binding, RNG, output, input, or registry mutation. The
caller must then acquire a generation call `(kind=2,budget=G)` and pass the
same input to the execution call. Execution repeats the length and `P+G`
checks, so bypassing or mismatching preflight cannot admit an oversized frame.

## Exact owned-input route

Execution begins one scoped I1 borrow from the authenticated Step 15A owner and
requires the exact dense CPU-i64 descriptor `[1,P]`, zero offset, aligned
`8*P`-byte data, and byte-token values in `[0,255]`. It retains no lease, view,
pointer, or copied token staging. `P=1` passes the sole borrowed value to the
accepted scalar P1 leaf. `P=2` passes the exact two-element borrowed span to the
accepted P2 leaf. There is no P3 route, fabricated padding token, fabricated
K/V row, or generic numerical fallback.

The matching numerical leaf remains the sole cache and binding publisher. Its
accepted schedule therefore produces exactly one K/V row for P1 or two K/V
rows for P2 and returns the genuine last-position logit row. The adapter adds
no numerical operation, provider dispatch, parameter snapshot, token rewrite,
or RNG access.

## Failure atomicity

Invalid identities, dead inputs, `P+G` overflow, wrong call state, active I1
borrows, I1 lease allocation failure, malformed owned descriptors, output
aliases, and every propagated numerical failure leave the old cache, parameter
binding, generator RNG, caller output, and active-call metadata unchanged. A
success or failure ends the scoped I1 borrow before returning. Cleanup restores
the first reported error; failure to end an already authenticated lease is an
unrecoverable invariant violation.

The focused witness proves both routes are bit-identical to direct invocation
of the accepted P1/P2 leaves, records all 21 authentic dispatches for each
route, compares committed cache and binding snapshots, checks exact canonical
prompt tuples and cache lengths, and covers all 42 P1/P2 dispatch cuts plus
pre-acquire, I1 allocation, active-borrow, and alias failures. Normal and repeated output must
be byte-identical to the ASan/UBSan/LSan run in the supported Ubuntu 22.04 /
LLVM 21.1.8 image.

## Limits

This leaf adds no Eshkol operation, public result or logits owner, output/text
owner, tokenizer decode, EOS behavior, generation loop, package export, or
installed symbol. Those remain pending dependency-ordered work.
