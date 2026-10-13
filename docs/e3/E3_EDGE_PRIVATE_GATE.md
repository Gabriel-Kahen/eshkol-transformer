# E3 private empty, singleton, EOS and one-token-tail gate

This is a **source-staged, pending compiled-private** §9 N5 gate. It extends
the existing noninstalled retention aggregate only; no public API, production
kernel or expected CE/logit carrier changes. `tests/e3_horizon/edge_generate.py`
creates deterministic checksummed D1 resources and independent packed D2
reference rows at the fixed N1/T2/V256/D4 profile:

| Case | D1 tokens | D2 rows at entry | Expected evaluator path |
|---|---:|---:|---|
| empty | 0 | 0 | Stage-12 `invalid-state`, six outputs unchanged |
| singleton | 1 | 0 | Stage-12 `invalid-state`, six outputs unchanged |
| entry-eos | 3 | 1, consumed before call | Stage-12 `invalid-state`, six outputs unchanged |
| one-token-tail | 4 | `11`, then `10` | Success, three active tokens, two batches, weight bits `1077936128` |

The final valid row is `input (0,0)`, `target (255,0)`, mask `(true,false)`.
The gate derives these rows and exact entry cursor bytes from the D2 reference
before runtime. It makes no claim about the numerical loss beyond positive
finite binary32 output from the real seed-1729 M3 model and L2/metrics path.

`tests/e3_private/edge_runtime.esk` invokes the genuine private
`e3-evaluate-into!` with the M3 model, raw T1 tokenizer, D2 dataset, E3 frame
and four I2 destinations. For zero-row cases, the native finalizer rejects
because both counts are zero; the driver asserts the first mapped E1 stage-12
error and six-output atomicity. Every case compares the exact D2 cursor,
17-bit mode image, all parameter and gradient words/metadata, initializer RNG,
and live/retired native ownership controls to its pre-call image. The successful
tail asserts count and weight from authentic masks, without invented logits.

`scripts/test-e3-edge-private.sh` is the proposed supported compiled gate. It
regenerates all four resources twice and compares bytes, builds canonical and
fresh A/B plus sanitized noninstalled packages, checks the existing closed
symbol/source inventories, compiles two optimized AOT binaries and a sanitizer
binary, and runs normal, poisoned and ASan/UBSan/LSan profiles. The observed
cursor lines must equal the independently generated oracle. It retains all
build and runtime evidence in a fresh directory. No compiled execution has
been run from this source-staging branch.

Independent source review, then the supported compiled gate after PR #206 CI
settles, are required before marking these four N5 cases proved. Exact
8,192/8,193 and full resource/ownership horizons remain separately gated.
