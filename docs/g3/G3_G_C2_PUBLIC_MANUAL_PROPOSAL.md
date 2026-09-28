# G3-G diagnostic-C2 public manual prefill/decode proposal

**Proposed contract only.** This is the next bounded successor to the
[G3-G revision-2 G0/G1 package](G3_G_C2_PUBLIC_G0_PROPOSAL.md), based on the
source-private [P1 prefill](G3_M_PREFILL_P1_LEAF.md),
[P2 prefill](G3_M_PREFILL_P2_LEAF.md), and
[one-token decode](G3_M_MANUAL_DECODE_COMPOSER_PROPOSAL.md) composers. The
decode implementation is still an isolated candidate
([leaf](G3_M_MANUAL_DECODE_LEAF.md)); package implementation waits for its
independent source review and integration. This document adds no code or
accepted public capability.

## Exact public and package boundary

Add only the two A0 names below to `lib/transformer/generation.esk`. Keep the
thirteen revision-2 names, arities, G0/G1 routing, and diagnostic-C2
configuration contract. No new input constructor or native G3-T API is
assumed.

| Name | Arity | Result and private route |
|---|---:|---|
| `generator-prefill!` | 2 | Authentic generator and existing G3-owned byte input of length P=1 or 2; delegate once to `g3m-prefill-p1!` or `g3m-prefill-p2!`; return its detached kind-3 owned CPU f32 `[1,256]` last-row logits shell. |
| `generator-decode-step!` | 2 | Authentic generator with a committed P1 prefix and existing G3-owned P1 byte input; delegate once to `g3m-decode-one!`; return its detached kind-3 owned CPU f32 `[1,256]` last-row logits shell after the `1→2` append. |

Widen the existing `generation-tensor-release!` to admit an authentic kind-3
manual logits shell through the accepted private typed tensor release. It is
idempotent for the exact dead shell. The public result is an **opaque G3-owned
logits handle**, not an M3T logits owner or a general public tensor: there is
no accepted production G3-T logits read/clone/interop operation. Numerical
inspection from an installed facade, and the full A0 floating tensor
interoperability promise, require a separate upstream contract. Tests may
inspect bits through the existing private testing closure; the public facade
must not expose its hooks or claim user-readable values in this revision.
`generation-output-ids`, `-lengths`, `-text`, `-rng`, `-cache-lengths` and
`generation-output-release!` accept generation outputs only and reject a
manual logits shell as the wrong kind. A manual call creates no generation
output.

Use a new fixed **G3-G C2 revision-3** E1B tuple, provisionally named
`g3g_manual_package_*`, `libeshkol_transformer_g3g_manual.a`, one owning
`g3g_manual_package.o`, `scripts/build-g3g-manual.sh`, and a distinct default
`$(project_build_dir)/g3g-manual`. Install the same eight facades with only
the generation facade extended. The two new boxed exports are
`et_e1b_public_g3_generator_prefill_v1` and
`et_e1b_public_g3_generator_decode_step_v1`, paired with private boxed
`et_e1b_private_g3_generator_prefill_cabi_v1` and
`et_e1b_private_g3_generator_decode_step_cabi_v1`; the existing thirteen
signatures stay unchanged. Relative to revision 2, 102 boxed exports and
108 globals/public-name strings are **targets to measure**, not evidence.
Source-compose the revision-2 closure plus exactly
`native/g3t_manual_p1_extension.esk`,
`native/g3m_prefill_p1_extension.esk`,
`native/g3m_prefill_p2_extension.esk` and
`native/g3m_manual_decode_extension.esk`, with one G3-T/M3 registry owner and
the existing N2/N3K/A2/G3N providers. Compile the G3-T transport with its
accepted manual logits, P1, P2 and decode private feature flags. Measure and
pin the exact source/native/object/export/undefined/string manifests. Never
link the v1/v2/v3 archives together; a shell from another aggregate is
foreign. Sequential default v1→v2→v3 builds must preserve older archives,
objects and installed facades byte for byte.

## Admission, ownership and transaction

The facade authenticates the receiver and input with the existing G3-T
registry before dispatch. For prefill it uses the revision-2 constructor's
exact-shell prompt-provenance record only to choose P1 or P2; the private
native frame still proves owned I1 rank/shape, byte values, liveness,
unborrowed storage and same-aggregate identity. A scalar token constructor
produces P1; only an authentic T1-backed two-byte input produces P2.
Do not infer length from an Eshkol tag, copied T1 value, test observer or
failed probe. Both operations accept a generator configured with budget 0
or 1; manual scheduling does not run the generate-only draw or full-request
preflight. Categorical exhaustion therefore does not block a manual call.

Manual prefill atomically replaces an existing P1/P2 cache and binding,
including after a completed G0/G1 call, without consuming RNG. The old
detached logits and generation outputs remain independent. Decode requires
exactly one committed cache position and current model binding, including
after manual P1 prefill or P1/G0. It appends at position one into the same
A2 cache; no P2/full-cache, missing-prefix, stale-binding or second append
route is admitted. A new manual prefill can subsequently replace that cache.
Neither operation emits IDs/text or enables `generator-generate!` to continue
from a prefix; the existing generate preflight still rejects a committed
prefix.

Each call uses the accepted single outer `m3-call`, fourteen model pins,
pending kind-3 logits owner, exactly 21 ordered G3-N/N2/A2 roles, native
prepare/preflight/commit and adjacent no-failure finish. The public wrapper
returns the already-live private result without postcommit allocation or
conversion. The caller retains input ownership. A detached result survives
generator close, cache replacement and later calls until typed release;
the generator retains model/tokenizer references and owns cache, binding and
RNG. No random draw, gradient change or scalar numerical fallback occurs.

Every precommit failure aborts the acquired call, kills its pending logits,
drains pins and preserves entry cache/binding/RNG and older detached results;
the original error is re-raised. No acquired call means no abort. Decode
failure retains its length-one prefix. Successful commit publishes cache and
one detached result together. An impossible postcommit or cleanup failure
is fail-stop. The public E1 boundary reports the **invoked public operation**
with bounded data-only source domain/category/code and `cause #f`; it
preserves native I1/F32/A2/provider error details before cleanup. Wrong-kind,
forged and foreign shells are `invalid-argument`; exact dead, busy, borrowed,
missing/full prefix and stale binding are `invalid-state` according to the
accepted private precedence; malformed dimensions are `shape-mismatch`.
No wrapper success or recoverable error may expose a pending result.

## Acceptance gates and deferrals

First review/integrate the exact private manual P2/decode sources and tests.
Then gate the candidate revision-3 tuple with fresh-cache public AOT callers,
exact manifests, one owning archive member, eight facades, feature-off and
private-symbol negatives, mixed-tuple rejection, Q0 and supported CI. Prove
both P1 constructors, T1-backed P2, both configured budgets, prefill after
P1/P2/G0/G1, decode after P1 manual and P1/G0, cache replacement, output
lifetime, typed release and byte-identical older package artifacts. Check
forged/copy/foreign/dead/wrong-kind/P2/inline-only/borrowed/malformed inputs,
missing/full/stale/busy contexts, categorical exhaustion, failed release,
allocation/provider/A2/old-cache-borrow cuts, original E1 details, rollback
and retry. Public wrapper preallocation/dispatch cuts must show no orphan
owner and no postcommit allocation.

The private numerical gate must compare every one of 256 P1, P2 and decode
last-logit words and committed K/V positions **bitwise** against independent
M3T, including the P2 causal mask and decode `[1,1]` mask; check cache lengths,
unchanged RNG, prior-result survival and normal/repeat/ASan+UBSan+LSan runs.
Preserve the accepted P1/G1 and P1/P2 G0 regression gates. The installed
facade currently proves ownership/routing/errors, not readable logit values.

Deferred: public logits read/clone or general tensor interoperability,
capacity above two, P2 decode, repeated append, P2/G1, a generation
continuation loop, N>1, persistence and CLI. None follows from this package.
