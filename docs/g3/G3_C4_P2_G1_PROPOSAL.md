# G3-C4 private P2/G1 pending-transcript proposal

**Proposed contract; no implementation or public API is accepted.** This is a
single-row `diagnostic-c4` numerical and pending-output transcript using the
accepted C4 P2 prefill and one-token frame. It does not change the installed M3T/diagnostic-C2 or G3-G
packages. M3T is fixed to context two, including a `[2,4]` position table and
`[1,2,*]` forward tensors; its owner cannot be relabeled as C4.

## Available numerical path and owner

The C4 owner has fourteen parameters with a real `[4,4]` position table.
`et_g3c4_private_prefill2_v1` computes two authentic prompt rows and commits
two K/V positions into capacity-four A2 storage. The accepted
`et_g3c4_private_token_frame_begin_last_v1` samples its `[1,256]` last-logit
row with G3-S, obtains the committed length two, and opens a width-one A2
transaction. `et_g3c4_private_token_forward_v1` uses query position two and
the existing G3-C4/G3-N/N2 rows to stage the sampled token's actual K/V and
next logits. The C4 `et_g3c4_private_prefill3_v1` computes the same
three-token full prefix through its separate 21-role schedule. A2 already accepts
capacity four and exposes full-capacity K/V, effective length, and keep mask.
No new numerical provider row or A2 ABI is required for this tuple.
This P3 path is an internal same-provider full-prefix numerical reference,
not an independent provider or a general no-cache generation implementation.

## Narrow pending-transcript change

Keep the existing private signatures and the exact C4 model, T1-backed input,
generator, call and pending-output authorities. The C4 call ABI already admits
`(kind,budget)=(2,1)` and has no `P` field. Under a separate C4 feature gate,
widen only prompt/output guards to admit `(P,G)=(2,1)` into a *pending
transcript*, retaining the existing three pairs. `G` remains at most one; `P`
remains at most two. The read-only prompt preflight checks `P+G<=3` before
acquisition or pins;
execution repeats that check against the authenticated owned input. Widen the
matching pending-output reservation, invariant and numeric-readiness guards
together. The Eshkol reservation guard at
`native/g3c4_output_envelope_extension.esk:111` also requires the same bound.
Source currently caps these checks at two; no one-sided admission is allowed.

Reserve one pending ID, run authentic P2 prefill, sample once from its last
row, and stage the position-two token frame and next logits. Numeric output
preparation may record a pending length one, pending cache length three, and
the candidate successor RNG; the existing ID-copy/text-readiness coordinator
may run only against that linked pending output. These are observations for a
private witness, not a successful generation result. On call abort, the
pending output and candidate append are discarded; the committed P2 prompt
cache/binding and generator RNG remain. Greedy/categorical candidate and
successor RNG are checked against the accepted G3-S/Philox oracle without
claiming a committed draw or EOS result. P3 prefill makes no sampling draw.

`g3c4_call_entry_extension.esk` invokes native `call_prepare_end`, which
currently rejects every pending output in `g3c4_model_owner.c`. The accepted
output-envelope witness expects that rejection. This leaf must preserve it:
no call finish, live output, generated ID/text publication, EOS behavior or
successful full-request retry is claimed. A **separate reviewed private
generation frame prepare/commit and output-publication contract** is required
before widening final-preparation guards or claiming those behaviors. The
public C2 facade continues to reject P2/G1.

`src/eshkol_transformer/g3c4_model_owner.c` owns the native tuple, pending
output and frame guards; `native/g3c4_output_envelope_extension.esk` owns the
Eshkol reservation check. G3-C4/G3-N providers, A2 and M3T are unchanged.

## Smallest proving fixture and gate

Extend the C4 prompt/output-envelope fixture with one authentic sealed C4
owner and length-two T1-backed prompt. Stage P2/G1; independently compute
the sampled ID and successor RNG by advancing the accepted G3-S/Philox oracle
once from the P2 last-logit row. Greedy leaves the seed unchanged. Run
accepted P3 prefill in a separate context on
`[prompt0,prompt1,sampled]` solely as a numerical reference. Compare all
256 next-logit words, candidate transaction K/V for both heads at position
two against P3, committed P2 cache length/mask `2/1100`, and pending ID,
text byte and recorded prospective length three. Assert
`call_prepare_end` rejects the pending output, then abort drains it and the
candidate frame while preserving P2 cache/binding and original RNG. Cover
greedy/categorical candidates, byte boundaries, feature-off P2/G1 and
budget-two rejection before pins, allocation/provider cuts, pending-owner linkage,
abort and a fresh transcript after abort. Run normal, repeat,
ASan/UBSan/LSan, source-closure and feature-off symbol gates on supported
pins. The fixture proves no live output, committed third position, EOS
termination, or general no-cache parity.

The next dependency is a separately reviewed private generation frame
prepare/commit and output-publication gate. Only afterward may an Eshkol
result/public-package gate be proposed. Repeated decode, `G>1`, N>1,
general text/IDs, persistence, and CLI3 remain separate.
