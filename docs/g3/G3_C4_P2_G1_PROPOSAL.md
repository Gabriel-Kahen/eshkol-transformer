# G3-C4 private P2/G1 successor proposal

**Proposed contract; no implementation or public API is accepted.** This is a
single-row `diagnostic-c4` continuation of the accepted C4 P2 prefill and
one-token frame. It does not change the installed M3T/diagnostic-C2 or G3-G
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
This P3 comparison is a full-prefix C4 parity witness, not a general
no-cache generation implementation.

## Narrow admission and transaction change

Keep the existing private signatures and the exact C4 model, T1-backed input,
generator, call and output authorities. Under a separate C4 feature gate,
admit `(P,G)=(2,1)` in the generation call while retaining `(1,0)`, `(2,0)`,
and `(1,1)`. `G` remains at most one; `P` remains at most two. Read-only
`et_g3c4_private_prompt_prefill_preflight_v1` must reject `P+G>3` before
acquisition or pins. The execution preflight repeats that check against the
authenticated owned input. Widen the matching output reservation, invariant,
prepare, and final publication checks together: source currently caps the
tuple at two in the prompt preflight, output reservation, output invariant,
output preparation, and final preparation. No one-sided admission is allowed.

For P2/G1, reserve the owned one-ID output before prompt commit, run the
existing P2 prefill, sample exactly once from its true last row, append at
position two, stage ID/raw-byte text/RNG, and publish cache length three and
generated length one in the existing no-failure final tail. EOS includes the
emitted byte in IDs, text and cache. Greedy preserves RNG; categorical consumes
one accepted Philox block. A recoverable post-prefill failure retains only
the committed prompt cache/binding and the old RNG, with no partial output;
precommit failure preserves the entry state. The public C2 facade continues to
reject P2/G1 and no C4 public facade or CLI command follows from this leaf.

`src/eshkol_transformer/g3c4_model_owner.c` owns the tuple guards, output
invariants and commit; `native/g3c4_primitives_provider.c`,
`native/a2_kv_cache.c` and M3T are unchanged. The focused witness belongs
with `tests/g3c4/test_prompt_prefill.c` and its output/last-logit successors.

## Smallest proving fixture and gate

Extend the native C4 prompt/output fixture, using one authentic sealed C4
owner, one authentic length-two T1-backed prompt, and two independent C4
generator contexts with identical parameters and seed. One context executes
the P2/G1 transcript; the other runs accepted P3 prefill on the exact
`[prompt0,prompt1,sampled]` IDs. Compare all 256 next-logit words, K/V for
both heads at all three committed positions, length three, keep mask `1110`, the
sampled ID, raw byte, and successor RNG. Exercise greedy and categorical,
EOS equality/inequality, byte boundaries, P2/G2 preflight rejection before pins,
every newly reachable allocation/provider cut, output release and retry. Run
normal, repeat, ASan/UBSan/LSan, existing source-closure and feature-off
symbol gates on supported pins before accepting this private tuple.

Only after that gate should G3-M/G3-G propose a C4 public package. Repeated
decode, `G>1`, N>1, general text/IDs, persistence, and CLI3 remain separate.
