# CPU v1: one bounded context-four profile

Integration decision, 2026-10-22. This is the necessary shared-profile source target;
implementation and compiled admission remain open. The user authorized the scoped
CPU extension. It does not broaden support to arbitrary geometry, dtype or device.

## One model through the complete workflow

Use the existing C4 topology/model identity
`eshkol-diagnostic-byte-decoder-c4-v1`: CPU f32, batch one, context/sequence four,
vocabulary 256, hidden size four, two query/KV heads of size two, one layer,
FFN width eight, learned positions, affine LayerNorm, exact-erf GELU, tied token/head
weight, no dropout. Training and evaluation process actual four-position rows;
there is no implicit two-position truncation or context-two-to-four conversion.
The canonical raw-byte tokenizer remains unchanged.

The fourteen unique parameters keep their established canonical order: four
`[4,4]` attention matrices, `[4,8]`/`[8,4]` FFN matrices, six `[4]` normalization
vectors, `[256,4]` token/head weight and `[4,4]` learned positions. This is 1,192
f32 values / 4,768 unique value bytes. The token/head tie is one owner, not a copied
parameter. Activation `[1,4,4]`, attention `[1,2,4,2]`, FFN `[1,4,8]`, logits
`[1,4,256]`, and genuine D2 input/target/mask `[1,4]` contracts must be explicit.

Reuse the sole C4 model authority and accepted seeded construction. Training,
evaluation and generation must authenticate that same canonical model and its
original fourteen parameter handles; source review must define any delegated
execution adapter without creating a second authority or presenting C4 as the
legacy context-two M3T owner. An admitted checkpoint restore creates the same
C4 model identity with every parameter word authenticated from the checkpoint.
An equal-seed fresh constructor is not restored trained state.

`C2` names the checkpoint format/workstream; it is not permission to reinterpret
legacy context-two model geometry. Preserve the versioned, checksummed,
non-executable envelope. Extend only the explicit model/state capability needed
for this fixed profile. Persist model/optimizer/scheduler, RNG/initializer state,
configuration/token count, tokenizer identity and exact D2 cursor. Validate all
parameter and optimizer shapes, original tie and profile identity before publishing
any restored candidate. Unsupported legacy callers must reject the C4 schema.

## Owners and dependency sequence

1. **Integration + existing independent reviewer:** finalize exact construction,
   ownership, transport, state-shape and restore contracts from current source;
   record decisions in the completion index. Public names and native signatures
   are not invented by this target document.
2. **CLI3 upstream implementer:** add bounded C4 model/training support, actual T4
   forward/backward transport, exact reviewed provider rows and reproducibility
   state. Preserve legacy context-two behavior and explicit capability negatives.
3. **G3:** bind checkpoint restore and the terminal/typed/public generation path to
   that same C4 authority and parameter words, preserving closed commit/rollback
   and preallocated public provenance. Implement genuine save/replay/continuation.
4. **E3:** adapt the evaluator to the exact admitted T4 model/provider and objective
   contracts. Keep numerical, mode/gradient, failure/rollback and resource evidence
   distinct from prior context-two/private/provider-only results.
5. **CLI3 consumer, then integration:** compose only landed upstream exports into
   train → checkpoint → evaluate → multiple-token/EOS generation; run the retained
   walkthrough, combined independent review, exact-head CI and merged-main checks.

Current in-flight gates finish unchanged and remain bounded predecessor evidence.
Future heavy gates prioritize the shared C4 path. No duplicate legacy learning or
broad retention run takes priority over an executable C4 prerequisite.

## Proving gates and negative compatibility

The existing source check `python3 -B scripts/check-g3c4-model-authority.py` passes
on integration source; it proves the existing constructor's source contract only.
The next existing executable construction gate is
`bash scripts/test-g3c4-model-authority.sh` on the pinned supported tuple under the
shared heavy lock. Authenticate reusable exact-source results before running it.
The new upstream owner must then supply a frozen, executable T4 forward/VJP gate
before downstream consumers: independent logits/loss/VJP references, numerical
checks for all differentiable operations/parameter contributions and explicit
new shape-row admission with no scalar Eshkol or CPU fallback.

Required rejection cases include legacy `[2,4]` positions under the C4 schema,
wrong model identity/context, missing/duplicate parameters, broken token/head tie,
malformed state/moment shapes, noncanonical tokenizer/cursor, unsupported dtype
or device, corrupt or mismatched checkpoint/companion records, busy/borrowed owners
and restore cuts. Failure must leave the original model/state/output unchanged
and permit the specified valid retry.

Current-source CPU acceptance requires one-batch overfit, held-out improvement,
complete-state interrupted/resumed equivalence, numerical/gradient correctness,
evaluator atomicity and resources, and genuine multiple-token/EOS generation plus
fresh-process save/reload/continue equality. State exact supported horizons,
measured time/memory and unsupported cases; no language-quality, stochastic-training,
GPU, mixed-precision, arbitrary-size or scaling claim follows from this tiny model.
