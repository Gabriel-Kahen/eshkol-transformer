# Trained C4 checkpoint prerequisite for G3-R multi-token continuation

**Source-grounded proposal only.** This document defines the first upstream
gate for a *trained* context-four checkpoint. It does not admit a C4 trainer,
restore, generation profile, or persisted continuation.

## Current boundary

The current G3-R pair producer targets a genuine CLI3/TR3 context-two C2 image.
The CLI3 command fixes `model.context-length=2` and D2 sequence length two
(`native/cli3_b_extension.esk`); its cross-command fixture uses the same
configuration (`scripts/test-cli3-cross-command-chain.sh`). The trainable M3T
constructor checks context two and registers a `[2,4]` position table
(`native/m3t_transport_extension.esk`). TR3's private factory repeats the
context-two profile check (`native/tr3_c_private_factory_extension.esk`). The
installed CLI3/G3-G generator is bounded to one output byte. Relabeling its
C2 image as context four would change both config and a parameter shape.

The separate G3-C4 model constructor has fourteen unique parameters, a
`[4,4]` position table, and distinct `diagnostic-c4` identity
(`native/g3c4_model_extension.esk`). Its accepted C4 numerical provider is
forward-only; construction and P2/G2 generation do not establish a trainable
M3T backward path. C2/K2 already admit `[4,4]` alongside the other listed
C4 shapes (`docs/C2_TRAINING_STATE.md`), but that storage witness does not
prove a trained model or live restore. The current 288-byte G3RCV1 companion
encodes at most two history IDs and `max_new<=1`
(`docs/g3/G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md`); it cannot be reused for
a four-position P2/G2 history by changing only a profile tag.

## Earliest upstream gate: trainable C4 model

Define and review one exact, feature-local M3T C4 construction/forward/backward
contract before extending CLI3 or G3-R. Its model must retain the existing
fourteen unique parameter paths and head/token-embedding alias, change only
the parameter-shape schedule's position table to `[4,4]`, and bind the
resolved context-four CPU-f32 configuration, seed, tokenizer identity,
parameter identities and native
provider in one registry. Specify the actual tensor shapes at every C4
forward and VJP operation, valid lengths/masks, and failure atomicity. Every
new differentiable operation requires numerical gradient checks, including
position rows 2 and 3; a C4 forward-only provider cannot stand in for these.
The context-two constructor and its diagnostics remain exact and independently
accepted. No implicit truncation, scalar fallback, or seeded G3-C4 owner may
be called a trained C4 model.

After this gate, extend TR3's factory/step/evaluate and D2 sequence choice
under an explicit context-four profile. A real optimizer update must change
the C4 model and its C2 SAVE image; independent pre/post parameter and loss
observations must identify the changed weights. Keep complete trainer state:
model, 28 shape-matched moments, scheduler, X1 RNG, tokenizer, D2 cursor,
configuration and token count. Admit C2 LOAD into a *fresh, live C4 M3T*
receiver with exact config, path, alias, shape and parameter-bit checks. The
same-registry G3-C4 generator must consume that receiver's actual parameters
and eval mode through a reviewed model-authority bridge; a separately seeded
four-position owner or copied C2 bytes are negative controls.

Only then extend the command/fixture or use a private producer to train and
save one exact context-four checkpoint. The minimal fixture has authentic raw
V256 T1, CPU f32, hidden four, one layer, two query/KV heads, context four,
D2 sequence length four, at least one nonzero update, and a P2 prompt. Prove
the same C2 bytes and parameter bits are loaded into the generator's registry
in a fresh process. Keep context-two tests unchanged and reject cross-profile
restores. Versioned C2 wire format and K2 shape admission need no speculative
change for this fixed profile.

## Downstream G3-R gates

1. Admit the existing private C4 P2/G2 first/second frame and terminal/EOS
   path against the **restored trained receiver**, with exact learned-position
   and provider identity. Compare each sampled ID, raw byte, K/V, cache mask,
   256 logits and all four successor RNG words to independent references.
   First-token and second-token EOS, no EOS, greedy and categorical must be
   separate cases. A forced EOS value is a terminal-control witness; an
   ordinary policy must also exercise actual repeated decode.
2. Define a versioned, checksummed C4 companion record with ordered history
   capacity at least four, prompt/generated/manual partition, complete policy
   and four RNG words, canonical T1 fingerprint and full C2-image digest.
   Decide its exact bytes and import APIs in a separate reviewed contract.
   The context-two G3RCV1 decoder must reject this distinct profile/version.
3. Complete the existing G3-R retained-owner/ledger restore prerequisites,
   then replay the saved history through the trained live C4 model with zero
   sampling draws. Compare pre-save and fresh-process cache/logit/RNG state
   before continuing from the same checkpoint. Check unchanged C2 bytes,
   malformed-pair negatives, no partial publication, and bitwise uninterrupted
   versus resumed generated IDs, EOS, lengths and successor state.

A P2/G2 call proves at most two generated tokens and second-token EOS. Broader
G3 repeated decoding still needs a separately admitted general loop and
larger context; this fixed C4 witness cannot establish it.

## Evidence and handoff

Each gate needs an immutable source/tree, independent source-and-test review,
focused numerical and negative tests, a pinned compiled normal/repeat and
sanitizer receipt, and a post-exit file seal. Training, SAVE, fresh LOAD,
same-registry generation, and persisted continuation are distinct results;
report the first unmet dependency rather than promoting a source fixture or
storage-schema test to end-to-end acceptance.
