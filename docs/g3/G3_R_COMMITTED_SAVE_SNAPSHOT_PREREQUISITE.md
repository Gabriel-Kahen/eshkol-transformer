# G3-R committed-save snapshot prerequisite

**Source-gap analysis of main `09f7ed52`.** The bounded descendant
[native owner leaf](G3_R_COMMITTED_HISTORY_OWNER_LEAF.md) now adds a private
inline ledger and idle copyout, including G1 partial-commit denial. It adds
no Eshkol save authority, file writer, continuation, or acceptance. This is the first
missing same-owner save step in the bounded CPU-f32 `diagnostic-c2` G3-R
[data-only continuation contract](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md).
The accepted G3RCV1 codec and typed G3-T RNG import can encode and reconstruct
data once it has authority; neither attests to a live generator's committed
history.

## Source finding

On main `09f7ed52`, `src/eshkol_transformer/g3t_transport.c`'s `g3t_context` owns `policy[6]`,
`rng[4]`, cache, last logits and transient call flags. It has no committed
ordered IDs, origin-prompt/generated/manual counts, or latched EOS state.
`g3t_frame` holds P2 prompt IDs or one frame token only while a frame is
active. Every manual P1/P2 prefill, manual decode, zero-budget final and sampled
final branch of `et_g3t_private_frame_commit_v1` clears that frame. The sampled
final also copies `successor_rng` into the generator; `call_finish_v1` clears
`final_committed`, `sampled`, `sample_token` and the successor scratch. Thus a
later idle read cannot recover how the same length-two cache arose: P2 origin
prompt, P1 plus manual append, and P1 plus generated token have different
G3RCV1 partitions. `g3t_output` contains generated IDs, `P/G`, and successor
RNG, then detaches from the generator at final publication. It does not own the
complete prompt or intervening manual history. The Eshkol generator registry
entry in `native/g3t_generator_constructor_extension.esk` retains policy and
model/tokenizer roots, not this missing history. No caller-supplied history or
detached output may be promoted to same-owner save authority.

There is also a partial-commit trap: one-token generation commits its P1
prefill cache before sampling. `et_g3t_private_call_abort_v1` clears the frame,
pending output and sample scratch after a later recoverable failure, but does
not undo `prefill_committed` or that cache. The resulting idle generator is
not a completed generation and has no publishable save history. Idle alone is
therefore insufficient snapshot admission.

The existing `et_g3t_private_idle_rng_words_equal_v1` checks a live, idle
generator against **caller-provided** four words. It proves replay did not
draw once the record is independently authenticated; it cannot copy or
authenticate the generator's current words for a new save. Its model-registry,
busy, pin and pending-pointer checks are a useful read-side preflight, not a
save snapshot lease. The current native test word reader is test-only.

## Bounded source contract and remaining admission

The native leaf chooses a guarded fixed-struct copyout from the exact G3-T
generator. It does not provide an Eshkol owning sidecar or stable model/cache
lease. Those still need exact bytevector backing checks, alias rules, failure
status, lifetime, registry identity, lease order and release behavior before
same-aggregate SAVE composition. An API that accepts history or RNG words from
the caller as truth is insufficient.

The same generator owner must track committed events at these exact mutation
boundaries. Preparation or abort must not publish ledger history, although an
aborted G1 call can retain its native P1 prefill cache:

1. Manual P1/P2 prefill commits (`call_kind == 0`): copy the authenticated
   input IDs in order as origin prompt; set origin length one or two, generated
   and manual counts zero, and no EOS terminal even if a prompt byte equals EOS.
2. Manual decode commit (`call_kind == 1`): append its authenticated frame
   token only after the A2 commit; increment manual count. An EOS-valued manual
   byte remains nonterminal.
3. Zero-budget generation prefill commit (`call_kind == 2`, `max_new == 0`):
   commit its P1/P2 origin IDs and zero generated count. Budget completion
   must not invent an emitted EOS.
4. One-token generation P1 prefill commit (`call_kind == 2`, frame kind 1):
   record the authenticated origin ID and mark G1 pending. This is a partial
   state and cannot be snapshotted after an abort.
5. Sampled generation final commit (`call_kind == 2`, frame kind 2): append the
   committed sampled ID, increment generated count, copy the successor four
   RNG words, and latch emitted-EOS terminal iff that ID equals configured
   EOS. Clear pending G1 provenance only in this successful final tail.

A post-prefill G1 abort must leave explicit partial/aborted provenance and
make the idle generator unsaveable, or a separately reviewed rollback must
restore the exact pre-call cache and provenance. A subsequent save must never
mistake its P1 cache for a completed origin-only state. Successful final
commit may clear provisional provenance only while publishing the exact
origin/generated ledger together with the successor RNG.

Before exposing a snapshot, authenticate the exact Eshkol generator entry,
native G3-T registry record, model and tokenizer in one aggregate. Require a
live idle generator with no active call/frame, pending output/logits, active
model call link, pins or partial/aborted provenance; its authentic `c->model`
remains live. Hold stable model/cache and generator leases across the entire
copy, and compare the committed policy, ordered IDs, partition, terminal
classification and all four RNG words from **that owner**. Admit only the
G3RCV1 capacity-two partition and exact diagnostic-C2 policy constraints;
return a detached data-only copy, never a raw native pointer. Release leases
before any C2 or companion file I/O. The save coordinator must then bind this
copy to the authenticated C2 image from the same model state and publish the
companion last; this snapshot prerequisite alone grants no file authority.

## Focused acceptance cuts for that implementation

- Positive P1, P2, P1→manual, P1→generated, zero-budget, EOS prompt/manual,
  generated EOS, greedy and categorical counter carry/exhaustion cases; compare
  ordered IDs, partition, terminal state, policy bits and all four RNG words.
- Reject forged/stale/cross-aggregate owners; busy call or frame; pending
  output/logits; held model/cache/I1/A2 leases; wrong profile/policy or
  malformed partition, with destination unchanged. A valid full P2 history
  remains savable; a later generation request rejects its full context.
- Inject failures before commit and abort, then prove history/RNG/terminal
  unchanged except for the genuine committed P1 prefill; specifically abort
  G1 after that prefill and reject an
  idle snapshot. The poisoned generator needs exact reviewed rollback or
  reconstruction, or closure; a later manual decode cannot launder the erased
  staged origin ID. Inject copyout allocation or lease failures and prove no
  owner mutation, escaped lease or partial destination publication. An
  impossible failure after the A2 commit remains fail-stop.
- Keep feature-off and existing G3-T callers unchanged. Require independent
  source/tests review and supported normal, repeat and sanitizer gates on the
  exact source head before using the snapshot in a G3-R save writer.

Full G3-R still needs file-pair save/publication, fresh candidate retirement,
model/tokenizer restoration, no-draw history replay and the separately admitted
larger repeated-decode profile. This document supplies no such proof.

The restore compositor must also preserve the authenticated G3RCV1 partition:
manual P2 prefill of a saved P1-plus-generated or P1-plus-manual history
reproduces cache/logits but would classify both IDs as origin under the fresh
commit ledger above. Candidate-only reconstruction must transfer the checked
partition and terminal classification before publication, without accepting an
unverified caller-supplied partition. That transfer is a separate reviewed
restore contract, not save snapshot authority.
