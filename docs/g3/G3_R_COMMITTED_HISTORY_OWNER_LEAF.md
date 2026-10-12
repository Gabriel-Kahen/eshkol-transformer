# G3-R committed-history owner leaf

This source-private, feature-gated owner is the native G3-T kind-1
`g3t_context`, not an Eshkol registry shadow or a detached output. The inline
record holds two ordered raw V256 IDs, length, origin prompt length, generated
count, manual count, a generated-EOS terminal bit and a pending-G1 provenance
bit. Its zero-initialized
state represents no committed prefill. The generator retains it after input,
logits and output release; close zeroes it before leaving the dead identity
tombstone. The gate is `ET_G3R_COMMITTED_HISTORY_PRIVATE` and requires the
accepted final-publication, manual P2 and manual decode native routes.

Each P1/P2 manual prefill records the frame's authenticated prompt. The
existing old-cache preflight requires an empty prior cache; this leaf does not
admit replacement after a committed prefill. G0 and G1's first frame
also record the frame prompt. P1 uses `frame.token`; P2 uses the frame's
verified `prompt_ids[0..1]`. Manual decode appends `frame.token` as manual;
G1's second frame appends it as generated and clears pending provenance. Only
that generated append can set
the terminal bit, by comparison with `policy[5]`. Every path preflights byte
range, capacity, partition, phase and EOS provenance before A2 commit. The
record mutation is allocation-free inside the established no-failure tail,
before frame clearing. Failed preparation or aborted frames leave the last
committed record intact; a G1 prefill remains after a later sample/frame
abort. The record invariant is `prompt + generated + manual = length <= 2`.
The G1 first-frame commit marks its P1 transcript pending. An abort retains
that bit and cache; a native idle snapshot, later manual decode or new prompt
preflight rejects the poisoned generator. Closing it clears the bit. The
bounded leaf has no rollback or reconstruction entry.

The only read added is native
`int64_t et_g3t_private_committed_history_snapshot_v1(void *generator,
et_g3r_committed_history_snapshot_v1 *destination)`. The fixed struct contains
`policy[6]`, the generator's `rng[4]`, `ids[2]`, length, prompt length,
generated count, manual count and generated-EOS terminal bit. Native admission
requires an exact live idle kind-1 owner, a valid model, no active call/frame,
pin, pending logits/output or model activity, and valid policy/RNG/transcript.
It rejects pending G1 provenance even after a successful abort returns idle.
It checks destination non-null and non-overlap with the native owner before a
single final copy. The C caller must already own a writable, sufficiently
large, disjoint destination and external serialization.

This read is **not a SAVE snapshot or public API**. A later Eshkol same-
aggregate wrapper must authenticate its exact generator entry, model and raw
V256 T1 tokenizer, hold the M3 guard and stable leases, compare native policy
to entry slot 7, cross-check A2 cache length through a released read borrow,
and validate destination capacity/disjointness before calling this read or
publishing data-only staging. No file writer, replay, checkpoint format or
public generation API is added here.

Focused local evidence: `scripts/test-g3r-committed-history-owner.sh` compiles
the feature-gated native source, runs the owner-core P1/P2/manual/generated
partition and EOS test, then exercises the actual native G1 prefill commit,
abort, denied idle snapshot, detached output release and close functions with
controlled model/tensor/A2 stubs. It injects a recoverable A2 commit rejection
and an I1 borrow allocation cut before a G0 commit. A helper-built completed-G1
ledger and detached output release remain snapshot-positive; the sampled native
final commit and `call_finish_v1` are not exercised by that positive. A source check maps the
native hooks and same-aggregate Eshkol wrapper routing. The controlled stubs
do not prove real A2, model pins, or full P1/P2/manual/G0/G1 Eshkol runtime
under the new gate. Eshkol SAVE admission and A2 cache-length snapshot
equivalence remain downstream gates.
