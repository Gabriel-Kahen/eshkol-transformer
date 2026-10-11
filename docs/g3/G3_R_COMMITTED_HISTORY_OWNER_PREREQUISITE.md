# G3-R committed-history owner prerequisite

**Proposed source-private contract; no implementation or save snapshot is
accepted here.** This prerequisite applies only to the CPU-f32,
`diagnostic-c2`, raw-V256, batch-one, context-two profile in the
[G3-R data-only continuation contract](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md).
It adds no public generation API, checkpoint writer, file format, replay
transaction or larger-context admission.

## Exact missing authority

At source `7291ef80716b8dbe1510b46fe25340614e6c88ba`, the G3-T native
kind-1 `g3t_context` holds `policy[6]`, `rng[4]`, cache, logits and call flags,
but no committed ordered token history, origin prompt length,
generated/manual partition or EOS-terminal provenance
(`src/eshkol_transformer/g3t_transport.c`, `g3t_context`). Within the
generator, P1/P2 prompt IDs and the manual-decode token exist only in
`g3t_frame` while a call is active
(`et_g3t_private_frame_begin_v1`). Each accepted manual P1/P2, manual decode,
G0 and G1 commit clears that frame before the call finishes
(`et_g3t_private_frame_commit_v1`). The Eshkol generator's twelve-slot
registry entry retains policy and the active-call link, but no history
(`native/g3t_generator_constructor_extension.esk`, `g3t-generator-create`).

A completed output is detached and independently releasable. Its kind-8 RNG
clone reads that output's successor words, not arbitrary current-generator
state (`et_g3t_private_output_rng_clone_v1`). Output bytes also omit the
origin prompt and any manual append. The public G3-G input-provenance list
is dispatch metadata for an input shell, not a generator-owned transcript
(`native/g3g_manual_public_extension.esk`). Cache length and logits cannot
recover the partition: genuine P2 prefill and P1 followed by manual decode
can both leave length-two cache state with different prompt/manual counts.
Therefore a writer cannot produce the G3RCV1 history fields from today's
same-owner state, even though the in-memory codec and typed RNG import exist.

## Required prior owner contract

Add a feature-gated, source-private committed transcript to the authentic
G3-T kind-1 generator owner, with capacity two raw IDs in order, length,
origin prompt length, generated count, manual count and an explicit
generated-EOS terminal bit. Require `prompt + generated + manual = length`.
An EOS byte in a prompt or manual append never sets the terminal bit; a
committed generated last byte sets it exactly when it equals the configured
EOS. Do not infer terminal state from a detached output, cache mask or last
byte alone. Specify the native/Eshkol ownership and any snapshot signature in
a reviewed leaf before implementing it; a caller-supplied history vector is
not authority.

Every successful source-private commit path must update this same owner in
its existing no-failure tail, before erasing the frame:

- Manual P1/P2 prefill records the authenticated frame prompt IDs as the
  origin, with both counts zero. A successful prefill may replace the prior
  cache, so it resets the prior transcript in the same commit tail.
- Manual decode appends its authenticated owned-I1 `frame.token`, increments
  only the manual count, and preserves the origin and generated count.
- G0 records the authenticated P1/P2 prompt with zero generated count and no
  draw. G1 records its authenticated P1 prompt when the first prefill commits;
  its second frame appends the committed sampled token, increments only the
  generated count, and derives terminal provenance from that generated token.
  A speculative sample or an aborted uncommitted frame records nothing; a
  previously committed P1 prefill remains in the transcript if a later frame
  aborts.

Before an A2 commit, preflight byte range, capacity, partition, input
identity, current transcript phase, EOS provenance and all required storage.
The transcript publication must then be allocation-free and infallible in
the same closed tail as cache/RNG/output publication. A recoverable failure
before that tail leaves transcript, cache, RNG, policy and old outputs
byte-identical; an impossible failure after its first mutation remains
fail-stop. Generator close zeroes the transcript while preserving its dead
identity tombstone. Keep the transcript on the generator after independent
input, logits, output or output-RNG release.

## Later private save snapshot admission

A separate reviewed save seam may copy committed policy, this transcript,
terminal bit and **the generator's own four RNG words** to data-only staging
before either file writer runs. It must authenticate an exact live
same-aggregate Eshkol generator entry, model and raw-V256 T1 tokenizer, then
the registered native kind-1 owner; reject forged, stale, dead, wrong-kind or
cross-aggregate shells. Hold the shared M3 guard and stable owner leases
through all reads. Require no active call/frame, held model pin, pending
output/logits, busy owner, or competing I1/A2/model lease; detached completed
outputs need not remain live. Check native transcript invariants and policy
against the accepted G3RCV1 bounds, and compare the native policy to the
canonical Eshkol generator entry's slot 7. Cross-check committed history
length against A2 cache length through a stable read borrow released before
publishing staging. Authenticate and validate the destination's exact capacity
and disjointness before copying; a failed admission, borrow, staging allocation
or copy preflight leaves every destination byte unchanged, publishes no
snapshot and mutates no live owner. The kind-8 output RNG clone cannot
substitute for this read.

Focused implementation evidence must cover P1, P2, P1 plus manual decode,
G0, G1 and generated versus prompt/manual EOS; aborted and allocation-cut
calls; retained state after output release; forged/stale/cross-aggregate and
active/borrowed owners; all four RNG words including carry/exhaustion; and
failure-atomic snapshot staging. No such evidence is claimed by this document.
