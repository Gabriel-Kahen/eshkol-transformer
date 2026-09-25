# G3-T authentic native context leaf

This source-private leaf follows the accepted [G3-T contract](G3_T_PRIVATE_CONTRACT.md)
and the [M3T model-admission precursor](G3_T_MODEL_ADMISSION_LEAF.md).
`src/eshkol_transformer/g3t_transport.c` includes `m3_model.c` once, so it sees
the actual M3T native owner registry without a new cross-package accessor or a
copy of M3T ownership. It includes no C4 source. The matching private header
declares only the implemented accepted subset: `generator_seed`,
`generator_close`, `call_acquire`, `call_abort`, and three scalar error reads.
No installed header, public generation name, or full G3-T package is added.

`generator_seed(model_owner, seed, mode, temperature_bits, k, p_bits, max_new,
eos)` authenticates the exact sealed M3T native owner and its original/successor
initializers and all 14 canonical native parameter identities before reading
owner fields. The trusted Eshkol caller must first run
`g3t-model-entry-live` under `m3-call` to check the matching P1 shell, profile,
topology, eval mode, and idle frame. It accepts only the C2 policy and seeded
Philox words `[1,seed,0,0]`, allocates an empty A2 cache with exact
`[1,1,2,2,2]` storage, and enrolls the stable context only after every
allocation succeeds. The context retains the model, policy, RNG words, cache,
and embedded fixed-14 pin record. Failed construction leaves no enrolled
context and preserves the first G3-T or A2 error snapshot.

`call_acquire(ctx, kind)` accepts only kinds 0..2, authenticates both registries,
acquires the shared fixed-14 pins, then publishes `owner.active = ctx` and the
busy token. `call_abort(ctx)` checks/drains all pins and clears only that token;
impossible acquired-owner or pin invariant defects fail-stop. It has no frame
or output to roll back yet. Close rejects active work, destroys
the owned A2 cache, scrubs policy/RNG/model roots, and retains a dead registry
header. Exact dead close is idempotent; forged, wrong-kind, and stale pointers
cannot regain authority. Each native boundary requires external serialization.

The pinned source/AOT witness constructs a genuine M3T C2 model, validates the
Eshkol admission precursor, and exercises native identity, policy, allocation
failure/retry, A2 ownership, same-model manual-frame exclusion, fixed pins,
abort, close, and tombstones in normal/repeat/sanitizer modes. Test hooks are
compiled only for the witness and are absent from the production native object.

The next dependency is the rooted Eshkol generator constructor and matching
native kind-8 RNG owner, followed by the frame and result state machine. This
leaf cannot prefill, decode, sample, commit output, or publish generation.
