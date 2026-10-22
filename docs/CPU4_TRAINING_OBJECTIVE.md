# CPU4 single-microbatch objective and schedule

Source private; not installed and not a complete trainer. Uses the canonical
binding and exact CPU-f32 N1/T4/V256/D4/H2/Dh2/L1/F8 transport. No M3T owner cast.

`ET_CPU4_OBJECTIVE_PRIVATE` additionally enables two transport calls:

- `et_cpu4_private_batch_copy_v1(lease, view, bytes, kind)`: checked genuine D2
  callback view `[1,4]`, i64 inputs/targets (kind0/1, 32 bytes), bool mask (kind2,
  4 bytes). Destination includes the standard 8-byte bytevector header. IDs must
  be 0..255 and masks0/1. No numeric model operation occurs here.
- `et_cpu4_private_objective_prepare_v1(logits, seed, targets, mask, observation)`:
  distinct registered same-lease logits/seed owners; target32/mask4/observation12
  bytevectors. L2 forward CE, L3S masked reduction and numerator seed, L2 VJP.
  The exact accepted T4 L3S implementation comes from E3 e0f9fae0 (production
  unchanged in accepted d5970d77). K1 dispatch owns all arithmetic. Complete
  staging precedes seed write and the final three little-endian f32 observation
  words `[numerator, weight, mean]`. Failure preserves seed and observation.

`cpu4-training-microbatch-internal!(binding, live-batch, ordinal, observation)`
uses the actual canonical binding/P1 roster, authenticates the batch owning D2
state and exact raw-byte fingerprint/V256/N1/T4 profile, copies and closes all D2 views,
enters native READ, executes the explicit 21 forward and 25 reverse roles in
Eshkol, then atomically contributes fourteen complete gradients with the masked
weight and ordinal. It drains its native children and leaves READ. It preserves
parameter values and caller batch ownership. Cleanup defects fail-stop; no
recoverable error follows gradient publication. Observation publication is
staged until successful contribution. The caller still owns dataset transaction,
rollback, accumulation policy, WRITE/O2 update and trainer counters.

Development oracle computes masked CE independently through all1192 canonical
values using binary64 central differences. The native fixture checks full1024
seed and1192 parameter gradients, invalid IDs/masks/shape/dtype, cross-lease and
native record/tensor aliases, K1 allocation failure, borrowed seed rejection,
first error and unchanged destinations, metadata/value preservation, and payload
counts. `scripts/test-cpu4-training-objective.sh FRESH_EVIDENCE` is the supported
normal/repeat/sanitizer recipe. Runtime acceptance and genuine Eshkol schedule
execution remain NOT RUN until sealed evidence says otherwise.
