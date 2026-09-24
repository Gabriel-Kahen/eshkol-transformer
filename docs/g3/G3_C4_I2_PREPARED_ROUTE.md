# G3-C4 I2 prepared-route seam

This bounded Step 2 implements the four C4 construction operations accepted in
[`G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md`](G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md).
It composes the accepted P1 prepared split and C4 I2 admission bridge. It does
not add a model registry, model constructor, context, cache, sampler, transport,
or public API.

The sole I2 construction record now has eleven slots. Slots 9 and 10 retain the
closed route and exact native owner. The existing M3T begin writes `m3t` and
`#f`; the C4 begin writes `g3c4` and the supplied exact owner. Both routes
use `i2-construction-registry-root`, so only one can be active.

The C4 aggregate installs two source-private fixed adapters. They are not
operation arguments and are never stored in a record. One calls
`et_i2_private_g3c4_construction_parameter_preflight_v1`. The other locates
the matching one of the accepted fourteen native owner parameters, publishes
the P1 handle through the existing
`et_g3c4_private_model_owner_bind_v1`, and immediately calls the same accepted
three-argument bridge for the bound registration. M3T continues to call only
`et_i2_private_construction_parameter_preflight_v1(parameter, handle)`.

Prepare first preflights the anchor and every retained carrier, calls P1 slot
71, then changes only the I2 phase to `prepared`. If P1 prepare raises after
performing its required P1 abort, I2 restores its prior module registry/count
and invalidates the retained carriers before rethrowing the original error.
Prepared seal calls P1 slot 72 before its nonallocating ledger-clear tail.
Open or prepared abort preflights the complete retained set, calls P1 abort
before restoring I2, clears carrier authority and the native owner reference,
and leaves native C4 storage untouched.

The focused runtime test covers prepare failure rollback, exact route/owner
retention, cross-route rejection, prepared abort after native-owner prepare,
native storage survival, and the prepared seal/owner commit order. Its runner is
[`scripts/test-g3c4-i2-prepared-route.sh`](../../scripts/test-g3c4-i2-prepared-route.sh).
The static contract check is safe to run independently:

```sh
python3 scripts/check-g3c4-i2-prepared-route.py
```

The supported strict AOT/runtime runner is intentionally a separate queued
gate while another supported-build job owns the shared compiler slot. Until
that runner and the existing M3T construction gate pass in the released slot,
this change is source-complete but has no runtime acceptance claim.
