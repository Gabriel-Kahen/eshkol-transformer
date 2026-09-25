# G3-C4 private model authority and seeded construction

This bounded Step 3 implements Section 3 of
[`G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md`](G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md).
It consumes the accepted P1 prepared split and C4 I2 route. It adds no public
model facade, forward path, context, cache, sampler, restore constructor,
persistence operation, or G3-S source.

`g3c4-model-registry` is the sole process-lifetime C4 model authority. Each
entry has the accepted twelve slots: root shell, `c4-model`, lifecycle,
native owner, `seeded`, original words, successor words, reserved `#f`,
fourteen canonical handles, pending I2 construction, active-call `#f`, and
`diagnostic-c4`. Pending entries are never admitted by an observer; this step
adds only `g3c4-model-create-seeded-internal`.

Construction runs under the existing `m3-call` aggregate guard. Before native
enrollment it allocates and roots the pending entry, original and successor
vectors, handle vector, module-prefix index, and cleanup ledger, then reads each
back canonically. Captured local owner and construction identities protect
rollback if recording either identity into the rooted ledger itself raises.

The constructor creates the accepted native seeded owner and begins the C4 I2
route with that exact owner. It builds the established fourteen-parameter
module topology, with the C4 position shape `[4,4]`, registers the token/head
tie, validates the complete canonical schedule, initializes indices 0 through
13, and captures original `[1, seed, 0, 0]` and successor
`[1, seed, 292, 0]` words.

I2/P1 prepare precedes native-owner prepare. A final read-only preflight checks
the canonical entry, cleanup ledger, fourteen distinct handles, prepared I2
route, exact owner, word vectors, and sole active construction. The no-fail tail
then seals I2/P1, commits the native owner, clears the pending construction
slot, and publishes `live`. A nonzero native prepared-commit status exits 134
as an implementation defect.

Before that tail, every failure unwinds in the fixed order: C4 I2 abort, native
owner abort, then a `dead` model tombstone retaining only slots 0 through 2.
Native storage is never aborted while P1 handles remain live.

Focused source and native checks are:

```sh
python3 scripts/check-g3c4-model-authority.py
cc ... -c tests/g3c4/model_authority_native.c
```

[`scripts/test-g3c4-model-authority.sh`](../../scripts/test-g3c4-model-authority.sh)
is the queued strict AOT/runtime gate. Its regression covers invalid seeds,
native-owner allocation rollback, pending invisibility, dead tombstones, exact
entry and word state, fourteen unique handles, the tie and position shape,
eval-mode publication, owner seal, guard cleanup, and multiple process-lifetime
models.

The strict AOT/runtime runner, the Step 1 and Step 2 supported gates, and the
existing M3T construction regression remain queued while the shared compiler
slot is occupied. This source slice therefore makes no runtime or package
acceptance claim.
