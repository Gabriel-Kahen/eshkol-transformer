# CPU4 same-canonical snapshot/restore consumer

This private consumer operates on an existing registered CPU4 trainer from
`cpu4_trainer_control_extension.esk`. It retains that trainer's canonical C4
model, seventeen modules, fourteen handles/carriers, O2 optimizer and D2 dataset.
It creates no replacement model, M3T owner alias, TR3 trainer or public facade.
The C4 constructor entry's slot7 remains `#f`.

`native/cpu4_checkpoint_root.esk` establishes one identity universe, extending
D2's T2/P1/I2 root with the existing O2/C2 validators, M3 transport/schedule/model
helpers, G3-C4 construction/model authority, CPU4 P1/binding/control producers,
and the actual I2/O2/D2 restore component bindings. It deliberately does not load
TR3 lease/core/snapshot/joint roots or their trainer authenticator. Existing
component bindings use `tr3-lease-fail-stop` only for two irreversible native
contract defects; this aggregate supplies a fixed-arity direct call to the
already declared CPU4 `_Exit`, with no old lease authority.

The two private consumer signatures are:

- `cpu4-trainer-snapshot-internal(trainer,result-cell)`: the exact empty one-slot
  cell receives one deep C2 owner after the native WRITE parent closes. Caller
  owns that detached owner and releases it with the existing C2 release API.
- `cpu4-trainer-restore-internal!(trainer,source)`: source is an exact live deep
  C2 owner, not bytes or a projection. Source remains caller-owned and unchanged.
  File parsing/checksum authentication belongs to the existing C2 load API;
  this entry does not accept a file path or bypass that loader.

Snapshot preallocates and roots its cleanup envelope before parent publication.
It enters the trainer's snapshot WRITE parent, copies authenticated controls,
clones `module-state-dict` of the exact canonical model and snapshots its genuine
O2 optimizer. The real C2 compose request is the transfer ledger even if a
wrapper raises after publication: either both component owners remain in their
input slots, or both clear and one exact registered C2 owner appears. Cleanup
releases only the authority actually published, before ending the parent.

Restore stages fourteen parameter clones and twenty-eight optimizer moment
buffers under the real C2 owner borrow/token. It verifies every one of the
fifteen P1 entries, version/provider/dtype/device/layout, exact canonical C4 path
and shape schedule, and head/token alias. The position tensor is `[4,4]`, derived
from `g3c4-model-shapes`; the old TR3 `[2,4]` schedule is not reused. Source O2
config/paths/aliases/update count are cross-checked against its P1 topology and
controls. All source accesses close before the receiver's WRITE parent begins. Every later
metadata leaf (config, paths, shapes, aliases and controls) is detached with
the existing bounded O2/C2 copy helpers before the C2 borrow ends.

CPU4 controls stage before the D2/O2/I2 plans can make participants busy. The
consumer prepares the genuine D2 plan, lets O2 take all twenty-eight moment
stages through its published output slot, and appends fourteen exact receiver
carrier/handle/source triples to I2. CPU4 P1 recheck, control precommit, D2 check,
O2 check and I2 prepare all precede the first live write. The sealed order is I2
values/moments, O2 config/options/update count, O2 finalization, D2 cursor, then
CPU4 controls last. All fourteen value clones release before parent end. An
exception after the first write fail-stops; it cannot report recoverable rollback.

Before any write, cleanup ends subordinate source access/borrow first, aborts
I2/D2/O2 plans or releases untransferred stages, drains parameter clones, aborts
the control stage, then ends the parent. Rooted envelopes retain the exact child
identities throughout. A blocked prewrite cleanup preserves its first error and
its exact remaining custody. Either subsequent operation drains pending cleanup
for that same trainer before new idle admission. Captured genuine I2/O2/D2,
control-stage and parent rows distinguish consumed/closed authority on retry;
C2 borrow and owned-carrier field clearing likewise prevents a second end or
release. No new get/attach, generic rollback callback or destructor is added.

Compound factory custody is a **separate unresolved contract/gate**. This
consumer neither allocates nor retires the trainer's binding/O2/D2 children.
The existing candidate Scheme surface passes null optimizer/generator at retire,
and native candidate adoption/retirement reject a live binding lease. A future
factory needs approved closed typed ownership publication/drain for optimizer,
binding and trainer, including construction failures and adopted success. This
consumer supplies no invented attachment or native getter.

Host checks:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.cpu4_checkpoint.test_source_contract
guile --no-auto-compile -s tests/cpu4_checkpoint/host_test.scm
```

The host test executes the actual staging/prepare/commit/snapshot/abort functions
with explicit doubles: all fourteen values/twenty-eight moments, exact C4 shape,
stage-before-busy and controls-last order, poisoned expired metadata,
pre/post-publication faults and blocked cleanup after ledger consumption. It does not admit Eshkol compilation, native byte/shape validation or
sanitizer behavior. Those remain the next integration-owned compiled gate.

The existing CPU4 binding native roster remains the base. The consumer also
needs `native/o2_optimizer.c`, `native/o2_wave2_package_bridge.c`,
`native/c2_x1_canonical.c`, and `native/tr3_c_restore_bindings.c`. Reuse the actual
TR3-C component feature flags from `scripts/test-tr3-c-joint-runtime.sh`:
`ET_C2_I2_MODEL_COPY`, `ET_I2_PRIVATE_OWNED_CLONE_MATCH`,
`ET_TR3_C_I2_RESTORE_PRIVATE`, `ET_TR3_C_O2_RESTORE_NATIVE`,
`ET_C2_O2_RECONSTRUCT_BRIDGE`, `ET_TR3_C_RESTORE_BINDINGS`, and
`ET_TR3_C_D2_RESTORE`, alongside the CPU4 binding/training flags. The O2 bridge
must use `ET_O2_NATIVE_HELPERS_ONLY` to preserve the sole I2 bridge. No full TR3
training/objective object roster or alternate owner is required by this root.
The pinned compiled gate must retain dependency files and IR/import closure.

Exact source dependency: the one production line in
`native/cpu4_trainer_control_extension.esk` from published
`502f0671a9bc4ba0b1f45f50128d9ed0c8d22645` is composed here verbatim. It uses
`d2-core-bytes=?` to compare canonical UTF8 by value; fe9 `equal?` compares
bytevector identity. None of that leaf's fixture edits are imported. The base
P1 module blob is already exactly identical to the d5b3 bounded-helper repair.
This dependency is source composition, not runtime admission.
