# G3-R C4 trainable workspace topology authority

**Proposed source contract only.** This document follows the private C4
trainable constructor at `59670d39a9fa272e07b34394981662b962dfa25c`
(tree `226fdb44521173ceee37b57b369aef4eaee24660`). It specifies a first
read-only topology and owner preflight; it does not implement a workspace,
trainable forward/backward graph, optimizer, trainer, or checkpoint route.
Compiled constructor admission is a separate prerequisite.

## Authority already present

The constructor roots one 12-slot entry in `g3c4-model-registry`: exact model
shell at slot 0, `'live` at 2 after commit, native owner at 3, `'trainable` at
4, nine-slot X1/T1/initializer metadata at 7, fourteen ordered P1 handle
shells at 8, no construction at 9, no call at 10, and `'trainable-c4` at 11.
It registers `token_embedding.weight` as the **same** P1 handle as
`head.weight` at index 10. It binds each of those handles once to the
corresponding native owner parameter during the I2 construction transaction.
The sealed native owner retains its own fourteen parameter/identity bindings.

The canonical paths and dimensions are `g3c4-model-paths` and
`g3c4-model-shapes`: four `[4,4]`, then `[4,8]`, `[8,4]`, four `[4]`,
`[256,4]`, two `[4]`, and final position `[4,4]`. The existing TR3 fixed
profile has position `[2,4]`; its status routine cannot be reused as a C4
topology authority. The constructor's `module-construction-parameters-internal`
was valid only while its construction was open. `module-parameters` is also
unsuitable: it calls `commit-module-observation!` and can install canonical
paths/finalize modules. No read-only live C4 topology status is currently
provided by those calls.

P1 already owns the true module/handle records, mode and provider identity.
I2 uses provider `'i2-dense-cpu-f32-v1` and retains its construction/admission
ledger. Native C4 already has the fixed-fourteen scoped
`et_g3c4_model_pins_begin_internal`, `check_internal`, and `end_internal`
operations and validates bound parameter identities. The existing
`et_g3c4_construction_parameter_preflight_internal` requires the **staged**
owner and therefore cannot authenticate this committed live owner.

## First proposed private statuses

Append a source-private trusted P1 slot **74**, leaving slots 0..73, the
ordinary feature-off ABI and seeded C4 entry untouched. Its candidate
operation is `module-c4-train-topology-status-internal(root, handles)`.
Inputs are the exact rooted model shell and its fourteen handle **shells**;
the result is a closed integer status, never a tree, handle, tensor, carrier,
or raw-pointer array. It must authenticate the shell registry, true root and
seventeen-module parent/child/binding graph, I2 provider, `train` mode for
every child, fourteen canonical paths and distinct handles, shape/dtype/
device (CPU f32; final position `[4,4]`), and the exact head to
token-embedding tie. A forged vector with matching printable paths must fail.
It must not call a finalizer, `module-parameters`, mutation/registration,
construction begin/seal, or any public tree observation.

A separate private native status helper must admit the **sealed live** C4
owner from its registry, reject an active native call, and scope the existing
fourteen-pin begin/check/end around its own bound parameters and identities.
Its fixed C4 pin profile checks contiguous row-major value storage and the
same fourteen tensor shapes.
It returns only a closed status. It must release every acquired pin on both
success and failure, and must not return a view, parameter, identity, pointer
array or workspace. This is distinct from the staged construction helper.
The helper's exact symbol/signature and status mapping are a next source
review decision, not an API claimed to exist here.

The combined caller must independently authenticate the exact live trainable
C4 entry and its X1/T1/initializer metadata and RNG snapshots in the same
registry, require `module-mode-internal='train`, and reject active C4 cleanup,
construction and call ledgers before invoking either status. Its result is
status-only; no parameter bits, registry entry, mode, RNG, tie, or native
owner state may change on any failure. The status may require temporary
native pins solely to prove topology and must leave no pin held afterward.

## Required association proof and next gate

The constructor's one-time native binding is a prerequisite, **not** by
itself a complete live owner-to-P1 association proof. The implementation
must compare the exact current fourteen P1 shells to the native owner's
fourteen bound identities without sending a raw carrier array through
Eshkol. A candidate is a native per-index scalar status check against the
owner's registered identity, composed with the P1 status and complete native
pin status. Review the exact native/Eshkol calling convention and error
mapping before choosing that helper. A swapped P1 handle, foreign owner,
stale shell, changed binding, wrong final position shape, wrong dtype/device,
broken tie and eval-mode controls must each fail without mutation. Pin
release and unchanged owner/parameter words must be checked after failures.

This first gate can establish **topology and owner identity only**. An
active generation/workspace/lease exclusion cannot be declared complete until
the relevant C4 workspace and trainer lease ledgers exist and are checked
read-only. The current TR3 enrolled-handle overlap predicate is a useful
existing control, but it covers its current fixed trainer registry; it does
not authorize a future C4 trainer or substitute for its lease ledger. A
later reviewed gate must combine full exclusion, storage nonoverlap, and
resource lifetime before allocating a workspace or invoking a VJP.
