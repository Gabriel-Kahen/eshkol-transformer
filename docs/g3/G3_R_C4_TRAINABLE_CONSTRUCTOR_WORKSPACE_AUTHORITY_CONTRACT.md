# G3-R C4 trainable constructor authority

**Proposed private contract; no constructor or workspace implementation is
admitted by this document.** The next gate is constructor-only. A read-only
workspace preflight is a separate later gate. Neither gate runs C4 forward,
backward, an optimizer, a trainer, C2 SAVE/LOAD, or generation.

## Existing authority and the missing transition

`native/g3c4_model_extension.esk` owns the sole C4 model registry and its
atomic seeded constructor. The 12-slot entry has fourteen distinct P1 handles,
one tied token-embedding path at head index 10, and position weight `[4,4]` at
index 13. The seeded entry is tagged `seeded`, has slot 7 `#f`, is prepared in
`eval`, and uses a native C4 owner bound to those handles once. The native
owner, P1 and I2 ledgers are existing authorities, not templates for parallel
registries. `native/g3c4_call_entry_extension.esk` currently admits only that
seeded tag and requires eval mode. Its diagnostic generator is a negative
control for trained-model identity.

The C2 M3T constructor checks context two and position `[2,4]`; its workspace
has two ID slots. It cannot be reinterpreted as C4. The accepted C4 I2 route
had only `i2-g3c4-construction-prepare-eval-internal!`, backed by P1 trusted
slot 71; slot 72 seals a prepared construction. The separate source candidate
adds **C4 train-mode prepare** in P1/I2 at append-only production slot 73.
It reuses the existing P1/I2 prepared seal and abort ledgers, without changing
the seeded path or the M3T route. The train sibling performs the same fallible
schedule, attachment, canonical-path, finalization, mode verification, and
native-prepare checks as `construction-prepare-eval!`, with final mode `train`.
Its focused source and fixture tests are not a compiled runtime admission.
Constructor implementation waits for independent compiled admission of this
train-mode prepare sibling.

## Proposed constructor authority

The private entry is a new **tagged branch in `g3c4-model-registry`**, not a
second model type or copied generator owner. Its proposed operation is
`g3c4-model-create-trainable-internal(resolved-config, tokenizer, initializer)`;
`native/g3c4_trainable_constructor_extension.esk` now contains a source-only
candidate for this operation. It is neither installed nor compiled-admitted.
Admit only the exact resolved
context-four profile: CPU f32, deterministic, vocabulary 256, hidden size 4,
one layer, two query and KV heads, head size 2, and context length 4.
Authenticate a live canonical raw V256 T1 shell with the existing T1 registry;
the first constructor fixture requires the existing C4 call entry's exact
baseline tokenizer fingerprint
`sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704`.
Any broader raw V256 admission needs a later generation-contract change.
Require `run.seed` to equal the exact initializer seed. Preserve the X1
canonical bytes and fingerprint, T1 shell and fingerprint, initializer shell,
algorithm and four words under the entry's rooted trainable metadata. None of
these may be inferred from a matching printable fingerprint alone.

The 12-slot entry keeps existing slots 0..3 and 8..10 with their present
meanings. Slot 4 is the symbol `'trainable`; slot 11 is the exact private
profile symbol `'trainable-c4`, distinct from seeded `'diagnostic-c4`. Slot 7
is `#f` for seeded entries and the following **nine-slot vector** for a
trainable entry. No optional fields or alternate ordering are admitted:

| Index | Slot 7 value | Admission rule |
| --- | --- | --- |
| 0 | `'g3c4-trainable-metadata-v1` | Exact tag |
| 1 | resolved X1 config shell | Exact live shell in the X1 registry |
| 2 | X1 canonical serialization | Value from `x1-private-config-canonical` for slot 1; retain an owned copy |
| 3 | X1 fingerprint | Value from `x1-private-config-fingerprint` for slot 1; retain an owned copy |
| 4 | raw V256 T1 tokenizer shell | Exact live shell in the T1 registry |
| 5 | T1 fingerprint | Value from the authenticated T1 entry for slot 4; retain an owned copy |
| 6 | M3T initializer shell | Exact live shell in the M3T registry |
| 7 | `'n3k.philox4x32-10.uniform-f32.v1` | Exact `m3t-public-initializer-algorithm` result for slot 6 |
| 8 | four-word vector | Owned copy of the authenticated live initializer words at construction admission |

The entry remains rooted while live. Its native owner is the exact owner at
entry slot 3, not a second pointer in metadata. Recheck shell identity through
the corresponding registries, and compare canonical serialization and
fingerprints by value; matching printable values alone cannot substitute for
the shell checks. Slot 7 is private and must not become a public owner or path
export. This is a tagged extension; seeded entry bytes and ordinary
feature-off admission stay unchanged.

Slots 5 and 6 are distinct, owned four-word vectors read from the native C4
owner. For the admitted nonnegative exact i64 seed, the live M3T initializer
words and C4 native original words must each be exactly `[1, seed, 0, 0]`.
The C4 native owner initializes the fourteen parameters in its existing
order; its successor words after initialization and native prepare must be
exactly `[1, seed, 292, 0]`. This is the current C4 owner's fixed draw
transition: its eight rank-two uniform initializations cover 1,168 f32 values
in 292 four-value counter blocks; rank-one initialization does not draw. It does
**not** consume or advance the M3T initializer; the M3T shell is an
authenticated seed and algorithm authority. Compare slot 5 with the M3T
metadata words and native original, and slot 6 with the native successor,
before publishing live. Do not infer either snapshot from the seed without
native reads. The four-word snapshots are not exposed as mutable caller data.

Construction uses the existing aggregate guard, C4 model root/cleanup root,
native C4 owner, and C4 I2/P1 transaction. Validate config, T1 and initializer
before enrollment. Root a pending entry and cleanup ledger; create one native
owner; register the exact fourteen canonical ordered paths with shapes
`[4,4]` four times, `[4,8]`, `[8,4]`, `[4]` at 6..9, `[256,4]` at 10,
`[4]` at 11..12, and `[4,4]` at 13. Register `token_embedding.weight` as
the exact handle at index 10, never a fifteenth parameter. Initialize and
bind that owner once. Prepare `train` through the new P1/I2 sibling, prepare
the native owner, and complete all fallible cross-registry checks before the
existing no-failure prepared commits. Publish `live` only after both commits.
On every precommit error, revoke P1/I2 identities before aborting native
storage, mark the pending entry dead, and clear cleanup authority. A failed
attempt cannot expose a live model or a usable handle.

## Separate future workspace gate: identity and shape only

The constructor gate does not create a workspace or call a workspace
preflight. After separate source review and compiled constructor admission,
the proposed private operation
`g3c4-trainable-workspace-preflight-internal(model)` returns a typed status,
not parameter arrays, tensors, pointers, or scratch. It admits the exact live
trainable shell in the **same C4 registry**, with matching rooted config, T1,
initializer and native owner; `train` mode; no active construction, generation
call, workspace or lease; exact fourteen ordered P1 handles and native
bindings; CPU f32 contiguous values of the listed shapes; and the exact
head/token-embedding tie. Native checking may scope the existing C4 fixed-14
pin begin/check/end operations inside one call, but must release pins on all
paths and expose only status to Eshkol. Admission leaves the model, parameter
bits and registry unchanged. No workspace allocation or forward operation is
authorized by a successful status.

This later preflight must not call `module-parameters` or its
`commit-module-observation!` path: that P1 observation installs canonical
paths and finalizes modules. Its fourteen-handle and tie checks need a
separately reviewed read-only P1 topology authority (or an immutable
construction receipt), plus scoped native pins. This topology API and the
status representation are not specified or implemented by the constructor
gate.

Constructor positive tests must use a newly constructed trainable owner in
its original registry. Constructor negatives must reject foreign or unresolved
X1 shells, foreign or stale T1/initializer shells where their lifecycle admits
that state, changed canonical identities, context-two config,
wrong initializer seed or algorithm, and forged or changed four-word snapshots.
X1 resolved shells have permanent lexical registry membership and no
withdrawal/close route; a stale resolved X1 shell is not a representable
source state. The unresolved-shell negative exercises its actual admission
boundary without mutating X1's accepted private registry.
A constructor fault matrix must separately prove rollback before and after
P1/I2 enrollment and native prepare, leaving no live shell or usable handle.
The source candidate has one inert, source-private fixture hook at the
post-registration, post-P1-prepare and post-native-prepare cuts. It is neither
a public API nor an installed route. The compiled fixture must use it to
capture the exact pending shell/handle/owner, inject one failure per cut,
and prove P1 handle revocation and native owner abort after each failure.
Feature-off builds must retain the ordinary seeded ABI and reject private
trainable symbols. The later workspace gate must separately reject seeded or
foreign model shells, swapped handles, wrong position shape or dtype, broken
tie, eval mode, active generation/workspace/lease, and incomplete construction.
Each failure must preserve the live model and unrelated registry entries;
workspace identity/shape negatives and no-mutation proof are not constructor
admission.

## Later gates

Trainable-to-eval must transition the **same live entry and owner in place**;
generation lookup must authenticate the new tagged entry and its exact bound
handles after that transition. Copying parameters into another seeded owner
does not count. A separate composition gate must connect the individually
reviewed C4 VJP leaves (including embedding and attention), give every
intermediate tensor an exact shape, and check complete gradients numerically.
Only later gates may add optimizer/TR3/D2, C2 SAVE and fresh LOAD into the same
registry, then G3 continuation. The isolated VJP leaves and this contract do
not establish any of those claims.
