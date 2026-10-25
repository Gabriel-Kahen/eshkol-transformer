# Private C4 workspace ownership source leaf

**Source candidate only.** This leaf follows the reviewed authority union and
P1 slot 75. It implements the three private operations in the
[workspace contract](G3_R_C4_WORKSPACE_OWNERSHIP_CONTRACT.md):
`g3r-c4-workspace-create-internal(model)`,
`g3r-c4-workspace-reset-internal!(workspace)` and
`g3r-c4-workspace-release-internal!(workspace)`. A caller receives a closure
shell, never the native record, parameter array or carrier. The sole C4 model
registry remains authoritative; a separate list roots only live workspaces.

Create checks the exact trainable entry/metadata, slot 74 association,
fourteen current carriers, native no-context scan (including an idle cached
generator), and native parameter lease status. It stages one native allocation,
then repeats the owner, metadata, association, context, lease and original
carrier checks under `m3-call`. The native prepare rechecks before a
no-allocation commit sets `owner->active`; the C4 entry slot 10 and live list
are then published by vector writes. A guarded abort frees the staged native
record on every precommit exception. The private fault hook tests the
after-stage and before-commit cuts.

Reset requires the exact live C4 and native edges, slot 75 lifetime topology,
slot 74 train topology, all original handle/carrier identities and native
per-index bindings. It has no scratch or graph to clear. Release rechecks the
same live identity with slot 75, so a uniformly finalized eval tree remains
releasable. Native cleanup checks held parameter/tensor leases before clearing
its edge and freeing the record; only precomputed Eshkol edge/tombstone writes
follow. The dead closure shell may be released again. Dead rows leave the
global live list, so a discarded shell and its tombstone are GC-reclaimable;
the native record is freed on each release. A native nonce distinguishes
reused allocation addresses.

Local native normal/repeat/ASan+UBSan passed 530 checks per mode with matching
stdout and empty stderr. The fixture tests allocation failure, abort, idle
same-owner generator and held-lease exclusion, wrong owner/nonce/binding, prepublication
commit failure, independent live-owner coexistence, held-borrow release
rejection, context-entry exclusion, and 32 create/release cycles with measured
test-only native workspace retained bytes returning to zero each time. The
private Eshkol fixture contains genuine C4
constructor, same-owner/foreign/C2, fault, binding-swap, changed-carrier,
eval-mode and idempotence controls, but it is **UNRUN** as a compiled package.
The Python source assertions and balanced-form check are source checks only.

The native fixture uses sealed seeded owners to exercise the native edge; it
does not establish the Eshkol trainable-C4 package. The prerequisite
constructor/P1/association/context/lease compiled admissions and independent
review of a pinned exact-source workspace recipe remain open. This leaf adds
no numerical forward, VJP, graph, optimizer, SAVE/LOAD or generator resume.
