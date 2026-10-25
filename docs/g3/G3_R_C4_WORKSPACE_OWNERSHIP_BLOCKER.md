# C4 workspace ownership: missing live P1 topology authority

**Source dependency; no workspace implementation or compiled admission.** This
finding is against the reviewed authority union `3c81838b` and the
[workspace ownership contract](G3_R_C4_WORKSPACE_OWNERSHIP_CONTRACT.md).

The proposed create path can use the existing train-only P1 slot 74 and the
empty-slot C4/native owner association status. A live workspace changes both
conditions: C4 entry slot 10 and native `owner->active` must identify the
workspace, and `module-eval!` may change all P1 module modes before release.
The contract requires release after that mode drift while still proving that
the fourteen original handles retain their exact positions in the current P1
module tree.

The current read-only authorities cannot make that proof:

* `c4-topology-status`, exposed through trusted slot 74, rejects a module
  outside train mode in both its capture and recheck (`module.esk`); the
  existing P1 fixture checks its eval-mode rejection.
* `g3r-c4-trainable-owner-association-status-internal` requires entry slot 10
  empty and calls slot 74 twice. It therefore cannot validate a live workspace
  or an eval-mode release.
* `parameter-handle-tensor-identity-internal` authenticates a live handle and
  returns its carrier. The native per-index binding status proves the supplied
  handle and parameter belong to the sealed owner. Neither reads the handle's
  current parent module, binding position, or the complete 17-module tree.
  A retained handle/carrier snapshot cannot substitute for current topology.

The next upstream authority needs a **read-only, mode-neutral P1 C4 topology
status** for the exact model and fourteen ordered handles. It must retain the
slot-74 checks for the 17-module parent/child graph, all bindings, tied
head/embedding alias, paths, CPU f32 shapes including `[4,4]` position, and
current carrier identities; it may accept a uniformly finalized eval tree for
release without admitting a mixed or mutated tree. Create must continue to
use train-only slot 74. This is a required contract for a subsequent source
leaf, not an API implemented or admitted here.

Until that authority exists, implementing create/reset/release together would
either reject a contractually releasable eval-mode workspace or allow release
without proving unchanged P1 ownership. No native workspace allocation or
edge-publication API has been added in this branch. Existing constructor,
association, context exclusion, lease and union source candidates remain
unchanged. Their individual compiled gates remain prerequisites.
