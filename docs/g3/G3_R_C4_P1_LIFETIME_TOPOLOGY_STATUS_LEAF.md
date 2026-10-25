# C4 P1 lifetime topology status

**Source candidate only.** The C4 workspace ownership contract requires release
after `module-eval!`, while the existing trusted P1 slot 74 intentionally rejects
eval mode. This leaf appends trusted slot 75 and a private wrapper. Slot 74
continues to govern workspace creation and future train-mode computation.

`module-c4-lifetime-topology-status-internal(root, handles)` returns a status
integer. It accepts only the exact finalized 17-module C4 tree in one uniform
mode, train or eval, with the same provider, parent/child graph, bindings, tied
head/token-embedding handle, fourteen ordered live handles and distinct CPU-f32
carriers, exact paths and shapes including the `[4,4]` position table. It
captures current modules/carriers and rechecks their identities before return.
It performs no observation commit, tensor borrow, native call, mode change or
allocation that is published to the caller. A caller receives no module,
handle, carrier or pointer from this status.

This status alone cannot prove a workspace's **original** bindings. A future
workspace live-edge recheck must compare its retained fourteen original
handle/carrier identities with the current P1 tree and exact native-owner
bindings, while authenticating both workspace edges under one serialized
aggregate. Reset still requires slot 74 and train mode; release can use slot 75
after eval mode drift. No workspace stage/commit/release API is introduced here.

The source fixture extends the exact C4 topology test with train/eval/return-to-
train, swapped handles and C2 `[2,4]` position negatives. The source union
includes the independently accepted slot-74 repair and fixes slot 75's same
boolean comparison guard; a focused mutation test pins both forms. The prior
`f61436de` recipe remains sealed and unrun because it predates those repairs.
An exact-union compiled
normal/repeat/sanitizer and feature-off gate is pending. This leaf is not C4
workspace, numerical training, checkpoint or continuation admission.
