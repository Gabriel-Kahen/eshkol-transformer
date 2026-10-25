# C4 P1 lifetime topology status

**Intermediate compiled normal/repeat evidence accepted.** The C4 workspace
ownership contract requires release after `module-eval!`, while the existing
trusted P1 slot 74 intentionally rejects
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
The exact-union normal/repeat receipt
`g3r-c4-p1-lifetime-6d350a9c-attempt2/SHA256SUMS-RUN` (SHA-256
`952f25e4f26b24c45bae6d6a63990ca419bb9b4f38c8e26219e5e54e38408549`)
is independently accepted as intermediate evidence: 65/65 sealed files,
launcher/Docker zero, normal/repeat 17 byte-identical checks, and seeded
authority control 25 checks. The independent result review seal is
`3c8853c29c9a15c08c0b2b5d97c2da1e9abb2a71f801fb6ac4004ad483079e64`.
Exact-source sanitizer and feature-off gates remain pending before full slot-75
admission. This leaf is not C4
workspace, numerical training, checkpoint or continuation admission.
