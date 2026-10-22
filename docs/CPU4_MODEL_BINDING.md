# Canonical CPU4 model binding

This private producer binds one real C4 root, its native owner and fourteen
P1/I2 handles/carriers to an X1 context-four CPU-f32 configuration. It preserves
C4 entry slot7 and checks all eight native initializer words, including original
seed and successor counter292. It creates one typed training lease after all
configuration/P1 checks. A durable registered binding prevents rebinding that
model to another configuration, including after successful lease release.

P1 slot74 contains the closed CPU4 capture/recheck pair. Existing slots0..73
retain their positions; slots69/70 still require position[2,4]. CPU4 requires
position[4,4], the same seventeen finalized train-mode modules, fourteen distinct
parameters, fifteen paths and exact head/token tie. Existing mode slots64..68
save/enter/restore all seventeen modes; the binding checker runs before eval
entry and after real mode restoration, after native pins drain. It permits a
matching READ lease with a live workspace; it does not publish evaluator output.

`cpu4-model-binding-create-internal(model,resolved)` returns a registered token.
The check/native-lease/model accessors and release operation authenticate that
exact token. Release destroys no parameters. P1 carriers/native lease selectors
prove the same native parameters and handle identities; no M3T owner alias is
created. Methods are direct fixed-arity extern calls, never captured bare FFIs.

D2 cursor helpers require a genuine idle N1/T4/V256 dataset with at least68 bytes
of physical batch capacity. They capture/check an encoded cursor identity and
reuse checked D2 seek for restoration. This seek is an allocating fallible
operation, not E3's prepared no-fail rollback tail.

Host source tests exercise exact P1 shape wrappers and binding publication/error
paths with explicit native doubles, then actual SHA256/D2 cursor semantics.
The separate genuine Eshkol fixture checks actual P1/I2/native pointer identity,
all seventeen modes and a real D2 cursor. Neither proves a complete CPU4 trainer,
optimizer/control snapshot, joint checkpoint restore or public generation.
