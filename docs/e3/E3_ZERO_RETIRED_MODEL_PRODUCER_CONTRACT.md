# E3 zero fixture: retained graph and M3T retirement dependency

**Source contract only.** The wrong-tie construction test is prepublication and
uses the existing abort. It does not make a sealed M3 handle stale. No Eshkol
sealed-graph producer or M3T retire operation is implemented here. The
feature-gated native P1 candidate roster, preflight and revoke in
`native/p1_identity.c` is an upstream candidate, not an Eshkol teardown API.

## Exact source boundary

`m3t-public-model-create(config, state)` in
`native/m3t_transport_extension.esk` creates a native owner, opens an I2/P1
construction, registers fourteen handles plus one tied alias, then seals I2/P1,
the owner and successor initializer. Its guarded `i2-construction-abort-internal!`
and `et_m3t_private_owner_abort_v1(void *)` admit only unsealed construction.
`native/i2_wave2_extension.esk` clears its member/carrier rollback slots on
seal. `internal/p1/lib/transformer/module.esk` clears the construction's
schedule and has no sealed-shell release. Native P1's ordinary seal frees the
roster; only `ET_G3R_CANDIDATE_RETIRE_PRIVATE` retains a candidate roster.
Native `et_m3t_private_owner_seal_v1(void *)` sets owner state 1, while
`et_m3t_private_owner_abort_v1(void *)` explicitly rejects state 1.

The reviewed `G3_R_P1_I2_SEALED_GRAPH_RETIREMENT_CONTRACT.md` proposes the
feature-local Eshkol producer signatures
`(i2-candidate-construction-begin-internal! enrollment-cell)`,
`(module-candidate-construction-begin-internal provider enrollment-cell)`,
`(g3r-candidate-graph-revoke-preflight-internal! ledger o2-witness)` and
`(g3r-candidate-graph-revoke-commit-internal! ledger graph-witness o2-witness)`.
For E3's disposable model with no O2 owner, a separately reviewed fixture
adapter must authenticate that exact no-optimizer condition; passing a fake
`o2-witness` to the G3 contract is forbidden. The existing native signatures
are `et_p1_private_candidate_construction_begin_v1(void *)`,
`et_p1_private_candidate_graph_preflight_v1(void *, void *)`, and
`et_p1_private_candidate_graph_revoke_v1(void *, void *)`, all guarded by
`ET_G3R_CANDIDATE_RETIRE_PRIVATE`. No public setter, FFI shortcut, or ABI
change to the ordinary model is authorized by this document.

## Required producer and consumer edges

Before the first token, an Eshkol fixture ledger must root the model shell,
original and successor initializer, native owner, P1 construction identity and
all future rollback cells. Native P1 must retain its exact scoped module/handle
roster at seal; Eshkol P1 must retain the corresponding shell records and root;
I2 must retain each exact module membership/carrier and prior active count.
Every new token, shell and carrier is enrolled before another fallible step.
The fixed fourteen unique paths, one token/head alias, owner bindings and
native f32 identities are compared by identity rather than copied vector
contents. A post-return adapter cannot reconstruct the discarded ordinary
roster, so this requires a feature-local construction route and closed object,
source, symbol and test inventories.

Retirement preflight admits only one sealed, unpublished fixture model with
its exact ledger and no O2 optimizer, TR3 lease, G3 model lease, active M3T
frame/workspace/input/logits, P1 state/copy plan/provider callback, I2
carrier/builder, E3 anchor/failure slot, f32 borrow/pin, or other graph
consumer. It verifies the whole retained roster, all fourteen owner/handle/
parameter identities, tie alias, initializer ownership, and f32 destroy
readiness **before any mutation**. Foreign/null identities are not availability
queries. A changed ledger or held lease rejects with the first typed error and
leaves the live model usable.

After a stable full recheck, a single-use retained witness may authorize a
nonallocating, noncallback tail: native P1 revokes only its exact sealed roster;
Eshkol P1 invalidates those shells; I2 unlinks exactly its membership/carrier
records; only then may native M3T destroy the fourteen f32 parameters and mark
the owner and fixture-owned successor initializer dead. The caller-owned
original initializer stays live. Token and owner tombstones remain rooted so
stale addresses cannot become authority. An unexpected failure after first
irreversible write is fail-stop, never an ordinary retry or silent partial
release. The native M3T preflight/commit signatures proposed by
`G3_R_FRESH_CANDIDATE_RETIREMENT_CONTRACT.md` remain unimplemented and need a
separate exact ABI/source review for this fixture's no-O2 branch.

Only after that producer exists may a test cache a genuine previously registered
head shell, retire its owning sealed graph, and require `invalid-state` from
`parameter-handle-shape` and `diagnostic-model-profile`. Normal/repeat and
supported sanitizers must prove unchanged failure state, canonical same-process
retry, parameter bytes, gradients, modes, tie, RNG, D2 cursor/destinations,
and measured live/tombstone counts. The present wrong-tie test proves none of
those sealed-retirement behaviors; the stale-M3 row stays OPEN.
