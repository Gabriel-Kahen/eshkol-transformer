# G3-R candidate P1/I2 sealed-graph retirement prerequisite

**Proposed private producer contract; no revocation is implemented.** This is
the graph-authority dependency of [fresh-candidate retirement](G3_R_FRESH_CANDIDATE_RETIREMENT_CONTRACT.md).
It applies only to an unpublished diagnostic-C2 fresh candidate under
`ET_G3R_CANDIDATE_RETIRE_PRIVATE`. It does not change ordinary M3T construction,
feature-off builds, public P1/I2/module APIs, or any caller-owned graph.

## Source authority and the lost roster

`internal/p1/lib/transformer/module.esk` roots a construction record keyed by
the native construction pointer, its provider and root shell
(`:195–253, 4333–4363`). P1 shell registry entries for modules and handles
hold that exact construction identity in field `3` (`:390–481`). Seal checks
the complete parameter schedule, finalizes the module topology and calls
native seal (`:4280–4428`). Native `et_p1_construction` holds the enrolled
module/handle token records while open; `et_p1_private_construction_seal_v1`
currently frees that `entries` roster (`native/p1_identity.c:87–101,
502–643`). Native token records remain live but have no construction back
pointer. A post-seal global-token scan cannot safely identify one candidate.
Construction abort admits only open/prepared construction; P1 context release
rejects live tokens.

`native/i2_wave2_extension.esk:560–728` roots a Scheme construction record,
memberships, carriers and rollback list. Seal marks it sealed and clears the
construction's membership/carrier/rollback slots, while the ordinary I2
module registry and count remain live (`:82–91, 742–768, 894–900`). I2 has
no native graph type. Its provider carriers are **borrowed** by P1 module
bindings (`module.esk:2470–2522`); M3T owns the f32 parameters and destroys
them only after graph authority is revoked. The candidate path must retain
its exact I2 membership identities in a feature-local sidecar before seal.

## Candidate-aware construction handshake

The aggregate registers a fresh candidate ledger in its private registry
**before the first model constructor call**, preallocating identity/cleanup
cells and proving the initializer is a permitted construction input. It calls
only `(g3r-candidate-model-create-internal! ledger config initializer)`, a
feature-local variant of the fixed M3T constructor. The ordinary
`m3t-public-model-create` retains its existing signature and behavior. A
post-return adapter cannot work: that constructor has already sealed P1/I2
and native P1 has discarded its roster.

The candidate constructor enrolls its newly returned native M3T owner in
the rooted ledger before calling an I2 candidate construction begin. That
begin creates/roots its I2 record and calls a private P1 candidate begin,
which uses the existing native P1 construction scope/token kinds and creates
the first module token **only after** recording the exact native construction
identity in its rooted record and the candidate's preallocated cell. Proposed
closed signatures are:

```c
/* Hidden; uses the existing context result_ptr and construction identity. */
int64_t et_p1_private_candidate_construction_begin_v1(void *context);
int64_t et_p1_private_candidate_graph_preflight_v1(
    void *context, void *construction);
int64_t et_p1_private_candidate_graph_revoke_v1(
    void *context, void *construction);
```

```scheme
(g3r-candidate-ledger-begin-internal! config initializer) ; -> ledger
(g3r-candidate-model-create-internal! ledger config initializer) ; -> model
(i2-candidate-construction-begin-internal! enrollment-cell) ; -> I2 record
(module-candidate-construction-begin-internal provider enrollment-cell)
(g3r-candidate-graph-revoke-preflight-internal! ledger o2-witness)
  ; -> graph-witness
(g3r-candidate-graph-revoke-commit-internal! ledger graph-witness o2-witness)
  ; -> #t
```

The P1 candidate begin differs only in an authenticated candidate-retirable
marker on the existing native construction. Its native seal keeps the exact
token roster until graph revocation; ordinary seal still frees it. A feature-
local P1 Scheme sidecar retains the exact scoped shell records and root; an
I2 sidecar retains exact member cells and prior count. Root holders and cleanup
guards precede the first token; each resulting token/carrier is enrolled
before the next fallible operation. Complete rosters are retained before seal.
The model constructor preallocates all retirement sidecars before
the seal boundary. If enrollment, construction or seal preflight fails before
native P1 seal, use existing open/prepared P1/I2/M3T construction abort and
retire the inert ledger. The native P1 seal and subsequent I2/M3T/initializer
seals must have a reviewed non-failing tail; an unexpected failure after
native P1 seal is fail-stop 134, never a retry through construction abort.

## Whole-graph preflight and revocation order

All preflights run while the candidate's **exact O2 optimizer remains live**.
The O2 retirement preflight authenticates its enrolled Scheme/native identity,
14 matching P1 handles/parameter/storage owners, idle moments/operation and
absence of another O2 consumer of this graph. The resulting `o2-witness` is
retained in the candidate ledger. P1/I2 graph preflight admits that exact
optimizer and witness as the sole still-live O2 consumer; a foreign optimizer,
second optimizer on any candidate handle, or changed O2 enrollment rejects.
It also authenticates the sealed native P1 construction, complete retained
module/handle token roster and nonces before dereference, exact P1 shell
scope/root, I2 membership cells, 14 unique paths and one tied alias, fixed
provider, and M3T native owner/handle identity. It rejects a published or
shared graph, active TR3 lease, P1 state/copy-plan or provider callback,
active I2 carrier lease, M3T workspace/pin, or G3-T model lease. Caller
graphs and live independent tokens are excluded by exact provenance, not by
matching vector contents.

The final retirement sequence is O2 optimizer commit, then graph revocation,
then M3T parameter/owner retirement. Graph commit consumes its single-use
preflight witness only after authenticating that **same** O2 optimizer's
native and Scheme tombstones produced by O2 commit; it rejects a merely
missing, foreign or newly substituted optimizer. No other consumer may have
appeared. Native P1 marks only roster tokens dead, releases the retained
roster, and leaves token control tombstones for stale-pointer defense. The
Eshkol tail invalidates only candidate P1 shell records and unlinks only its
I2 membership nodes, adjusting the active count, before M3T f32 destruction.
The commit tail allocates nothing, invokes no provider callback or Scheme
user code, and cannot raise after the first irreversible write. Any invariant
failure there is fail-stop 134; pre-commit failures preserve the first cause,
leave optimizer and graph live, and permit an authenticated retry.

## Producer gate

The first separable implementation gate is **native P1 retained roster**:
candidate-only begin/marker, retained exact `et_p1_record *` enrollment on
seal, registry-first preflight of a sealed candidate, non-failing exact-token
revocation, and live/tombstone/retained-roster census. Use the existing
`et_p1_construction`, `et_p1_record`, context and token kinds; introduce no
native I2 object. Its tests must cover ordinary seal unchanged/default-off
symbol absence, exact versus forged/foreign construction, copied/stale token,
missing/duplicate/changed roster, active-context conflict, allocation
failure before seal, and repeated revoke with live count returning to baseline
and tombstones measured. A native-only gate does **not** prove Eshkol graph
authority or O2 order. Separate source review and a composed normal/repeat/
sanitizer gate must test candidate enrollment cuts, exact root/14/tie parity,
accepted live O2 witness versus foreign O2, held TR3/P1/I2/G3-T lease,
first-error/retry before commit, and fail-stop after it. No full candidate or
replay acceptance follows from this producer leaf alone.
