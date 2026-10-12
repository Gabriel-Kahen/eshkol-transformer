# G3-R authenticated candidate retirement ledger and commit order

**Design only.** This private `ET_G3R_CANDIDATE_RETIRE_PRIVATE` plan covers one
unpublished, fixed diagnostic-C2 CPU-f32 fresh candidate before G3-T replay.
The [fresh-candidate contract](G3_R_FRESH_CANDIDATE_RETIREMENT_CONTRACT.md)
and [P1/I2 graph contract](G3_R_P1_I2_SEALED_GRAPH_RETIREMENT_CONTRACT.md)
govern its producer. Receipt-to-TR3 same-reservation ownership is a separate
gate; a receipt, C2 borrow or matching contents do not establish ledger
authority.

## Current source and missing producers

At `779fddd`, native O2 has feature-gated pointer-only retirement preflight
and commit for one 14-entry optimizer (`native/o2_optimizer.c`); native P1
retains and can revoke one exact candidate construction roster
(`native/p1_identity.c`); f32 has read-only tensor and parameter
destroy-readiness checks (`native/f32_tensor.c`). The parameter check in this
tree has host normal/repeat/sanitizer evidence, with supported pinned and
same-aggregate admission still pending. Ordinary `m3t-public-model-create`
seals P1/I2 before returning; ordinary O2 create publishes its own shell.
The tree has no candidate-aware M3T/I2/P1 constructor union, Eshkol O2
retirement, sealed I2/P1 sidecar revocation, live M3T owner/initializer
retirement, or composed ledger gate. The separately proposed construction
enrollment authentication on `codex/g3r-candidate-construction-enrollment`
must be source-admitted before its lexical holder can be used. A post-seal
token scan and P1's native-only roster are insufficient.
`g3t-generator-close!` and its native close already exist for an authentic
idle generator (`native/g3t_generator_constructor_extension.esk`,
`src/eshkol_transformer/g3t_transport.c`); an active or pinned generator
rejects close. The fresh-candidate contract's statement that no live G3-T
release exists is stale on this tree. Composed post-replay cleanup remains
unproved.

## Authority and lifecycle

The aggregate must create one source-private row in a fixed lexical holder
before native owner construction. Its fixed original references are the
ledger, resolved config, input initializer and ownership flag, preallocated
M3T owner/P1/I2 cells and sidecars, and provider identity. Write-once
enrollments bind the exact native owner, successor initializer, sealed P1
construction and roster, I2 memberships/count, 14 ordered paths, 14 unique
handles/native parameters/storage owners, one tied alias to handle 10, O2
Scheme shell/native optimizer, and idle TR3 identity. The row roots every
candidate child lease and later preflight witness. Authenticate exact row and
registry identities plus independent native registry admission; vector
contents, SHA bytes, raw pointers and caller-held shells are not authority.
The caller-owned input initializer is borrowed and never retired. An owned
input initializer needs proof of no external binding. An externally held
successor alias blocks retirement.

The construction path is one-use `issued -> owner-enrolled -> armed ->
consumed -> sealed`, with the owner enrolled before the first P1 token.
Before native P1 seal, failure aborts the exact open/prepared P1/I2/M3T
construction and marks the row `aborted`. After seal, construction abort is
invalid. Retirement is `sealed-live -> preflighted -> committing -> retired`.
A precommit failure returns to `sealed-live`, preserves the first cause, and
leaves model and optimizer live for retry. Changed enrollment or lease state
invalidates the witness. Retain a retired row tombstone against copied/stale
ledgers. This row is an in-memory authority record, not an artifact: it has
no wire version or checksum. C2 and G3RCV1 retain their own versioned,
checksummed formats; serialize no ledger bytes into either.

## Nonmutating owner preflight

1. Authenticate an unpublished sealed row and reconcile its original cells
   with separate native registries, P1 root/shell records, I2 sidecar, M3T
   graph and fixed configuration. Prove 14 path/owner identities and the tied
   alias. Reject foreign, copied, stale or shared identities before native
   dereference.
2. End ledger-owned C2 inspection and O2-state/metadata borrows, close
   workspace/input/output leases, and unenroll an idle TR3 lease through
   their authenticated release paths. These fallible child operations precede
   owner commit. Prove absence of other TR3, P1 state/copy-plan, I2 carrier,
   M3T workspace/pin, active O2 operation, provider callback and G3-T model
   lease, and of published or external graph bindings. A child-release error
   leaves model and optimizer live.
3. Preflight the exact O2 14-entry roster and 28 distinct unborrowed moments;
   compare each entry with its enrolled P1 handle, M3T parameter and
   canonical storage. Bind both O2 identities in a proposed Eshkol witness.
   **While that optimizer is live**, preflight P1/I2 admitting only it as a
   consumer. Preflight the M3T owner, each exact-slot f32 parameter, successor
   and exclusively owned input initializer. Each producer admits its own
   registry identity before dereference. Missing O2 or null is not readiness.
4. Retain private single-use witnesses, then recheck the whole closed set,
   exact live O2 enrollment and lease state immediately before commit.
   Preallocate roots/error storage. No provider call, allocation, Scheme user
   callback or new borrow may occur after this boundary.

## Irreversible order and failure state

The first owner write enters `committing`. Commit **O2**: destroy 28 moment
payloads, free its entries and tombstone native and Eshkol optimizer
identities. Commit **P1/I2** only after proving that same optimizer's native
and Scheme tombstones: revoke only its retained P1 roster, invalidate its
shells and unlink its I2 memberships/count. Commit **M3T**: destroy 14
parameters and f32 value/gradient payloads, tombstone the owner and successor,
then retire only an exclusively ledger-owned input initializer. Mark the exact
row `retired`. O2's current pointer-only commit re-preflights itself, but
cannot authorize this aggregate order without all other witnesses.

The tail must be allocation-free and non-failing. An unexpected defect or
injected fault after the first irreversible write exits 134 and never
publishes a candidate. If an earlier cleanup result is indeterminate,
quarantine the row, preserve the first cause and report cleanup faults
separately; do not report retryability or run a second LOAD. Exact
already-retired release may be idempotent only for the same authenticated
row. A foreign or cross-candidate release rejects. Native control shells,
P1 token tombstones, Scheme registry nodes and arena bytes remain measurable
retention; live O2 entries/moments and M3T/f32 payload bytes must return to
baseline.

## Executable gate for the later producer

Build one noninstalled feature-on aggregate with closed FFI/symbol inventory
and exact feature-off object/public-export comparison. Exercise construction
cuts before the first token, between native P1 begin and rooted enrollment,
and before seal. An injected fault after successful native P1 seal must exit
134 in a subprocess without publication; it must never use construction
abort or offer retry. Reject copied/foreign/stale row, P1 cell/token,
optimizer, native owner and initializer; substituted parameter/storage,
wrong provider, missing/duplicate path, wrong tie, foreign/second O2 and
published/shared graph. Each negative must preserve independent candidates,
the caller's initializer, first error and live payload census.

Hold each C2/O2/TR3/P1/I2/M3T borrow, plan, workspace or operation:
preflight rejects without owner mutation and succeeds after authentic
release. A held G3-T model lease must reject without mutation; after
authentic idle `g3t-generator-close!`, retry may proceed. Busy or pinned
generator close must fail without altering the candidate. Inject each fallible
child/preflight cut and prove first-error and retry. Inject a fault after the
first commit write and assert exit 134 with
no publication. Repeat successful retirement, checking live payload
baselines separately from bounded tombstones/arena bytes. Run normal,
repeat, ASan/UBSan/LSan and the supported pinned aggregate gate. Native-only
P1/O2/f32 gates do not suffice. Pre-replay retirement does not establish
post-replay generator/cache cleanup or continuation acceptance; test that
composition separately.
