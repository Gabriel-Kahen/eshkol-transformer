# G3-R fresh-candidate retirement prerequisite

**Proposed private contract, not an implemented release or restore claim.** This
leaf applies only to the fixed diagnostic-C2, CPU-f32, batch-one, raw-V256,
context-two fresh candidate in
[G3-R restore prerequisites](G3_R_RESTORE_PREREQUISITES_CONTRACT.md#fresh-candidate-recipe-and-publication).
It adds no public API, checkpoint field, feature-off symbol, general model
destructor, or authority to retire a caller's existing receiver. Fixed CLI3
construction is an admissible recipe only after its exact configuration,
14-parameter schema and tied alias are checked against the authenticated C2
inputs; it does not broaden Wave 3 support.

## Existing ownership and the gap

`m3t-public-model-create` in `native/m3t_transport_extension.esk` seals an
I2/P1 module graph, a native M3T owner with 14 bound f32 parameters, and its
successor initializer. Its input initializer is separately owned and may have
been supplied by a caller. The native owner keeps both initializer pointers,
14 parameter pointers and handles, and an active-workspace link
(`src/eshkol_transformer/m3t_transport.c`). The sealed successor can be
returned by `m3t-public-model-initializer`. Native construction abort rejects
a sealed owner or initializer; it is **not** a live-owner destructor. Workspace,
input and logits release do not release the model. The same aggregate must
retain the exact candidate ledger edges for the model, its newly created
initializer (if any), successor, P1/I2 graph, and every dependent child.

`o2-public-optimizer-create` in `native/o2_wave2_extension.esk` enrolls an
optimizer shell/record and the native receiver created by
`et_o2_optimizer_builder_finish_v1`. The native O2 receiver owns its entry
array and two f32 moment tensors per entry; its entries refer to M3T-owned
parameters, P1 handles and canonical storage. O2's state-release API releases
detached state, **not** this live optimizer. `tr3-lease-unenroll-internal!`
only unlinks an idle trainer lease; its five component receivers remain live.
An Eshkol region cannot reclaim these native `calloc` allocations.

The authenticated optimizer configuration already has a private route: with
an active C2 borrow, obtain its owner token and source O2 state, call
`o2-require-c2-owned-state`, and copy canonical state field `2` before ending
the borrow. This contract requires comparison with the candidate's fixed CLI3
options; it requests no new public getter.

The existing `et_f32_parameter_destroy_v1` releases value/gradient tensor
data, shapes and strides after idle/pin checks, but retains retired parameter
and tensor control shells for stale-identity defense. M3T owner and
initializer records likewise use an address registry and must remain inert
instead of being freed and reused as authority. O2's live optimizer currently
has no retire operation. Its moments and entry array are reclaimable only
after a new full preflight proves they are idle; the optimizer control shell
must remain a dead registry tombstone. P1/I2 sealed graph retirement is also
missing: construction abort and `et_p1_private_context_release_v1` cannot
substitute while live tokens remain. A private graph revocation seam is a
prerequisite to destroying the parameters referenced by those handles.

## Private closed interfaces and admission

Under a new feature-local `ET_G3R_CANDIDATE_RETIRE_PRIVATE` build flag, the
same E1/P1/I2/M3T/O2/TR3 root may add **only hidden/private** preflight and
retirement entry points. The source implementation must pin exact signatures
and one closed linker/FFI inventory before a supported gate. The proposed
native boundary is status-only; pointers are opaque identities, never
caller-supplied object layouts:

```c
int32_t et_o2_private_candidate_retire_preflight_v1(
    et_o2_optimizer *optimizer, et_o2_error_v1 *error);
int32_t et_o2_private_candidate_retire_commit_v1(
    et_o2_optimizer *optimizer, et_o2_error_v1 *error);
int64_t et_m3t_private_candidate_retire_preflight_v1(
    void *owner, void *exact_original, void *exact_successor);
int64_t et_m3t_private_candidate_retire_commit_v1(void *owner);
int64_t et_m3t_private_candidate_initializer_retire_preflight_v1(
    void *initializer);
int64_t et_m3t_private_candidate_initializer_retire_commit_v1(
    void *initializer);
```

The corresponding closed Eshkol interfaces are
`(g3r-candidate-retire-preflight-internal! ledger) -> witness` and
`(g3r-candidate-retire-internal! ledger witness) -> #t`. The second consumes
the exact witness; it cannot accept arbitrary component pointers. The
[candidate P1/I2 sealed-graph contract](G3_R_P1_I2_SEALED_GRAPH_RETIREMENT_CONTRACT.md)
defines the required producer handshake and private graph preflight/commit
seam. Graph preflight admits the exact candidate O2 optimizer **while live**
with its retirement witness; graph commit requires that same optimizer's
authenticated tombstone after O2 commit. No existing P1 release is represented
as that seam. The
behavioral admission rules are:

* `g3r-candidate-retire-preflight-internal!` accepts only its registered,
  unpublished candidate ledger and exact canonical model, optimizer,
  initializer and trainer identities. It obtains native read-only M3T and O2
  preflight statuses plus P1/I2 graph revocation readiness; it returns a
  private retained witness, never a raw owner or caller callback. An absent
  ledger, forged/copied shell, foreign native pointer, stale handle, changed
  14-path/tied-alias graph rejects before mutation. A caller-owned input
  initializer borrowed for construction is not in the retirement set. An
  initializer claimed as candidate-owned but also shared or bound externally
  rejects before mutation.
* Native `et_o2_private_candidate_retire_preflight_v1` and
  `et_m3t_private_candidate_retire_preflight_v1` authenticate pointer identity
  by their own registries **before dereference**, prove live kind/provider and
  exact ledger-associated native owner, and inspect every dependent payload.
  Their corresponding `*_retire_commit_v1` functions are feature-gated,
  allocation-free, callback-free and non-failing after the full stable
  preflight. They are not exported through public package facades. The O2
  side verifies `busy == 0`, no active operation, all moment tensors unborrowed
  and unpinned, valid count/provider/entries and exact 14 source P1 handles.
  The M3T side verifies sealed owner, `active == NULL`, all 14 exact native
  bindings, live f32 parameters, no parameter/value/gradient borrow or plan
  pin, no live workspace/result lease referencing it, and authentic original
  and successor initializer identity. The Eshkol root separately compares
  O2's enrolled 14 handles with M3T's canonical 14 and their native
  parameter/storage identities; native O2 validates its own cached handle,
  parameter and storage identities. `et_m3t_f32_parameter_preflight_internal`
  is useful for the f32 part; the construction-only M3T/I2 preflight is not a
  live-owner proof. Neither side may treat an arbitrary NULL/foreign pointer
  as an availability query.
* A private P1/I2 sealed-graph preflight and revocation commit must prove the
  exact root, all 14 unique handles and one tied alias, no active state/copy
  plan/provider callback lease, and no other live graph using the parameters.
  It admits only the candidate's exact live O2 optimizer and its preflight
  witness; all foreign or additional consumers reject.
  It revokes references before native M3T parameter destruction, without
  touching the caller's graph. Its native token and Eshkol module enrollment
  identities must be checked independently; a Scheme vector shape check alone
  is insufficient. This is a required upstream implementation seam, not an
  existing API claim.

The aggregate first preflights closure of every candidate child lease, ends
any TR3 restore/optimizer-state/C2 metadata borrow, and unenrolls its idle
trainer. It closes candidate workspace, output and input leases through their
own authenticated release paths. The model and optimizer remain live if a
child release fails. It then performs **every** Eshkol, P1/I2, O2, M3T, f32
and initializer preflight before the first destructive owner-retirement write,
including proof
that no G3-T generator/frame pins this model and that no published receiver
or external shell aliases the candidate. A ledger-owned initializer may be
retired only if its exclusive ownership is authenticated. A caller-owned
initializer merely borrowed by the candidate remains untouched, while a
candidate-owned initializer with any external binding rejects preflight. The
unpublished candidate graph must have no live external binding. The witness
is single-use and invalidated by any intervening enrollment/lease change.

After stable final recheck, commit order is: retire the O2 optimizer's moment
payloads and entries and tombstone its native/Scheme identity; revoke the
candidate's sealed P1/I2 graph and handles; retire the 14 M3T parameters and
their f32 payloads; then tombstone the M3T owner, its successor initializer,
and only a ledger-owned input initializer. This order keeps parameter
identities alive while O2 is retired. Native and Eshkol registries must both
change state exactly once. Repeated release of the exact already-retired
candidate may be idempotent; a forged or cross-candidate identity must reject.
No second LOAD, fresh unrelated owner or construction abort may be used to
make this path appear successful.

## Failure, accounting and acceptance

Before owner-retirement commit, any provider, authentication, allocation or
injected failure leaves the model and optimizer live for a retry and preserves
the **first**
failure. Cleanup faults are reported separately; they must not overwrite the
cause. Once the first irreversible commit runs, an unexpected invariant or
destructor defect is fail-stop (exit 134) rather than returning a partial
candidate as retryable or publishing it. Commit may not invoke fallible
providers, allocate, raise into Scheme or rely on region collection. A
failure during construction before sealing continues to use existing
construction aborts; a sealed candidate uses only this retirement protocol.

Acceptance must count live native M3T parameter/value/gradient payload bytes,
O2 moment/entry bytes, active borrows/pins and live owner entries before and
after repeated failed candidates. Those live counts must return to baseline;
retained M3T/O2/f32/P1 identity tombstones, Scheme registry nodes and arena
bytes must be reported separately and truthfully, with a per-attempt bound.
Do not report the total process footprint as zero or assert leak-free release
from a flat live-payload count. Normal, repeat and ASan/UBSan/LSan tests must
cover exact candidate teardown after restore and each construction cut;
foreign/stale/copied identities; caller-owned borrowed initializer survival;
falsely candidate-owned but externally shared initializer rejection; a
retained model initializer; bound TR3
trainer; active O2 operation; held f32/P1/I2 borrow or pin; live M3T workspace
and G3-T model lease; moment/provider mismatch; and fault injection before
and after the first commit. Preflight failures must preserve the candidate for
successful retry, never alter the caller's receiver, and show unchanged live
payload census. Feature-off build and public symbol inventory must be bytewise
unchanged.

This leaf is sufficient only for **pre-replay** candidate retirement. Current
`g3t_transport.h` exposes release for tensor, output and imported RNG owners,
but no live generator release. Full replay rollback needs a separately
reviewed authentic generator/cache retirement contract and gate before any
claim that failed replay can free every native candidate child. The future
recipe also needs a single composed identity root and supported end-to-end
tests; this document does not implement either.
