# Held O2 state adoption into C2: source gate

Status: **contract and causal negative only**. This leaf adds no coordinator,
SAVE pair, Eshkol owner, or file publication. Its native negative preserves
the distinction between a bit-matching O2 state and the exact state issued
under the original-owner hold.

## Observed source boundary

`et_g3r_private_o2_save_state_v1` in `native/o2_optimizer.c` returns a raw
`et_o2_optimizer_state *` while its original optimizer is held. The companion
`et_g3r_private_o2_save_issued_match_v1` checks that a registered live state's
configuration, options, and moment bits match the held optimizer. It does not
record or require the address returned by `save_state`. The focused native
fixture now snapshots another equal state **before** the hold, issues the held
state, and verifies that both distinct pointers pass `issued_match` (340
checks). A future caller must therefore retain the issued address itself.

`o2-public-optimizer-state` in `native/o2_wave2_extension.esk` preallocates an
O2 Scheme shell and enrolls an ordinary native snapshot in `o2-states`. That
snapshot call rejects the busy held optimizer. The separate
`c2_o2_reconstruct_extension.esk` enrolls a native state reconstructed from
checkpoint bytes after its builder commit; it does not accept a held original
optimizer or the pointer returned by held `save_state`. The C2
`c2-training-state-compose-internal` request takes the
registered Scheme O2 state shell, validates its topology/projection, and
claims it with a preallocated owner token. It cannot take a raw native state
pointer. The current `tr3-c-snapshot-compose!` calls the ordinary O2 snapshot,
so it does not bridge this gap. No current Eshkol source invokes the native
G3 and O2 held functions together.

## Minimum next interface

A feature-private Eshkol adoption helper must allocate and root its exact
Scheme shell, state record, registry cons, and native pointer cell **before**
calling the held `save_state` function. The native coordinator must hold the
original G3/M3/A2 and O2 owners, retain the exact returned pointer in its own
registered record, and check pointer identity as well as `issued_match` before
any C2 claim. The helper must put the returned pointer into the preallocated
root cell immediately; enrollment then uses assignments only. It may publish
the Scheme state shell only after the native pointer and logical O2 projection
are checked against the same registered optimizer. Existing ordinary O2
snapshot code is a lifecycle reference, not an authorized held adopter.

On a preclaim exception, the helper releases exactly that rooted native state
once and leaves the original holds live until release is known. If release is
uncertain, it retains rooted authority and fails stop; it must not return a
recoverable error. After C2 composition starts, its request slots and exact
owner token are the transfer ledger. A partial P1/O2 claim must be rolled back
or quarantined with the exact child shell, and no receipt or file I/O can begin
while the original holds remain live. The existing C2 rollback only covers its
own claimed Scheme children; it cannot repair an unregistered raw O2 pointer.
Native `et_o2_optimizer_state_release_v1` may return a provider defect **after**
marking the state DEAD. The adopter must inspect the exact native lifecycle
after a nonzero release: a still-LIVE state retains its root for a controlled
retry, DEAD means its authority was consumed even though the defect is
reported, and RELEASING or an unprovable lifecycle fails stop. The existing
Scheme `o2-public-optimizer-state-release!` distinguishes its wrapped `-1`
and `-2` outcomes; a raw adopter cannot collapse every nonzero status into
"unchanged state." Preserve the first typed error across this inspection.

The first combined source/runtime gate should prove: distinct bit-equal state
rejected by pointer identity, one held issue and one claim, pre-enrollment
exception, post-enrollment exception, O2 release failure, partial C2 claim,
first typed error retention, reverse-order hold release, original model/O2
unchanged on recoverable cuts, and zero publication before all holds drain.
Then the 17-word G3 snapshot and original tokenizer/cursor/trainer controls
can be bound to the detached C2 owner and its exact image receipt. No current
source or test establishes that combined transaction.
