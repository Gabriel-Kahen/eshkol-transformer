# G3-R private M3T candidate constructor enrollment

**Proposed bounded producer contract; implementation and runtime admission are
pending.** This is the feature-local M3T constructor dependency named by
[the sealed-graph contract](G3_R_P1_I2_SEALED_GRAPH_RETIREMENT_CONTRACT.md).
It uses the accepted P1/I2 candidate construction handshake. It does not add
a public model constructor, alter `m3t-public-model-create`, release a sealed
model, restore C2, or replay G3. The input initializer remains caller-owned.

## Existing authority and ordering

`m3t-public-model-create` validates one live M3T initializer and the fixed
diagnostic config, stages model and successor entries in `m3t-registry`, then
calls `et_m3t_private_owner_create_v1`. The native owner immediately owns 14
f32 parameters and one pending successor initializer. The ordinary function
then calls ordinary I2 construction, registers the fixed 14 parameter paths
and one tied head alias, initializes each parameter, obtains the successor,
seals I2/P1, and seals the native owner and successor. Existing construction
abort works only before seal. `i2-candidate-construction-begin-internal!`
requires a fresh four-slot I2 cell containing a fresh four-slot P1 cell; it
roots exact membership before P1's first token. Both candidate seal routes
retain exact member identities. These are the only admitted graph inputs.

A post-return adapter cannot enroll the owner before I2 begins or retain
the candidate graph before it seals. The new constructor therefore has its
own feature-local source, loaded only in the private candidate aggregate.
It calls the same fixed M3T helpers and accepted candidate I2/P1 functions.
The ordinary source, package exports, and feature-off native object bytes
stay unchanged.

## Closed interfaces and ledger schema

The only new callable producer names are the already proposed private
interfaces:

```scheme
(g3r-candidate-ledger-begin-internal! config initializer) ; -> opaque ledger shell
(g3r-candidate-model-create-internal! ledger config initializer) ; -> model shell
```

`begin` accepts an exact live M3T initializer, validates the fixed diagnostic
config and seed using `m3t-config-check`, and captures both the config shell
identity and its canonical config/fingerprint. It stages pending M3T model
and successor entries and roots a fresh I2/P1 candidate cell before returning.
Any failure while staging marks already staged entries dead; no native owner
exists yet. The caller receives only an opaque one-slot shell. A private
append-only registry authenticates that shell by `eq?`, not its contents.
The first version of its rooted canonical record has these fixed slots:

| Slot | Retained value |
| --- | --- |
| 0 | exact opaque ledger shell |
| 1 | `fresh`, `building`, `sealed`, or `dead` |
| 2 | exact config shell |
| 3 | canonical config/fingerprint snapshot |
| 4 | exact input initializer entry, borrowed |
| 5 | pending/live M3T model entry |
| 6 | pending/live M3T successor entry |
| 7 | preallocated I2 cell, including its fresh P1 cell |
| 8 | returned native M3T owner pointer, initially false |
| 9 | exact I2 construction record, initially false |
| 10 | exact model root shell, initially false |
| 11 | monotonic seal-entered bit, initially false |

The registry and record are rooted before `model-create` can call native
owner creation. A `fresh` ledger is single-use. The constructor requires the
same config and initializer shells, rechecks their live M3T registry entries
and config contents against slots 3 and 4, and rejects copied, foreign,
tampered, reused, or concurrently active ledgers before native allocation.
No raw owner pointer or I2/P1 cell is accepted as an argument.

## Candidate constructor and failure cut

The constructor prepublishes the model entry's original initializer,
successor, config, 14-slot handle vector and profile exactly as the ordinary
fixed profile does. The first-error guard surrounds native owner creation.
Its lexical `native` local retains the returned opaque pointer even if the
first rooted publication raises: that native call has already allocated 14
f32 parameters and a successor. The constructor stores the pointer in
rooted ledger slot 8 **before** I2 begin or any fallible registration, then
stores it in the M3T model entry. Until slot 8 is published, cleanup uses
the lexical pointer; an abort failure exits 134. Slot 8 is never the sole
recovery route for its own publication failure.
The candidate I2 begin receives only rooted slot 7. If I2 begin succeeds but
publishing ledger slot 9 raises, cleanup recovers the exact construction from
I2 cell slot 1. The preallocated I2 anchor receives owner parameter zero
before the first fallible parameter registration, preserving I2's complete
native-owner abort preflight.

The fixed registration loop uses `m3t-paths`, `m3t-shapes`,
`m3t-child-parent`, `i2-module-register-parameter-internal!`, the native
owner bind/initialize functions, and `m3t-validate-schedule`. It enrolls
each returned handle in the rooted model entry before the next fallible
operation, creates exactly 14 distinct canonical parameters plus the one
head/token-embedding tie, and stores the returned successor pointer in its
rooted M3T entry before I2 seal. No caller callback, alternate topology, or
second construction route is accepted.

Before entering I2 seal, a guard preserves the first construction error: it
aborts the exact candidate graph only when its authenticated I2 cell is open.
An I2 begin self-abort leaves that cell dead; a rejection before I2 begin
leaves it fresh. Neither permits another graph abort. Cleanup then aborts the
still staged native owner and marks model, successor, and ledger dead. A
cleanup defect is fail-stop 134 rather than a retry with an unaccounted
native owner. The caller-owned input initializer is never aborted. A failed
preseal candidate cannot be reused; a new ledger is required for retry.

The constructor uses `i2-candidate-construction-seal-internal!` after the
complete M3T schedule and successor checks. All Eshkol graph retention and
pointer publication precede this call. It sets ledger slot 11 immediately
before entering that seal call. An exception from the call is fail-stop 134,
even if a source preflight happened to reject before native seal: the I2
cell can still say `open` after P1 has sealed natively and before P1/I2 state
publication. No handler may infer rollback authority from that cell once
slot 11 is set. This deliberately narrows recoverable failure cuts to those
before seal entry; a separate authenticated open-state query would be needed
to recoverably handle a seal preflight rejection. Direct native P1 seal
preflights its roster before an assignment-only commit, and the reviewed I2
tail stages the retained list head before calling it. Native M3T owner and
successor seal calls must return success in their already validated fixed
state; an unexpected status or exception after I2/P1 seal exits 134. The
remaining Eshkol tail
writes only immediate `live`/`sealed` lifecycle states and returns the
already rooted model shell. It allocates no list/vector, invokes no callback,
and publishes no new pointer after the irreversible P1 seal.

The sealed ledger retains exact model entry, owner pointer, original and
successor entries, I2 cell/record, root shell, and canonical 14 handles for a
future retirement preflight. This contract grants no retirement authority.
The model shell remains usable through existing same-aggregate M3T APIs;
no ledger or candidate name is provided by ordinary packages.

## Focused gate and limits

Source review must verify preallocation before native owner creation,
lexical native-pointer recovery on rooted-publication failure, pointer
enrollment before candidate I2 begin, recovery from I2 cell slot 1,
complete fixed 14/tie schedule, first-error cleanup, and the postseal tail.
A supported focused compiled normal/repeat gate must construct the candidate,
compare its config, parameter paths, tie and initializer words to the
ordinary fixed-profile result, authenticate its exact sealed P1/I2 sidecars,
exercise wrong/reused/tampered ledgers and constructor failure cuts, and
show unchanged feature-off exports and ordinary I2/M3T object bytes. Fault
injection must measure that preseal native owner/parameter live counts return
to baseline. If an injected I2-begin cut or the existing named-let compiler
warning prevents a valid compiled fixture, record that boundary rather than
claiming the failure cut passed. Independent source/test and recipe review
precede any heavy gate. This producer alone proves no candidate retirement,
O2/TR3 ownership, C2 restore, SAVE, replay, or continuation.
