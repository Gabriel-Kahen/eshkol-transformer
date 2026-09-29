# TR3 explicit interruption proposal

Status: proposed for independent review. This document does not change an
installed API or establish implementation evidence. It starts from the reviewed
bounded `trainer-train!` candidate at `304a2a1` (tree `767f982`).

## Runtime and contract boundary

The installed trainer call is synchronous. Eshkol supports first-class
procedures (`tests/features/new_functions_test.esk` at the pinned runtime
`fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`), and the accepted private step
composer already invokes synchronous hooks. The public trainer/leased operands
have no concurrent mutation contract (`PUBLIC_API_CONTRACT.md` section 4).
The pinned hosted runtime owns SIGINT and SIGTERM for process shutdown
(`lib/core/runtime_signals_hosted.cpp` at that runtime commit); its handler
sets a process-wide interrupt/shutdown flag and a second signal can force
exit. Its FFI also documents same-thread use (`docs/api/eshkol_ffi.md`). Therefore
the trainer must not install a competing signal handler, call Scheme from a
signal handler, or require another thread to mutate a trainer or Scheme cell.
Process termination remains recovery from the last durable checkpoint, not a
successful `trainer-train!` interruption.

## Proposed public operation

Keep the existing two-argument `trainer-train!` call, its five numerical
summary fields, and its invocation-delta limits. Add one optional
third argument: `(trainer-train! trainer stop-policy interrupt?)`.
`interrupt?` is a caller-owned zero-argument procedure. A caller that needs an
external request can have this procedure read its own IPC/host flag at a safe
point; the trainer does not create, own, mutate, or serialize that flag. The
procedure returns exactly `#t` to request a normal stop or `#f` to continue.
Validation rejects a non-procedure before any update as `invalid-argument`.
No callback, closure, IPC handle, or pending request enters `trainer-state` or
C2. The caller retains the procedure through the synchronous call; the facade
and private entry must root it until the last poll, then retain nothing. The
existing two-argument boxed C entry and its export stay valid; the optional
form needs a separately audited entry rather than changing that ABI. There is
no global request latch and no request carried between invocations or processes.

The two-argument form never polls a procedure and keeps its current stopping
behavior. The proposed `stop-reason` field below is its only result-schema
change.
The third argument is justified because a synchronous two-argument call offers
the caller no supported way to request an interrupt while it runs; a stop
threshold is a separate policy outcome. The callback is executed on the
calling Eshkol thread only, never within a signal handler or a numerical
microbatch. It receives no mutable trainer handle. Reentrant use of the same
trainer or its leased operands from the callback is unsupported and must be
rejected as `invalid-state` without changing a completed update. External
callback side effects remain caller-owned.

## Boundary and outcome

After each successful full update and its metrics/counter publication, first
check the existing token, update, and epoch limits in their accepted order.
The first reached limit wins; do not call `interrupt?` at that boundary. If no
limit is reached, call `interrupt?` once. `#t` stops before the next batch and
returns a normal immutable train summary for only this invocation's committed
updates. `#f` continues. There is no midstep early exit or extra D2 fetch.
Requests that arrive after a poll are observed at the next eligible committed
boundary. Because stop policies have at least one positive limit, a successful
call always commits at least one update and has a nonzero summary weight.

Propose one additional immutable train-summary field, `stop-reason`, readable
with `metrics-ref`: `max-tokens`, `max-updates`, `max-epochs`, or `interrupt`.
The original five fields retain their exact types, bits, meanings and keys;
step and evaluation maps do not change. A simultaneous limit and external
request reports the limit because the callback is not invoked at that
boundary. A two-argument call reports its limiting reason. This sixth field
is a deliberate schema extension so callers can distinguish a requested
interruption from ordinary finite stopping; it requires explicit review of
the existing exact five-key contract and metrics owner before implementation.

A callback that raises or returns a non-boolean fails `trainer-train!` with
the appropriate public error (`invalid-argument` for a non-boolean result;
otherwise preserve a categorized transformer error and wrap a foreign value
as `internal`). No summary is published. Earlier updates remain committed;
the current step has already committed; no following step starts. The trainer
returns to idle, and a subsequent call can use a new or the same predicate.
The callback is not retried by the trainer. A failed numerical step follows
the accepted prewrite rollback and does not poll the predicate for that step.
An interrupt result clears all per-invocation staging before return. It does
not clear the caller's external request flag; reuse observes whatever value
the caller next provides. `trainer-state` immediately after normal interrupt
is a complete update-boundary state accepted by C2 SAVE/LOAD and joint restore.

## Public acceptance witness

Compose the reviewed train candidate with public state/load (#149), public C2
checkpoint (#151), and fresh-process package (#153) in dependency order.
For A=1,2,3 use the accepted finite D2 two-row fixture and nonconstant O2
linear schedule. An uninterrupted reference makes K+R updates in one
two-argument train call; an independent segmented reference uses a one-update
policy and saves public C2 after each call. A separate producer invokes the
three-argument train with a limit beyond K. After its first committed update,
the callback sends a ready marker over bounded test IPC, waits for the external
driver's explicit request, and returns `#t`. Its summary reports `interrupt`,
K=1 update, and exact f32 loss/weight/token/epoch counts. Save the public state
through C2. A fresh receiver constructs compatible operands, capability-loads
C2, jointly restores, and makes R=3 two-argument train calls under the
one-update policy, saving C2 after each. Compare canonical C2 bytes at K,
immediately after restore, and after each suffix update with the segmented
reference, and compare both final paths with the uninterrupted single-call
reference. Compare train-summary reductions with separately observed public
step metrics. The bytes cover 14 model parameters, 28 O2 moments, schedule,
both D2 cursors, RNG words, tokenizer/X1 identity and counters. Exercise EOS
crossing, unequal active-token weights, limit/interrupt tie precedence,
false/true/reuse, invalid callback result/throw, corrupt file and X1 mismatch,
and prewrite failure then exact retry. Confirm no partial summary publication
on failure and no callback poll during failed or terminal steps.

This fixed trainer has no dropout and rejects nonzero RNG counters. The gate
can prove exact preservation of zero RNG state, not equivalence after RNG
consumption. That requires a separately accepted stochastic training profile.
The implementation gate is a pinned supported linked public caller with an
independent source/test review, followed by exact-head hosted CI and merged
dependency retest. No new checkpoint format or unsupported device is implied.
