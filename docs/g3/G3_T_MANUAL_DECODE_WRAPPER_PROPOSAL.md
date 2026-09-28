# G3-T private manual decode wrapper proposal

Status: **accepted for bounded private implementation** after independent
source review. This is a bounded
Eshkol wrapper leaf over the accepted [manual decode transport
contract](G3_T_MANUAL_DECODE_TRANSPORT_PROPOSAL.md) and the independently
reviewed native candidate `a4f7c53e2a7bdf20b12b544014249bf1c0af374a`
(tree `ad87a4e74dde192d633be82304c0694ff9adc14c`). That native commit is
isolated from this document's `origin/main` base; an implementation must first
compose the reviewed source. This proposal adds no implementation, new native
symbol, G3-M operation, public export, seeded generation or repeated append.

## Exact seam

The reviewed native `g3t_transport.c:576-603` already admits kind-1 calls;
`:1473-1562` admits a non-null authentic owned kind-2 input for frame kind 2
after a pending kind-3 logits result and a committed length-one prefix.
It copies one CPU i64 `[1,1]` ID and the prefix K/V snapshot before closing
the borrows. The existing native role, prepare, preflight, commit, abort and
finish branches own the 21 T1 roles, f32 `[1,256]` result, A2 `1→2` append,
and rollback. The Eshkol `g3t_prefill_sample_extension.esk:134-167` currently
admits only kinds 0/2, while `:235-251` permits frame kind 2 only with `#f`.
These two wrapper predicates are the missing consumer gate. The existing
`g3t_manual_p1_extension.esk:6-68` logits reserve, prepare-end and finish
wrappers already have generic manual-result linkage; they need no new route.

Widen `g3t-call-acquire(generator-entry, kind)` to accept exact integer kind
`1` alongside 0/2, preserving its shared `m3-call-state` guard, authenticated
live generator/model, no-active-call checks, 12-cell registry entry and
fourteen-pin native acquisition. Preserve the current pending entry enrollment
before native acquisition, tombstone-on-native-failure, and publication of
generator/model/guard links only after native success. Update the rejection
message to name all three admitted kinds. Do not change
`g3t-with-call`: its hard-coded kind 2 remains the generated-output route.

Widen `g3t-frame-begin(call,input,frame-kind)` so an authenticated live kind-2
input may be passed as a non-null native pointer with frame kind 2. Frame kind
1 still requires a live input; frame kind 2 may also carry `#f` for existing
generation decode. The wrapper must authenticate the call before input lookup
and reject forged, wrong-kind or dead input shells through its existing
registry/liveness checks. Native call kind, pending-result and frame state
remain authoritative: it admits non-null input only for kind-1 manual decode,
and `#f` only for kind-2 generation decode. Do not add a call-entry kind slot,
new owner API or wrapper-side ID/shape copy; native checks the typed I1,
inline-ID agreement, prefix/binding and A2 state. The existing
`g3t-generation-token-input-create` and `g3t-generation-t1-input-create`
produce authentic owned length-one inputs; the latter's T1 source must be a
same-aggregate P1 tensor. Pair-only inline IDs are not a substitute for the
owned I1.

No other wrapper is widened. The kind-1 caller reserves exactly one pending
logits owner, begins one frame, runs ordinals 0..20, then uses existing
`g3t-frame-prepare(call,logits)`, `g3t-call-prepare-end(call)`,
`g3t-frame-commit(call)` and immediately `g3t-call-finish(call)`.
`g3t-role-step` still checks exact ordinal range before native dispatch.
The result detaches at finish and remains live after call/generator close.
No output reservation, sample, draw, token carrier or text publication occurs.

## Failure and ownership boundary

Preserve wrapper error order: shared guard and call/active-link authentication
before kind/input checks; for frame begin, call/model readiness before input
registry identity and wrapper liveness, then native lifecycle/shape/prefix
checks. Wrapper kind rejection is `invalid-argument`/config; dead input or
illegal frame/input combination at the wrapper is `invalid-state`/lifecycle.
The existing `g3t-native-fail-raw` must expose native domain, category and
code without replacing I1, F32, A2 or provider failures with a generic wrapper
error. In particular native distinguishes forged input identity,
dead/wrong-length/busy input, malformed/borrowed typed I1, inline/I1 mismatch,
missing/length-two prefix, stale binding and failed A2 transaction according
to the accepted native contract. Out-of-range role ordinal remains wrapper
argument/config before native frame-state checks.

On failure after successful acquisition, the future caller must abort the
live call and re-raise the original error, as the current G3-M P1/P2 guards
do (`g3m_prefill_p1_extension.esk:13-18`,
`g3m_prefill_p2_extension.esk:11-17`). Existing `g3t-call-abort` clears native
frame/transaction and pending result before scrubbing wrapper call and active
links; native abort can refuse a borrowed pending result before cleanup, so
the borrow must end and abort be retried. Input release after native frame
admission cannot change the copied ID. Precommit failures retain the length-one
prefix, binding, RNG and older detached results. After commit starts there is
no recoverable wrapper action before finish; a failed native abort or finish
is fail-stop, matching existing wrappers. Do not use a generic
`g3t-with-call` cleanup around manual commit.

## Isolation and acceptance gate

The implementation changes only the two private wrapper predicates in
`native/g3t_prefill_sample_extension.esk`, with a focused source contract and
closed source manifest derived from the reviewed native decode closure.
The test aggregate loads these private Eshkol files explicitly; production
native build remains feature-off and its symbol/test-hook inventory unchanged.
Do not install the wrapper through a public package or alter native ABI.

Extend the pinned `scripts/test-g3t-p2-zero-budget.sh` aggregate with genuine
wrapper calls under `ET_G3T_MANUAL_DECODE_PRIVATE`: P1 manual prefix, kind-1
acquire, pending logits reserve, owned length-one input, frame kind 2,
21 roles, prepare/preflight/commit/finish, then verify all 256 logits and
both K/V positions bitwise against the independent M3T two-token reference.
Repeat with a P1/G0 predecessor. Prove input-copy ownership with a native
owner-release fixture after frame admission, as the reviewed native test does;
do not nest the Eshkol tensor-release wrapper inside the active `m3-call`.
check lengths rank-1 `[1]` value 2, mask rank-2 `[1,2]` values `[1,1]`,
unchanged RNG, detached result after parent close and no leaked pins/links.
Reject kind/frame mismatch, `#f` on manual decode, non-null input on generated
decode, forged/dead/wrong-length/borrowed/malformed I1, no/full prefix,
stale binding, borrowed pending result, failed reservation, provider and A2
cuts; show abort/retry and preserve original failure status. Retain P1/P2
manual and P1/G0, P2/G0, P1/G1 regression witnesses. Run the checker, Q0
Python isolation, pinned network-disabled normal/repeat/ASan+UBSan+LSan
aggregate, production feature-off compile/symbol inventory and source-manifest
closure; seal exact commit/tree, hashes and logs. Report synthetic dispatch
cuts separately from actual provider failures.

Only after independent wrapper source/test review may G3-M add a private
manual decode composer on this seam. That later leaf must own the 21-call
schedule and rollback around the existing operations. Public G3-G and a
seeded generation schedule remain separate downstream gates.
