# G3-M source-private P1 prefill orchestration leaf

`(g3m-prefill-p1! generator input)` has arity two and returns a new detached
kind-3 owned CPU f32 `[1,256]` last-logits shell. The generator must be the
authentic live G3-T C2/M3T owner and input an authentic live kind-2 owned CPU
i64 `[1,1]` byte input from the same aggregate. The accepted native
`frame_begin` checks the input's P1 length, byte ID and lifetime; a P2 input
rejects after acquisition and aborts without changing committed state. This
operation neither guesses length from a shell nor applies the generate-only
draw preflight to manual calls.

One outer `m3-call` guard admits the owner and installs rollback before
`g3t-call-acquire` takes fourteen parameter pins. It reserves and roots the
pending result, begins an unpublished P1 candidate, invokes exactly the 21
ordered `g3t-role-step` calls, prepares the result, preflights old-cache and
binding cleanup, then commits cache and detached logits. Native commit is
immediately followed by the accepted no-failure `g3t-call-finish` tail. Every
precommit exception aborts the acquired call and preserves the prior
cache/binding/RNG and any older detached results. An impossible postcommit
exception is fail-stop. No callback, numerical scalar loop, AD graph,
gradient mutation or random draw occurs.

This file adds one private source binding, recorded in
`native/g3m_prefill_p1_local_symbols.txt`. It adds no native API, public
facade, package export, manual P2/decode, or N>1 claim. The aggregate
production-operation witness must compare all 256 logits and committed K/V
with independent M3T reference bits, replace a committed P1 cache, retain
old detached logits, and cover forged/wrong-kind/dead/P2/busy input or
context, I2/A2 allocation and old-cache-borrow rollback, ownership release,
normal/repeat/sanitizer, Q0 and exact source closure. The next contract gate
for broader G3-M is manual P2/decode and cross-operation state transitions.
