# P2/G2 committed-prefix identity preflight

Status: source candidate; Eshkol fixture and compiled same-call terminal gate **NOT RUN**.

`g3c4-p2g2-terminal-prefix-committed-check(call, input, operation)` is a
private read-only predicate for a future terminal coordinator. It requires the
original pending output and protected call, the top exact P2 input anchor and
live input/native binding, and matching protected/anchor/ledger committed
phases. It returns `#t` or raises `invalid-state`. It does not decode, sample,
convert a carrier to a pointer, commit output, or expose the pending owner.
The existing before-prefix helper and abort-only scopes are unchanged.

This predicate **does not authenticate the sampled first ID**. Both original
raw and staging payloads remain mutable, and equal paired edits can agree
without matching the retained native first frame. Native terminal validation
must check that relation after an independently authenticated carrier backing
extent; this predicate cannot authorize a second frame or terminal commit.

The exact upstream Eshkol lowering at
`lib/backend/llvm_codegen.cpp:16286-16318` creates bytevectors with an
eight-byte length followed by `len` data bytes and calls
`arena_allocate_with_header(data_size=8+len)`; the allocator at
`lib/core/runtime_object_alloc.cpp:20-61` records that size. This supports a
*possible derived extent proof* for fixed literal allocation sites, bound to
the exact compiler ABI and the original authenticated allocation identities.
The current mutable extent vectors and `bytevector-length` checks do not
establish that proof. No raw payload-minus-header probe is allowed while the
trained-G2 E3 sanitizer failure classification is on hold.

The source fixture covers wrong phase, foreign input, malformed top anchor,
changed live native binding, changed original input anchor and ledger phase,
followed by repair and authenticated abort. It is intentionally uncompiled.
Before a terminal
successor, independently review the exact compiler/allocator provenance,
creation-time original allocation binding and native header pointer contract;
then gate the same-call first/EOS/second paths under normal, repeat and native
sanitizers after E3 classifies the trained-G2 failure.
