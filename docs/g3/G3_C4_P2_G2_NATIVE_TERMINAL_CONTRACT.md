# Private P2/G2 native terminal publication contract

Proposed for independent review; unimplemented. This closes the native
dependency after the EOS-capable first-frame carrier and authentic second-frame
carrier. It does not admit public G2 or establish Eshkol T1 provenance. The
second-frame source/tests and supported gate must be accepted before dependent
code is implemented. Existing P2/G1 publication and public adapters stay on
their original route.

## Exact private ABI and terminal forms

`ET_G3C4_P2_G2_TERMINAL_PRIVATE` requires the accepted private prefix, carrier,
EOS first-frame, second-frame, output reservation and token-forward features.
It adds these source-private entries, with no package export:

```c
int64_t et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *pending_output,
    const void *first_raw_header, int64_t first_raw_bytes,
    const void *second_raw_header, int64_t second_raw_bytes);
int64_t et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *published_output, void *destination_header,
    int64_t destination_bytes);
```

The terminal feature does not require `ET_G3C4_P2_G1_PUBLICATION_PRIVATE`.
Current published-state definition, admission and borrow-aware release are
guarded by that predecessor feature; they are not available merely because
output reservation is enabled. This leaf must provide a narrowly
terminal-guarded published-state definition/admission/release branch, including
the same published-state constant where its declaration needs a union guard.
Admission accepts only the distinct authentic terminal marker and exact
capacity/logical-length facts below. Release validates that terminal owner and
its I1 lease before mutation. When both features are enabled, their branches
remain disjoint; the P2/G1 path still rejects capacity two and cannot authorize
a terminal owner. This is an explicit native prerequisite, not an assumption
that the predecessor release code is compiled or already admits this output.

Commit derives the emitted count from authentic native state; there is no
caller count, token, logits, raw RNG or route selector. Both forms require the
original active `(kind,budget)=(2,2)` generator call and exact sole unpublished
capacity-two output, with unchanged model pins and prefill binding:

- First EOS has no committed prefix, a genuine READY position-2 frame and
  selected ID equal to the explicit policy EOS. It emits one token, commits
  cache `3/1110`, and advances RNG by the one real categorical draw or not at
  all for greedy. A non-EOS first frame cannot end the budget-two request.
- A second READY position-3 frame has the authentic non-EOS committed prefix
  and second-frame route marker. It emits two tokens, including a selected
  second EOS, commits cache `4/1111` and advances RNG only by the genuine second
  draw. No first-token output is independently live before this tail.

EOS is included in IDs, raw text, emitted length and cache, as required by
the generation proposal. There is no synthetic budget-exhaustion token.

The first raw carrier is an aligned length header declaring one payload byte,
with exact total extent nine bytes. A second-frame request requires a distinct
second carrier of the same extent. First EOS requires a null second pointer
and zero second extent; a caller cannot smuggle a second token into this form.
Each payload byte must equal its authentic selected V256 ID. Validate range,
header, alignment, pairwise disjointness and overlap with all native-owned
storage before reading. As for the accepted carrier bridge, arbitrary native
pointers cannot prove backing allocation extent or T1 identity: the protected
Eshkol caller must validate original bytevector identity, allocator extent and
same-registry T1 decode before entering this ABI.

## Capacity, logical length and fallible preflight

Keep the requested native `generated_length` capacity two and the physical
dense CPU-i64 `[2]`/16-byte output allocation. Record terminal ownership with a
distinct feature-gated native marker, and set `length` to the derived logical
count one or two only in the terminal tail. For first EOS the second physical
ID remains zero. Native output validation must recognize only this exact
capacity/logical-length relationship; no arbitrary short result or changed
requested budget is admitted. Feature-off state/source and symbols must remain
equivalent to the immutable predecessor.

Before any terminal mutation, authenticate context, output, policy, prefix,
sampled ID and all four successor RNG words. Verify the current committed
cache is the expected P2 or P3 predecessor, and the genuine view-free A2
candidate has exact staged K/V, length and mask already authenticated by the
forward route. Compare candidate bytes to the retained frame arrays, including
the prefix rows. Recheck model identities, parameter bits, prefill binding and
exact output parent/ledger/native registry edge after all recoverable calls.

The pending output is unready: numeric, ID-copy and text flags, length and
cache length are zero. Its original I1 is unborrowed, authentic dense `[2]`,
aligned and zero in both slots. A held I1 or staged A2 view rejects before
mutation, with the same owners and links live; releasing the genuine lease
permits retry. Forged, stale or cross-owner handles never authorize release.
Balance every preflight borrow on failure while preserving the first error.

Preflight both raw carriers and construct the two-i64 ID payload in native
scratch. The actual I1 `copy_from_v1` validates exact shape, source extent,
alias and absence of active borrow, then copies without allocation. After
complete preflight its write belongs inside the no-failure tail; an impossible
failure must fail-stop, not return a partially prepared result. There is no
allocation, T1 decode, provider dispatch, recoverable borrow or output shell
construction in the tail.

## Closed publication tail and release

All recoverable checks precede the first ID write. The closed native tail:

1. Writes the two physical IDs from native scratch, with unused G1 slot zero.
2. Commits the authenticated A2 transaction and copies its successor RNG into
   the generator and detached output facts.
3. Sets exact logical output length, cache length and readiness facts; clears
   speculative frame and prefix continuation; detaches and publishes the sole
   native output under the distinct terminal marker.
4. Drains the original model pins, native owner active edge and call state to
   idle without a recoverable return between publication and drain.

Any unexpected failure after step one is fail-stop. Successful terminal commit
has no pending child, live frame, pin or active native generator call. Existing
`call_finish`, P2/G1 frame publication/copyout and public result validators are
not widened to authorize this new owner. A second call to terminal commit or
abort cannot rewind or republish the output. Existing exact-owner output
release must work for the terminal owner, remain borrow-aware, scrub its
storage/RNG and support idempotent repeated release; it must not release
another output through a forged ledger or handle.

The future Eshkol composer must preallocate both logical G1/G2 shell and text
representations before prefill, capture original identities, and preflight its
closed ledger/shell tail before native commit. It then performs only trusted
ledger/shell writes and region completion after the native tail. Recoverable
cleanup or validation between native publication and Eshkol publication is
forbidden. This native leaf alone does not implement that composer.

## Detached snapshot and logical clones

Snapshot admits only an independently published terminal-marked output with
no parent. It authenticates original I1 storage, capacity two, exact logical
count/cache/RNG/readiness and unused zero slot for G1. The destination has an
aligned eight-byte header declaring 64 payload bytes and exact total extent
72, with full range/native-storage alias checks. Build private scratch, release
the genuine source borrow, reauthenticate owner/header, then write only the
64 payload bytes once. Errors leave destination bytes unchanged.

The snapshot is eight little-endian signed-i64 words: first ID, second ID
(zero if logical G1), emitted count, cache length, and four final RNG words.
It is a private copied descriptor, not a tensor, text owner, arbitrary-pointer
allocation proof or serialized continuation record. The existing scalar
P2/G1 detached-result factory cannot produce a two-ID tensor; it remains
unchanged. A separately reviewed Eshkol/live-result adapter must return only
logical `[1]` or `[2]` detached ID clones and exact raw text, never the unused
capacity slot, and must obtain raw bytes from the authenticated T1 carriers.
The snapshot does not replace that adapter or manufacture T1 provenance.

## Required proof and remaining dependencies

Use authentic P2 prefill, real first sampling/forward and either first EOS or
non-EOS prefix followed by genuine retained-logit second sampling/forward.
Cover two distinct prompts and greedy/nontrivial categorical policies; compare
both IDs, raw bytes and four RNG words against an independent G3-S/Philox
reference, and final K/V/logits against real P3/T4 full-prefix forwards.
Include first/second EOS, no EOS, singleton, carry, final available block and
exhausted second draw without an extra draw or partial output. Prove unchanged
parameter/gradient bits and metadata, original binding and correct masks.

Test wrong order, non-EOS early terminal, malformed/aliased/substituted
carriers, changed bytes, fake IDs, policy/RNG/binding/pins/candidate K/V
tampering, wrong/dead/foreign output, every allocation/preflight/borrow cut,
held I1/A2 lease rejection/release/retry, precommit abort and exact release.
After recoverable failures, first-EOS abort preserves P2/original RNG; second
abort preserves the committed P3/first-successor RNG and publishes nothing.
Inject impossible ID-copy, A2-commit and drain failures after the boundary and
prove SIGABRT. Snapshot negatives must prove destination byte identity.

Run pinned native normal/repeat and ASan+UBSan+LSan, independent references,
exact private-symbol/macro closure, feature-off preprocessing and relevant
EOS/second-frame/prefix/P2G1 predecessors. No differentiable operation is added;
gradient preservation is required, not a new gradient formula. Supported
native acceptance still leaves same-registry Eshkol terminal composition,
logical detached result adapters, public/larger-profile admission, repeated
decode and data-only continuation/save/reload/replay acceptance unfinished.
