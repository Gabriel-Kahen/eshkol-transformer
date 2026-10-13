# G3-R same-owner O2 issued-state to C2 owner binding

**Contract only; integration HOLD.** This design starts from merged main
`86f4eb36` and the independently accepted private O2 hold at `073e9be7`
(tree `df59fc33`; review
`/home/gabe/.codex/evidence/eshkol-transformer/o2-073e-independent-review-20261013-a/REVIEW.md`,
SHA-256 `3c6d38dd0a67dbc7b1034dc0d274e77f7f624cf7632e513cb01fd34a9e4808c4`).
The accepted current-history F32/M3/G3/A2 source leaves
are on `codex/g3r-current-history-held-alias-repair`, not main. Neither
branch has the joint coordinator or an O2-to-C2 binding. The current-main O2
union review (`03f81e906e9437ec1d00d12f4f3601168f88981b0ac046137e1b2b3fbb7f46ea`)
therefore keeps SAVE integration on HOLD. This document chooses no new ABI,
public operation, or file format.

## Existing authority and missing transfer

The O2 private sequence is `et_g3r_private_o2_save_begin_v1`,
`save_state_v1`, `save_issued_match_v1`, `save_check_v1`, and `save_end_v1`.
It authenticates the original optimizer against fourteen freshly supplied
parameter owners/identities, reserves it, and issues one native state with
twenty-eight exact moment clones. A pointer to that state is not a registered
Eshkol O2 owner. On main, `o2-public-optimizer-state` creates a *different*
native snapshot while registering its Eshkol wrapper; using it here would
break issued-state provenance. `c2-training-state-compose-internal` accepts
registered P1 and O2 owners, authenticates their path/shape/alias topology and
completed-update count, then claims both into one C2 owner. The accepted C2
SAVE borrows that owner, measures, encodes and fully validates an image, ends
its borrow, and only then invokes the atomic file writer.

A bounded **private issued-state adoption** is required between O2 issue and
C2 composition. It must register exactly the pointer returned by this hold's
`save_state_v1` as one Eshkol O2 state owner, using copied configuration,
canonical paths, aliases and update count from the *same registered original*
Eshkol optimizer. It must prove that wrapper's native pointer, optimizer
registry identity and M3/P1 fourteen-owner roster match the active O2 hold;
equal bytes, equal seeds, caller-provided paths, a second O2 snapshot, or a
foreign wrapper do not suffice. Preallocate the wrapper and its cleanup cell
before native issue where possible. Record native issue and Eshkol adoption
as separate ownership transitions. Before C2 claim, the coordinator alone
releases a failed issued state; after a successful C2 claim, C2 alone releases
it. An uncertain transfer must retain rooted authority and fail stop.

The P1 component must be cloned from the exact M3 model bound to the held G3
generator. Its fourteen primary values and tied alias must correspond to the
sealed M3 roster; F32 witnesses compare original value and absent-gradient
bits before and after transfer. C2 controls come from an authenticated
trainer-control issuer, including tokenizer fingerprint, D2 cursors, X1
configuration and counters. The C2 RNG controls are distinct from the four
G3 generation RNG words in the held 17-word history image. Match the original
raw-V256 tokenizer fingerprint to C2 controls; the held history image itself
contains no tokenizer authority. The exact trainer-control issuer and Eshkol
adoption API remain design dependencies to review before implementation.

## Transaction and publication order

1. Admit the idle committed generator, sealed M3 model, original optimizer,
   tokenizer and trainer controls in one registry universe. Allocate root
   cells and cleanup authority before acquiring leases. Derive the fourteen
   handles/identities from the M3 roster, not caller bytes.
2. Acquire the G3/A2 hold on that model, then F32 original-handle witnesses,
   then the O2 hold using the same roster. This is the proposed fixed order;
   the compiled union must prove coexistence and reverse-order draining.
   Clone P1 from that model; issue exactly one O2 state; privately adopt that
   exact issued pointer. Recheck G3's seventeen words, F32 witnesses, O2 hold
   and `save_issued_match_v1` around each clone/adoption boundary.
3. Compose the P1 owner and adopted O2 owner with the existing exact six-slot
   C2 request. Keep the request and result rooted across exceptions. Success
   clears both component slots and publishes one C2 shell. Check the original
   G3/F32/O2 authority and issued match once more before closing leases. A
   preclaim failure drains only coordinator-owned children; after claim C2
   release owns both. A partial or uncertain O2 claim must not be treated as
   a recoverable rollback without a compiled proof.
4. Copy the checked 136-byte G3 image while the hold is live. End O2 hold,
   F32 witnesses and G3/A2 hold exactly once, preserving the first typed
   source error; cleanup uncertainty fails stop. Only after all original
   leases have ended may the detached C2 owner enter `c2-checkpoint-save-internal!`.
   Its existing borrow/measure/encode/parse and `published?`/durability
   writer result remain authoritative. Release the detached C2 owner on
   success and on every writer outcome, without a second release after a
   trusted callback consumes it.

This order binds the detached C2 image to one checked original state before
file I/O; it does not itself expose C2 image bytes or their whole-file digest.
The later G3RCV1 companion and pair publisher still need a reviewed exact-byte
adapter. A C2 file alone is not a generation continuation.

## Source-independent compiled test matrix

| Cut | Required observation |
|---|---|
| Same owner, seeded P1/G1 committed history; repeat on that owner | One issued pointer equals the adopted O2 wrapper's native pointer; C2 parses with the same fourteen P1 values, twenty-eight moment bits, paths/aliases, options and update count; G3 policy/RNG/history/partition equals all seventeen checked words; all holds and owners drain before retry. |
| Equal-bit foreign M3, optimizer, issued state, wrapper or tokenizer | Reject before transfer and any output/file write; no equal-byte substitution grants registry identity. |
| Changed roster, tied alias, F32 value/absent-gradient bit, O2 option/update/moment, G3 history/RNG or A2 cache | Typed rejection at the next pretransfer check; original owner bytes and held authority remain recoverable or are drained once. |
| Hold allocation, partial F32/O2 clone, P1 clone, issued-state allocation, wrapper registration | Preserve first typed error, empty caller destination, no busy/borrow/clone leak and same-owner retry where recoverable. |
| C2 preflight, P1 claim, O2 claim, post-adoption exception, measure or encode | Trace exact request/result-cell ownership; no lost or double-released issued state, no file publication; fail stop on uncertain partial claim. |
| O2 output/error aliases into registered optimizer/state/handle/builder/borrow/retire storage | Reject before writes, including partial overlaps; retain the active hold and unchanged authority bytes. |
| Writer precommit and postcommit failures | Preserve C2 `published?` and durability status, release C2 exactly once, never claim a companion pair from an incomplete result. |
| Feature-off, normal/repeat, ASan+UBSan+LSan | Private symbols absent when off; one exact registry-owning F32/M3/P1/O2/A2/G3/C2 aggregate passes with pinned toolchain, sealed inputs/outputs and no leaked ownership. |

No test in this matrix is claimed by this contract. The next implementation
gate is independent review of the private adoption and claim/rollback seam,
then the exact combined-provider tests; SAVE writer, pair publication,
restore, replay and continuation admission remain separate gates.
