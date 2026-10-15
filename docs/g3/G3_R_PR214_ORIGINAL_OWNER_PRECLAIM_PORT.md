# PR214 original-owner and O2 preclaim source port

Status: **preclaim-only source prerequisite**. The later
[C2 partial-claim union](G3_R_PR214_C2_PARTIAL_CLAIM_UNION.md) ports the
rollback prerequisite and a separate C2 transfer check.

This branch starts from the accepted PR214 held-O2/TR3 union `ec97f61a`.
`g3r_original_owner_coordinator_root.esk` loads the original trainer-held G3
owner, held-O2 TR3 recheck, and original-owner coordinator in one source
registry. Its existing O2 bridge obtains a fresh sealed M3T roster and begins
the exact original optimizer hold. On failure or explicit end, it drains O2
before the G3 hold and trainer snapshot lease.

`g3r_original_owner_preclaim_root.esk` adds only the reviewed O2 adopter's
preclaim enrollment and release helpers. The adopter allocates its Scheme
shell, registry cell, native pointer cell, and ledger before issuing the held
O2 state. It checks the exact issued pointer and original optimizer before
enrollment. A preclaim failure retains the rooted ledger until native release
is proven dead; a proved-live release can be retried once, and uncertain
lifecycle fails stop. The original holds remain owned by the coordinator.

At this leaf's `6b49e29a` head, C2 compose had a one-bit P1 claim-progress cell and no
`c2-training-state-rollback-o2-or-live!` or
`o2-state-claim-c2-rollback-or-live-internal!`. If its O2 claim mutates and
then raises, the caller cannot prove or reverse that transfer. The private
source commit `5fff0548` introduced the missing O2 rollback-or-live contract
and postmutation runtime cuts; the later C2 partial-claim union carries the
independently reviewed port. The preclaim leaf's
`g3r-held-o2-adopter-c2-claim!` helper is intentionally excluded.

The preclaim host runner checks the coordinator's eight source mutations,
adopter's five source mutations, preclaim boundary, O2 and G3 native failure
cuts, and feature-off builds. It does not compile a linked Eshkol aggregate or
exercise C2 partial claim, post-adoption cleanup, SAVE, or replay.
