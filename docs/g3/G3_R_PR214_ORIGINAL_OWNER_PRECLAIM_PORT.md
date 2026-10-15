# PR214 original-owner and O2 preclaim source port

Status: **private source and host prerequisite; C2 transfer blocked**.

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

The PR214 C2 compose source still has a one-bit P1 claim-progress cell and no
`c2-training-state-rollback-o2-or-live!` or
`o2-state-claim-c2-rollback-or-live-internal!`. If its O2 claim mutates and
then raises, the caller cannot prove or reverse that transfer. The existing
private source commit `5fff0548` introduces the missing O2 rollback-or-live
contract and postmutation runtime cuts; it requires independent acceptance
and an exact PR214 port before a C2 claim can be attempted here. The adopter's
`g3r-held-o2-adopter-c2-claim!` helper is intentionally excluded.

The focused host runner checks the coordinator's eight source mutations,
adopter's five source mutations, preclaim boundary, O2 and G3 native failure
cuts, and feature-off builds. It does not compile a linked Eshkol aggregate or
exercise C2 partial claim, post-adoption cleanup, SAVE, or replay.
