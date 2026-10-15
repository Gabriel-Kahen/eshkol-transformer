# PR #214 original-owner coordinator port boundary

Base: `c71d39a98411d2285fa099f63d1c56170069da3c` (tree
`34ae8017fcb037548974073926bb26ef5fee7fdc`). The reviewed source-only
coordinator `d50d2bd507b13b5299465a831001549373e85b0b` (tree
`0d30fc8d9317463f656f5031287e308c59c27606`) descends from a different
source line. Its independent review accepts only the private G3/O2 hold
prerequisite, without a linked Eshkol run or C2 SAVE admission.

The trainer-held G3 leaf is dependency-closed on this base. Its implementation
files are byte-identical to the reviewed leaf; the host tests add feature-off
and source-mutation negatives. The port provides one feature-private
`g3r-trainer-g3-save-begin!`/`end!` root and the native roster-to-G3 bridge.
The source wrapper admits the same registered, idle trainer, train-mode M3T,
raw V256 tokenizer and live generator; it enrolls a TR3 snapshot lease before
the native G3/M3/A2 hold and drains that hold before aborting the lease. The
native bridge obtains the original model's fourteen-handle roster and passes
it to the installed G3 hold. Its returned private ledger owns no O2 or C2
child. The host gate exercises foreign/stale/busy and repeated dead-hold
rejection with controlled native stubs; it is not a linked Eshkol fixture.

The next reviewed coordinator root cannot be linked on this base. In
`d50d2bd`, `native/o2_optimizer_internal.h` declares the private
`et_o2_g3r_save_hold` and `et_g3r_private_o2_save_begin_v1`, check, state,
completed-updates, owner/issued-match and end entries. Its
`native/o2_wave2_package_bridge.c` then exposes
`et_g3r_private_trainer_o2_save_begin_v1`/`end_v1` and the held-O2 adopter
entries. None of those native contracts exists in the PR #214 files. Thus
`g3r_tr3_held_o2_snapshot_recheck_extension.esk` cannot call its required
completed-updates witness, `g3r_held_o2_adopter_extension.esk` cannot own an
issued held state, and `g3r_original_owner_coordinator_extension.esk` cannot
acquire or release its O2 cell. Copying their roots now would create an
unresolved and unauthenticated ownership chain.

The next gate is a separately reviewed selective port of the native O2 held
owner and its package bridge, with exact original-optimizer and fourteen-handle
binding, held moment/count checks, feature-off symbol absence, foreign/stale/
busy and release-fault negatives. Only then can the held-O2 TR3 recheck,
adopter, and original-owner coordinator roots be source-composed and given a
linked same-registry Eshkol gate. C2 claim, SAVE receipt, file-pair writer,
candidate rollback and resumed replay remain later gates.
