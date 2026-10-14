# TR3 snapshot recheck under the original held O2 owner

Status: **source-private prerequisite; no linked Eshkol gate or coordinator**.

`tr3-lease-snapshot-recheck-held-o2-internal(record, parent, hold)` is loaded
only by `native/g3r_tr3_held_o2_snapshot_recheck_root.esk`. Its caller must first
enter the ordinary idle-only TR3 snapshot lease with the exact ten-slot parent,
then acquire and root the original O2 hold. No held entry variant is needed:
the ordinary entry rejects a busy O2 owner, so a live original hold cannot
precede that lease. The function owns neither hold nor parent and publishes no
P1/O2/C2 state or result.

The recheck authenticates the enrolled 22-slot record and its exact live parent
in `snapshotting/constructing`, including the fixed model, P1 handles and
trainer control envelope. It repeats M3 and D2 idle checks, resolves the exact
registered O2 optimizer from trainer slot 13, and requires identity with the
retained optimizer record in slot 14. The held-safe native count witness takes
that record's original native pointer and the caller's hold. It authenticates
the hold-to-optimizer link, rechecks held moment bits and absent gradients,
then returns the captured count only when it remains within signed i64 and
matches the live owner. The recheck requires that count to equal trainer slot
20 before returning the enrolled record. A foreign, stale, altered, wrong-owner,
or count-mismatched hold fails before any publication.

The ordinary TR3 snapshot entry/recheck, restore path, and ordinary O2 count
and absent-gradient queries remain idle-only and unchanged. A caller that
receives an error must keep the exact parent and both original holds rooted,
drain the O2 hold before aborting the TR3 lease, and fail stop if native drain
cannot be proved. This helper does not implement that coordinator cleanup or
authorize held O2 adoption, C2 claim, SAVE receipt, or replay. A future compiled
joint gate must prove the ledger, failure cuts, reverse hold release, and zero
publication before any such admission.
