# Private TR3 snapshot eval-mode ticket

Status: **source prerequisite; compiled runtime and independent review pending**.

`tr3-c-snapshot-with-eval-internal!` accepts the enrolled trainer record, its
exact live ten-slot snapshot parent, and a trusted zero-argument callback. The
lease first authenticates the original 17-module/14-handle train set and the
snapshot parent. A private one-use ticket is rooted while the callback runs;
no ticket or P1 mode token is returned to the caller. The helper binds the
existing P1 E3 mode authority (trusted slots 64–68), prepares and enters eval,
then requires the new exact fixed-set eval recheck (slot 73) before and after
the callback. The callback may use the enrolled model only within that interval.

Normal return and a raised callback both restore the original train modes via
P1, require the unchanged exact train recheck (slot 70), unbind the P1 token,
and clear the active ticket before returning or rethrowing the first error.
If restore, train recheck, or unbind cannot prove custody, the process fails
stop with exit code 134 while the parent remains rooted. Snapshot abort checks
the active ticket before releasing C2/O2/P1 owners; lease abort and success
unlink reject an active ticket before marking the parent dead. This preserves
the snapshot parent v1 shape and train-only lease behavior.

The source fixture stages genuine trainer success, callback failure rollback,
copied/stale parent and overlapping ticket rejection. Source mutants enforce
the mode slots, eval checks, restore order, and guards before owner release or
lease unlink. The fixture and fail-stop cuts have **not** been compiled or run.
The helper is private to the existing TR3-C snapshot aggregate; no G3 binding,
C2 transfer, receipt, SAVE, or data-only continuation follows from this leaf.
Only one mode ticket may be active in a process at a time. P1's existing E3
unbind retains a dead token tombstone; repeated-scope retention has not been
measured for this composition.
