# G3-R held native save capture candidate

This feature-gated source candidate adds a private reservation over one exact
idle G3-T generator and its registered G3-T model. It is an implementation
prerequisite for the [same-owner SAVE writer](G3_R_SAME_OWNER_SAVE_WRITER_PREREQUISITE.md),
not a C2 owner, Eshkol wrapper, file writer, SAVE admission or continuation.
The gate is `ET_G3R_COMMITTED_SAVE_HOLD_PRIVATE`, layered on the committed
history and one-shot save-copy gates.

The private native API is:

```c
int64_t et_g3t_private_committed_save_hold_begin_v1(
    void *generator, void *expected_model);
int64_t et_g3t_private_committed_save_hold_copy_v1(
    int64_t token, et_g3r_committed_save_capture_v1 *destination);
int64_t et_g3t_private_committed_save_hold_end_v1(int64_t token);
```

`begin` returns a positive, monotonic token or zero with the existing G3-T
error record. Only live reservations are retained; `end` removes and frees
one, and old tokens cannot release a successor. The native owner holds all
14 F32 parameter pins and an A2 cache read borrow, records the exact cache
pointer, and marks the generator and model busy. Ordinary generator entries,
including abort and close, reject until `end`. The held `copy` validates the
same model, parameter identities and 4,736 bound parameter bytes, cache
identity/length, policy, all four RNG words, ordered history, partition,
generated-EOS classification and nonpending G1 provenance. It compares the
complete 4,872-byte capture with the exact baseline taken at `begin`, so even
a valid in-place policy, RNG or history change denies the copy. It writes one
detached `136 + 4736` byte capture only after all checks. The caller supplies
writable disjoint storage; the native entry checks owner and live-hold spans
and A2/F32/I64 storage overlap. `end` rechecks before draining both leases;
a recoverable validation failure still releases the reservation. An
impossible borrow-end defect fails stop.

Focused evidence is `scripts/test-g3r-committed-save-snapshot.sh`: feature-off
symbol closure and controlled native normal, repeat and ASan/UBSan runs. The
test covers hold/copy/end, unchanged destination on alias or model-bit drift,
wrong model, pending G1, pin/cache acquisition failure, cache-length and
cache-identity drift, valid policy/RNG/history drift, abort/call/close denial,
and stale-token ABA denial.
`scripts/test-g3r-committed-history-owner.sh` remains green. These controlled
stubs do not establish real A2/F32 lease behavior in a compiled Eshkol
aggregate. The next owner bridge must compare this pinned image with an exact
M3T/P1/O2 capture under one reviewed stable interval; it cannot treat this
detached native copy as a C2 owner or file digest.
