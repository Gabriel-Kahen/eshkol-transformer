# CPU-v1 resource producer — Stage A, incomplete

This private test producer is under implementation from base
`6368785d815451f409af3be42c7baec5a4c02756`. Its packet/API is provisional and
has **no admission**. Production and installed APIs stay unchanged when
`ET_CPUV1_RESOURCE_TESTING` is absent. History/capture is unsupported here.

Implemented source foundation: thirteen-domain fixed packet, aggregate
allocation/free/requested/allocator-usable counters, independent census fields,
fixed snapshot-only duplicate/cycle scratch, and explicit F32/P1/C4/C4T/E3/I64/T1
owner hooks/census. Static observer storage is reported separately from allocated
bases. F32/I64 own their child buffers; parent controls never charge those twice.
Normal glibc profile only; allocator/profile/overflow/alias/scratch failures must
prevent COMPLETE. The Scheme domain cannot be fabricated by a native test.

**UNIMPLEMENTED:** O2, generation/A2, D2, C2/IO and K1 owner hooks/census; runtime
current/adopted arena bridge; all owning-scope Scheme access/generator and genuine
P1 AOT smoke; complete collector, native unit/recipe and source-overlay custody.
All overall coverage remains unavailable until these are implemented, reviewed
and actually validated. The source allocation ledger is the initial lexical
baseline/superset, not proof its active sites are charged.

Cheap Python source tests currently check exact OFF token projection of the ten
modified original owner/include files against the immutable base, including
parenthesis/math mutation controls. This is **not** actual preprocessed-token,
object/symbol parity, C/C++ syntax, native execution or allocation-free AOT proof;
those remain NOT RUN. No training caller, 375-step loop, resource sidecar or full
resource admission is part of Stage A. Root owns common source pins and roadmap;
their intentional staleness is not bypassed or silently refreshed here.

The final Stage A leaf must implement the accepted resource contract, carry an
exact source overlay, and supply a bounded native unit plus separate genuine
owning-scope smoke recipe for root dispatch under the shared heavy lock. Stage B
waits for the independently admitted producer API.
