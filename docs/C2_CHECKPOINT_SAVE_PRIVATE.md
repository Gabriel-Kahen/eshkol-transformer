# Private C2 checkpoint save

`c2-checkpoint-save-internal!` is a source-composed C2-private operation. It is
not an A0/K2 API and provides no load, reconstruction, or capability surface.

The operation admits options, required features, path, policy, and the exact C2
owner before filesystem work. It borrows that owner synchronously, revalidates
and detaches its inert controls, uses the committed claimed-owner C1 model and
O2 staging seams, then assembles with `et_c2_checkpoint_encode_v1`. The native
bridge recomputes outer and moment digests and requires a complete
`et_c2_checkpoint_parse_v1` load-mode parse before returning the image. The C2
borrow ends before the existing C1 atomic writer publishes it. Atomic writer
errors retain their `published?` and `durability` fields.

SAVE materializes lazy P1 entry/tensor shells once in the owner's enclosing
arena and records a private monotonic readiness bit. A narrow P1 trusted helper
validates the exact active C2 marker and creates only the ordered entry-shell
list; it does not deep-project metadata or clone payloads. Registry cells and
entry/tensor shells are therefore created directly in the established
owner-lifetime allocation context, not evacuated from temporary scratch. One
preallocated five-slot tensor-access record is filled and cleared for each
serial access, so regional callers never publish a short-lived access record
into the owner. Each owner borrow also receives a fresh, compact private vector
identity as its access token. Authentication compares that exact identity; a
copied, foreign, prior-borrow, or post-release token remains invalid without
constructing and retaining a runtime exception graph on every SAVE.

Each SAVE then performs two independent regional component builds. The first
returns only the measured file size; after it is destroyed, the parent scratch
allocates the destination bytevector. The second rebuilds and encodes into that
destination, and is destroyed before the C2 borrow ends. The existing atomic
writer consumes the parent-regional destination synchronously, so its native
snapshot may coexist with one destination image but not with either component
stage. Scratch teardown is explicit region destruction; correctness assumes no
garbage collection or finalizer. Repeated-save and operational tests must prove
identity/counter stability, byte identity, and the fixed memory limits; this
private design alone does not establish the 16 MiB / 512 KiB / 8 MiB / 64
operational target. Physical payload section bytes and file bytes remain
separate internal quantities.
