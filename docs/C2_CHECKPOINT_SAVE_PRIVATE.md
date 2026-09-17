# Private C2 checkpoint save

`c2-checkpoint-save-internal!` is a source-composed C2-private operation. It is
not an A0/K2 API and provides no load, reconstruction, or capability surface.

The operation admits options, required features, path, policy, and the exact C2
owner before filesystem work. It borrows that owner synchronously, revalidates
its deep-owned inert controls in a discard-only preflight region, uses the
committed claimed-owner C1 model and O2 staging seams, then assembles with
`et_c2_checkpoint_encode_v1`. The native
bridge recomputes outer and moment digests and requires a complete
`et_c2_checkpoint_parse_v1` load-mode parse before returning the image. The C2
borrow ends before the existing C1 atomic writer publishes it. Atomic writer
errors retain their `published?` and `durability` fields.

SAVE materializes lazy P1 entry/tensor shells once before SAVE-owned scratch and
records a private monotonic readiness bit. A narrow P1 trusted helper validates
the exact active C2 marker and creates only the ordered entry-shell list; it
does not deep-project metadata or clone payloads. P1's registry retains the
canonical shell identities, while the bounded ordered list remains call-local.
One
preallocated five-slot tensor-access record is filled and cleared for each
serial access, so regional callers never publish a short-lived access record
into the owner. Each owner borrow also receives a fresh, compact private vector
identity as its access token. Authentication compares that exact identity; a
copied, foreign, prior-borrow, or post-release token remains invalid without
constructing and retaining a runtime exception graph on every SAVE. The exact
borrow/token/access graph is published through the owner's state-slot barrier
and read back from that slot before P1/O2 activation. Thus a caller-region
allocation is canonicalized/promoted into owner lifetime; only the activation
guards remain in temporary scratch. P1's existing write barrier promotes its
small installed marker, while successful guard state is destroyed before
`borrow-begin` returns.

Each SAVE first revalidates and recopies its authoritative serialized controls
inside one sibling preflight region, then discards that copy. The owner already
holds the only deep-owned, application-inaccessible control set. Private
set!-replaceable SAVE test seams are trusted and contractually read-only; the
ordinary vector/bytevector representation does not physically prevent a
defective trusted replacement from mutating it. Under that private contract,
the following stages read the exact owner controls without retaining a second
resolved-configuration/cursor image. SAVE then
performs two independent regional component builds. The first returns only the
measured file size; after it is destroyed, the parent scratch allocates the
destination bytevector. The second rebuilds and encodes into that destination,
and is destroyed before the C2 borrow ends. The existing atomic writer consumes
the parent-regional destination synchronously, so its native snapshot may
coexist with one destination image but not with either component stage or the
control-validation copy. Scratch teardown is explicit region destruction;
correctness assumes no garbage collection or finalizer. Repeated-save and
operational tests must prove identity/counter stability, byte identity, and the
fixed memory limits; this private design alone does not establish the 16 MiB /
512 KiB / 8 MiB / 64 operational target. Physical payload section bytes and
file bytes remain separate internal quantities.

## Operational evidence

The canonical operational gate now passes two independent fresh AOTs through
one-process LOAD, E=62/P=1 owner retention, three deterministic SAVEs, and
release. The two measured joint runs reported, respectively:

- baseline/retained/peak RSS of 32,600/269,284/517,012 KiB in 711,451 ms;
- baseline/retained/peak RSS of 34,996/271,332/519,048 KiB in 692,906 ms.

Both peaks remain below the fixed 524,288 KiB ceiling. Exact first-touch SAVE
retained 2,145,336 arena bytes to materialize permanent P1 identities. A
preserved developer probe measured the warmed second and overwrite SAVE deltas
at 1,712 and 1,808 bytes; the canonical runs independently enforce that each is
below 32 KiB. All byte outputs matched the exact 16 MiB fixture, and the full
file/metadata/tensor/count/corrupt one-over matrix passed on both AOTs without a
reject artifact.

This is development-host evidence from CachyOS with Clang 22. It does not claim
the project-supported Ubuntu 22.04/LLVM 21 result, public K2 composition, or a
released checkpoint capability; those gates remain outstanding.
