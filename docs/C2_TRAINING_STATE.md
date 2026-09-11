# C2 detached training-state checkpoints

Status: **implementation / dependency-gated**. Integration accepted the C2 1.0
logical schema and wire format in issue #1 comment `5628609463`, with the final
nonallocating model-copy seam accepted in comment `5628659732`. Separable
parser, codec, and ownership work may proceed. The public source-composed C2
artifact, capability-facing `checkpoint-load/3`, and provisional 81-global /
75-export boundary must not be published until K2 issue #73 is independently
approved and merged.

## Scope

C2 is a data-only, deep-owned checkpoint of one detached update-boundary
training state:

```scheme
(transformer-trainer-state 1 0 ()
  (library-identity "0.1.0-draft" eshkol-training-state 1 0)
  (compiler-identity eshkol 1 3 4 evolve
                     90cbd7130f47b8184bcc77b8d5c1b0026da980de)
  (tensor-provider 2 0 i2-dense-cpu-f32-v1 f32 cpu row-major)
  <owned-P1-state-1.0>
  <owned-O2-state-1.0>
  (rng philox4x32-10 1 key counter-low counter-high)
  (tokenizer fingerprint vocabulary-size)
  (dataset ESHKDCU1 1 0 current-cursor epoch-start-cursor)
  (resolved-config eshkol-resolved-run 1 0 canonical-X1 fingerprint)
  (counters total-tokens completed-updates epochs))
```

The state contains no callbacks, addresses, provider-selection authority,
capability evidence, filesystem path, timestamp, host identity, live batch,
gradient accumulation, or partially applied optimizer step. `completed-updates`
equals O2's successful-update count. RNG version is one, its key equals the X1
`run.seed`, and both 64-bit counter words are preserved exactly.

C2 owns canonical bytes, detached component reconstruction, exact cleanup, and
atomic file publication. TR3 owns a joint live model/optimizer/dataset/RNG
restore transaction and interrupted/resumed training proof. G3/TR3 own
save/reload generation equivalence. C2 does not claim those later gates.

## Public surface and lifetime

The existing A0 persistence arities remain:

```scheme
(checkpoint-inspect path policy)
(checkpoint-metadata-ref metadata key)
(checkpoint-load path policy capability-report)
(checkpoint-save! state path policy options)
```

C2 adds only `(trainer-state-release! state)` to the eventual
`transformer.trainer` facade. A registered idle live state transitions
`live -> releasing -> dead`. It invalidates dependent handles before draining
its P1 and O2 owners exactly once. Exact dead release is idempotent; every other
dead use is `invalid-state`. Busy, borrowed, reentrant, or releasing states are
`invalid-state`. Forged, copied, wrong-kind, or unregistered shells are
`invalid-argument` before native dereference. Cleanup continues after a
provider defect, publishes dead, then reports `internal`; consumed authority is
never retried. No finalizer or freed-region safety is claimed.

## Container 1.0

All integers are little-endian. The file has a fixed 256-byte outer header,
canonical outer metadata, a physical payload containing one complete C1 model
container followed by O2 moment bytes, and a final SHA-256 digest. The digest is
domain-separated with `eshkol-training-state-container-v1\0`; it provides
integrity, not authentication.

The header is:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 16 | fixed training-state magic |
| 16 | 2+2 | format version 1.0 |
| 20 | 4 | header bytes = 256 |
| 24 | 4 | little-endian identifier = 1 |
| 28 | 4 | SHA-256 identifier = 1 |
| 32 | 8 | required features = 0 |
| 40 | 8 | total file bytes |
| 48 | 8 | metadata offset = 256 |
| 56 | 8 | outer metadata bytes |
| 64 | 8 | physical payload offset |
| 72 | 8 | physical payload section bytes |
| 80 | 4 | model tensor count |
| 84 | 4 | unique parameter count |
| 88 | 4 | optimizer tensor count = 2P |
| 92 | 4 | aggregate tensor count |
| 96 | 8 | nested C1 model bytes |
| 104 | 8 | O2 metadata bytes |
| 112 | 8 | O2 moment payload bytes |
| 120 | 8 | canonical X1 bytes |
| 128 | 4 | current D2 cursor bytes |
| 132 | 4 | epoch-start D2 cursor bytes |
| 136 | 4 | tokenizer fingerprint bytes |
| 140 | 4 | config fingerprint bytes = 93 |
| 144 | 4 | provider bytes = 19 |
| 148 | 4 | library identity bytes = 57 |
| 152 | 4 | compiler identity bytes = 71 |
| 156 | 4 | optimizer group count |
| 160 | 24 | six adjacent component version pairs |
| 184 | 8 | total tokens |
| 192 | 8 | completed updates |
| 200 | 8 | completed epochs |
| 208 | 8 | RNG version word = 1 |
| 216 | 8 | RNG key word |
| 224 | 8 | RNG counter-low word |
| 232 | 8 | RNG counter-high word |
| 240 | 16 | zero reserved bytes |

The version pairs are P1 state 1.0, P1 provider interface 2.0, O2 state 1.0,
RNG state 1.0, D2 cursor 1.0, and X1 resolved format 1.0. Signed RNG words use
their exact two's-complement `u64` representation.

Outer metadata is packed, in order: tokenizer fingerprint, config fingerprint,
provider identity, library identity, compiler identity, canonical X1 bytes,
current cursor, epoch-start cursor, then O2 metadata. The tokenizer fingerprint
is exactly the 96-byte T1 or 95-byte T2 family; matching cursor lengths are 304
or 303 bytes. Vocabulary is derived from equality between X1 and both cursor
fields, not duplicated.

Public `payload-bytes` means tensor data only: nested C1 tensor payload bytes
plus O2 moment payload bytes. It excludes both container headers, metadata, and
digests. Physical payload section bytes and file bytes are separate internal
quantities.

The nested model is an exact C1 1.0 container using provider interface 2.0 and
`i2-dense-cpu-f32-v1`. Let `E` be its entry count, `L` all parameter-alias
members, and `G` parameter-alias groups. The unique optimizer parameter count
is exactly `P = parameter-entry-count - L + G`; buffers never contribute. O2
stores exactly two canonical moments for each unique parameter, in P1 path
order, with domain-separated per-moment SHA-256 digests and exact binary32 bits.
Paths, ranks, extents, aliases, group coverage, O2 completed updates, and all
duplicated lengths/counts must agree with P1 and the outer header.

## Validation and limits

Wire hard limits are 1 TiB file bytes, 256 MiB artifact-wide metadata, 256 GiB
per tensor, 6,826 total tensors, 4,096 model entries, 1,365 unique optimizer
parameters/groups, rank/path depth 64, UTF-8 segment bytes 65,536, D2 cursor
bytes 400, and canonical X1 bytes 16,384. A caller policy may only lower them.

The initial operational target is 16 MiB file bytes, 512 KiB artifact-wide
metadata, 8 MiB per tensor, and 64 total tensors. This tuple is not a wire
property and is not considered supported until exact/one-over, jointly
attainable, repeated-save, memory, timing, and no-warning probes pass on the
pinned toolchain. It must not be silently lowered.

Input uses one `O_NOFOLLOW` regular-file descriptor. C2 probes the outer and
nested C1 fixed headers, applies size/metadata/count checks before allocating
the full file, then reads the bounded complete image from the same descriptor.
Before parser allocation or callbacks it compares both retained headers with
the probes, reruns all available checked arithmetic and hard/caller/profile
limits on retained bytes, and requires final same-fd size plus exact EOF
agreement. Complete metadata parsing then enforces the per-tensor operational
limit before any tensor codec or native tensor allocation. The descriptor is
not an immutable snapshot; no claim is made to detect every concurrent write.

Strict load order is physical/header admission; full digest; UTF-8 and inert
identity checks; X1, both D2 cursors, C1, O2 metadata/moment digests and every
cross-component invariant; K2 runtime/capability admission; then I2/O2
reconstruction. Valid-checksum semantic corruption must invoke zero codecs.

Malformed lengths, reserved bytes, ordering, UTF-8, checksums, spans, and
semantic inconsistencies are `corrupt-data`. Well-formed nonexact format or
component versions are `version-mismatch`. Provider mismatch is `unsupported`.
A well-formed nonexact compiler identity is `unsupported` for inspect and
`determinism-unavailable` for load. Dead/busy/gradient-present state is
`invalid-state`; filesystem and native staging-I/O allocation failures are
`io`; logical projection/parser/reconstruction/owner allocation failures are
`internal`.

## Allocation-flat save seams

O2 moments use the localized state-owner validator followed directly by
`et_f32_tensor_copy_bits_to_v1`. Model tensors use the accepted C2-only
`et_c2_private_i2_state_owned_copy_bytes_v1` while C1's existing P1
state/tensor pin is live. The lexical I2 carrier preflight proves the
state-owned clone role; native code hardcodes the role and resolves liveness.
Neither path allocates or retires an I2/O2 borrow shell, retains a pointer,
performs numeric conversion, or transfers ownership. Repeated saves must prove
flat live and retired control counts separately from region-bounded payload
scratch.
