# C1 checkpoint container 1.0

## Status and boundary

This document specifies the accepted `eshkol-checkpoint` version-1.0 byte contract.
C1 supplies a data-only container around P1 logical state. It does not
change the five A0 persistence names, redefine a P1 state dictionary as a complete
trainer state, or implement C2 exact-resume composition.

Checkpoint bytes never name executable constructors, callbacks, native libraries,
or filesystem paths. The serialized provider identity is inert. A trusted caller
must explicitly supply an already admitted P1 provider and a separately reviewed
codec with the same provider ID. Metadata cannot select either one.

Merged P1 has no tensor byte codec and registers no production provider. I1 is not a
P1 provider or codec. Production tensor save/load therefore remains unavailable and
unsupported in C1; no public API or capability is advertised.
The isolated `fixture-v1` codec proves only binary-format, validation, ownership,
P1-binding, and atomic-I/O control flow; it is not tensor, dtype, device, numerical,
or backend evidence.

## Primitive encoding and limits

All multibyte integers are unsigned little-endian. All additions and
multiplications are checked against the selected policy before tensor-codec calls,
aggregate allocation, or slicing. Version 1 has these wire-format hard limits; a
persistence policy may only lower them:

| Item | Version-1 hard limit |
|---|---:|
| file bytes | 1 TiB |
| metadata bytes | 256 MiB |
| one tensor payload | 256 GiB |
| entries | 4096 |
| alias groups | 4096 |
| total alias members | 4096 |
| path segments / tensor rank | 64 |
| one path segment | 65536 UTF-8 bytes |
| provider ID | 127 UTF-8 bytes |

The 1-TiB value is a syntax ceiling, not a demonstrated operational size. Version 1
is non-streaming: load holds an Eshkol file bytevector plus a native staging copy and
temporary checksum/payload copies; save similarly constructs several aggregate
buffers. Deployments must select materially lower, memory-safe policy limits. No
claim is made that the hard ceilings fit available address space or memory.
Executable Eshkol tests admit the exact hard file, metadata, tensor, and entry policy
ceilings and reject each one-over value arithmetically without allocating those
sizes. A compact independently checksummed wire fixture covers the exact 4096-entry,
2048-effective-group, and 4096-member boundary plus entry/group/member one-over cases
in the independent parser. Attempting to inspect the approximately 430-KiB exact
fixture with the pinned Eshkol runtime exhausted its fixed 1-GiB heap and emitted heap
limit diagnostics, so this is explicitly parser-only hard-structure evidence, not a
runtime-validation claim. The executable Eshkol validator instead covers exact 64
and one-over 65 under a lowered entry policy. No runtime allocation at the hard
storage ceilings is claimed.

Version 1 accepts only `cpu`, `dense-row-major-contiguous`, parameter `f32`, and
buffer `bool`, `i64`, or `f32`. A valid P1 state outside that codec domain is
`unsupported`; it is never converted. Extents must fit `u64`. Payloads are exact:
bool is one byte per element and admits only 0 or 1, i64 is two's-complement
little-endian eight bytes per element, and f32 is IEEE-754 binary32
little-endian four bytes per element with all bit patterns preserved.

A trusted codec must be deterministic and must encode P1-value-equal tied snapshots
to identical canonical bytes. The encoder checks this for every alias group and
raises `internal` rather than write a self-rejecting checkpoint if the codec violates
that admission rule.

## Header

The fixed header is 128 bytes:

| Offset | Width | Field | Required value / rule |
|---:|---:|---|---|
| 0 | 16 | magic | `89 45 53 48 4b 4f 4c 43 4b 50 54 0d 0a 1a 0a 00` |
| 16 | 2 | major | 1 |
| 18 | 2 | minor | 0 |
| 20 | 4 | header bytes | 128 |
| 24 | 4 | endianness | 1, little-endian |
| 28 | 4 | checksum algorithm | 1, SHA-256 |
| 32 | 8 | required-feature bits | 0 |
| 40 | 8 | total file bytes | exact physical size |
| 48 | 8 | metadata offset | 128 |
| 56 | 8 | metadata bytes | exact metadata section size |
| 64 | 8 | payload offset | metadata offset + metadata bytes |
| 72 | 8 | payload bytes | exact tensor-payload total |
| 80 | 4 | entry count | 0..4096 and within policy |
| 84 | 4 | alias-group count | 0..4096 |
| 88 | 4 | provider-ID bytes | 1..127 |
| 92 | 2 | P1 state major | 1 |
| 94 | 2 | P1 state minor | 0 |
| 96 | 2 | P1 provider-interface major | 2 |
| 98 | 2 | P1 provider-interface minor | 0 |
| 100 | 28 | reserved | all zero |

Unknown major, nonzero minor, nonzero required-feature bits, and unsupported
well-formed state/provider interface versions are `version-mismatch`. Bad magic,
endianness, checksum ID, header size, reserved bytes, noncanonical or inconsistent
lengths, and malformed fields are `corrupt-data`.

The metadata section begins with a nonempty canonical UTF-8 provider spelling. The
parser keeps it as string/bytes and never interns checkpoint-controlled text. Decode
compares it with the spellings of the trusted codec and explicitly selected P1
provider symbols. It never drives lookup; the trusted provider symbol is the only
identity passed to P1.

The container remains version 1.0 because provider-interface major/minor were
already independent canonical header fields. Moving their required value from 1.0
to release-capable P1 provider 2.0 changes canonical bytes and the reviewed golden
digest, but does not add, remove, or reinterpret a container field. Provider 1.x
headers reject as `version-mismatch` before codec or provider callbacks; a matching
provider spelling cannot upgrade or emulate that identity.

## Entry records and payloads

Exactly `entry-count` records follow the provider ID. Each starts with this 80-byte
prefix:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | complete record bytes |
| 8 | 8 | payload offset relative to the payload section |
| 16 | 8 | payload bytes |
| 24 | 4 | encoded path bytes after the extent table |
| 28 | 2 | path segment count |
| 30 | 2 | rank |
| 32 | 1 | kind: 1 parameter, 2 buffer |
| 33 | 1 | dtype: 1 bool, 2 i64, 3 f32 |
| 34 | 1 | device: 1 cpu |
| 35 | 1 | layout: 1 dense row-major, zero-offset |
| 36 | 4 | reserved, zero |
| 40 | 8 | checked element count |
| 48 | 32 | tensor SHA-256 |

The prefix is followed by `rank` u64 extents and then the encoded path. A path is
the declared sequence of `segment-count` records, each `u32 byte-length || UTF-8
bytes`. Each segment is nonempty canonical UTF-8 and within the P1 bound. The path
byte field includes all segment length words and bytes. Records have no padding.

Records are in strict P1 segment-by-segment UTF-8 byte lexical order. Duplicate or
noncanonical paths reject. Payload offsets are contiguous in record order starting
at zero; overlap, gap, reordering, or an end other than `payload-bytes` rejects.
The declared element count must equal the checked shape product, and payload bytes
must equal element count times the exact dtype width.

The tensor digest is SHA-256 over the ASCII/UTF-8 domain bytes
`eshkol-checkpoint-tensor-v1\0`, followed by the complete canonical record with its
32-byte digest field replaced by zero bytes, followed by the exact tensor payload.
It binds path, kind, metadata, offset, shape, and data and detects corruption that
does not also recompute the digest. SHA-256 here is an unkeyed integrity checksum,
not authentication; an attacker able to rewrite the file can recompute every digest.

## Alias records

Alias records follow all entry records inside metadata. Each is `u32 member-count`,
`u32 reserved-zero`, then `member-count` u32 entry indices. A group has at least two
strictly increasing indices. Groups are ordered by their first index, are disjoint,
refer only to parameter entries, and have at most 4096 total members. The metadata
section must end exactly after the last group. The loader reconstructs canonical
paths from indices, verifies tied raw payloads are equal before tensor decoding or
P1-shell construction, and then lets P1 independently revalidate exact values and
aliases after decoding.

The header field can represent at most 4096 groups, but the two-member minimum and
4096 aggregate-member ceiling mean a canonical v1 file can contain at most 2048
nonempty alias groups. Tests exercise that effective exact ceiling, the 4096-member
exact ceiling, and independently checksummed one-over, overlapping, and reordered
group encodings through the independent parser. Executable Eshkol validation covers
multiple disjoint groups and the lowered entry-policy boundary described above; the
pinned-runtime hard-structure limitation is measured there rather than hidden.

## Container checksum and strict loading

The payload section is the exact concatenation of entry payloads. The final 32 file
bytes are SHA-256 over domain bytes `eshkol-checkpoint-container-v1\0` followed by
every file byte before the digest. The digest ends at EOF. Truncation or trailing
data always rejects.

Strict load performs, in order:

1. validate path/policy, regular non-symlink input, physical size, fixed header,
   version/features/reserved fields, checked section arithmetic, and exact EOF;
2. validate the whole checksum, every record/table bound, canonical UTF-8/path
   order, schema, payload span and tensor checksum, bool bytes, and aliases;
3. compare the inert provider identity with the explicitly supplied trusted codec
   and provider;
4. have each decoder callback publish exactly one detached owned carrier in its
   mutable request; release the complete published prefix exactly once on callback,
   request-shape, or later validation failure;
5. transfer the complete ownership ledger through P1's narrow adoption seam, build
   one unpublished logical state, explicitly bind the already admitted provider,
   and publish only after P1 validation passes.

No tensor object, P1 entry/state shell, codec callback, receiver mutation, or public
result exists before steps 1–3 finish. A late codec failure creates no P1 shells and
releases every published owned carrier. Admission failure clears carrier references,
releases all transferred owners, and revokes any unpublished state shell; only an
identity tombstone without tensor storage may remain. A later
`module-load-state-dict!` retains P1's all-or-nothing prepare/one-commit contract.
Container self-inconsistency is
`corrupt-data`; unsupported well-formed version/features are `version-mismatch`;
trusted codec/provider/device-domain mismatch is `unsupported` or the precise P1
dtype/device/layout category; system failures are `io`.

The decoder publication slot is single-assignment: a callback writes it at most once
and never clears or replaces a published owner, including before raising. An
overwritten owner is unobservable and cannot be reclaimed by C1, so every integrating
codec must prove this rule in its executable evidence. Every successful decoder
publication must be a new exact carrier identity. A codec
may not republish a carrier from an earlier request, even for tied values. C1 checks
the new identity against the complete owned prefix immediately after publication.
Before the first callback it records every descriptor/request-input identity, and it
adds each newly allocated request identity before that request enters the codec, so a
later callback cannot relabel an earlier payload, shape, or request as an owner;
on an exact duplicate it clears the duplicate ledger envelope and fails `internal`,
so the original owner is released exactly once. A codec also may not publish the
request object or any borrowed request input (payload, kind, shape, dtype, device,
layout, or operation) as its carrier. C1 clears that tentative envelope and fails
`internal` without invoking provider release on the borrowed object. A distinct
owned carrier that shares native storage is not confused with an exact identity;
P1 validates storage ownership and releases that distinct owner exactly once if
admission fails.

If decoding fails before adoption, P1's narrow release helper performs a final
registry check before provider lookup. An envelope that names storage already owned
by a live P1 state or borrowed by a registered module is cleared and rejected without
a release callback, while C1 continues cleaning every genuinely owned sibling. A
defective release callback is never retried: C1 drains the remaining ledger and then
reports one structured internal rollback defect.

The file-to-module wrapper necessarily decodes and successfully binds a fresh state
before it can compare receiver schema. It deterministically calls public
`state-dict-release!` after both successful and rejected receiver loads, so the
temporary state's carriers return to the provider baseline without changing the
receiver. Small invalidated identity tombstones may remain, but retain no tensor
storage.

## Internal build-only API

These names are private inputs to a future registry-owning consumer artifact, not
installed A0 APIs. They must be compiled with the trusted P1 root and one E1/E1B
registry owner; they are not independently linkable against another P1 registry.

| Name | Contract |
|---|---|
| `c1-persistence-policy-internal max-file max-metadata max-tensor max-tensors device` | Return a tagged lowering policy; version 1 accepts only `cpu`. The policy is borrowed and every field/ceiling is revalidated on every use, including after caller mutation. |
| `c1-checkpoint-codec-internal provider-id encode decode` | Build a trusted codec seam for one admitted provider spelling. Encode receives a carrier only during one paired state-backed borrow. Decode receives a mutable request and must return that same request after publishing exactly one newly owned carrier. The callbacks are trusted artifact code, never selected by bytes. |
| `c1-checkpoint-encode-state-internal state policy codec` | Validate a bound live P1 state, resolve each read-only state-backed handle only for a paired synchronous borrow, and return newly owned canonical bytes. No raw carrier is retained. |
| `c1-checkpoint-decode-state-internal bytes policy codec provider-name` | Strictly validate bytes and exact provider interface 2.0 before callbacks, explicitly select the trusted provider, then return a newly owned releasable bound P1 state. |
| `c1-checkpoint-save-state-internal! state path policy codec overwrite?` | Encode and atomically publish; borrow state and return `#t` on proved success. |
| `c1-checkpoint-load-state-internal path policy codec provider-name` | Exact-read then strict-decode into a newly owned bound state. |
| `c1-checkpoint-load-module-internal! module path policy codec provider-name` | Load a complete temporary state, delegate receiver mutation to P1's recoverable-error atomic load, and release the temporary on success or failure. Trusted-provider commit invariant failures are not rollback claims. |
| `c1-checkpoint-inspect-bytes-internal bytes policy` / `c1-checkpoint-inspect-internal path policy` | Validate the entire container without decoding tensors; return new CPU metadata. |
| `c1-checkpoint-metadata-ref-internal metadata key` | Read `format-id`, `format-version`, `required-features`, `checksum-algorithm`, `provider-id` (newly copied string), `tensor-count`, `payload-bytes`, `metadata-bytes`, or `file-bytes`. |

All failures use the E1 category vocabulary. File-system failures are `io`; C1
native failures include bounded status details. There is deliberately no standalone
prelocalized public C1 package in this workstream because that would create a second
P1 registry. C2 must place C1 and the final trainer schema inside its one reviewed
registry-owning E1B consumer artifact before exposing the A0 persistence facade.

## Atomic local I/O

Pinned Eshkol does not expose verified short-write-safe fsync, no-replace rename, or
directory-fsync primitives. C1 therefore uses a narrow versioned C11/Linux boundary
that sees only paths and bytevectors. It performs no P1 schema, tensor, provider,
callback, checksum, or capability logic.

Save opens the parent directory, creates an unpredictable same-directory mode-0600
temporary with `openat(O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC)`, completes a checked
EINTR/short/zero-write loop, fsyncs and successfully closes the temporary, then uses
`renameat` for overwrite or Linux `renameat2(RENAME_NOREPLACE)` for no-overwrite and
fsyncs the parent directory. Rename is the publication commit point. Before it, an
error leaves the destination unchanged and attempts cleanup of exactly that
temporary. A crash or cleanup failure may leave an untrusted orphan; C1 never
reuses, scans, or sweeps orphans. After rename, a directory-fsync error reports `io`
with publication already visible and crash durability unknown.

The native ABI is `1.0`; `native/checkpoint_io.h` is canonical. Its only production
symbols are `et_checkpoint_io_abi_major_v1()`,
`et_checkpoint_io_abi_minor_v1()`,
`et_checkpoint_io_read_exact_v1(path, bytevector, expected, maximum)`, and
`et_checkpoint_io_atomic_write_v1(path, bytevector, expected, overwrite)`.
The pinned Eshkol bytevector pointer addresses an i64 length followed immediately by
its bytes; the boundary verifies that length before access and snapshots write input.

The native status word is stable ABI data: bits 0..15 hold `errno`, bits 16..23
identify the failed stage, bit 24 is `published?`, and all other bits are zero.
E1B `io` errors retain that word as `source-code` and also expose
`native-errno`, `native-stage`, `published?`, and `durability`. Callers must use
`published?` rather than treating every reported failure as proof that the old
destination remains visible.

Stage numbers are: 0 none, 1 validate, 2 allocate, 3 open-parent, 4 open-source,
5 stat-source, 6 read-source, 7 close-source, 8 random-temp, 9 create-temp,
10 write-temp, 11 sync-temp, 12 close-temp, 13 publish, 14 sync-directory,
15 cleanup-temp, and 16 close-directory. Both sync-directory failure and a later
close-directory failure are conservatively reported with `published? = #t` and
`durability = unknown`.

The verified claim is limited to syscall sequencing, atomic old-or-new visibility,
and successful-return durability semantics on the tested local Linux filesystem;
supported Ubuntu 22.04 CI is still required. No power-loss experiment has been run.
It is not a universal
power-loss, storage-hardware, NFS, FUSE, network-filesystem, concurrent-writer, or
untrusted-parent-directory guarantee.
