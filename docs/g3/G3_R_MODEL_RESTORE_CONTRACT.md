# G3-R snapshot, restore and replay seam contract

**Proposed exact contract; implementation requires root acceptance.** This file
closes the snapshot/stage/copy/replay/rollback placeholders in
[the G3-R persistence proposal](G3_R_PERSISTENCE_PROPOSAL.md). It does not change
the capsule bytes, public calls, C1 format, SHARED-R2 capability, C4 numerical
schedule or sampler contract. It authorizes no implementation by itself. The
restricted sampler source remains outside this work.

The source basis is the C4 model/transport contract at
`ab876989245959b35198cfdbf57e6bee2eee7571`, runtime reservation candidate
`b7bb6d8e40e5316943d6c32058eca006b6d01e51`, and SHARED-R2 candidate
`8047cec9` over implementation `939f1c835282a468b901c1d4b48b9889e7e93c89`.
None is described here as an accepted final integration pin. Final compiler and
execution-lane identity bytes remain a separate dependency-owned freeze; an
artifact without that exact accepted manifest rejects SAVE and LOAD before R
authority is created.

## Fixed public boundary and private registries

The only public additions remain `generation-checkpoint-save!/4` and
`generation-checkpoint-load/3`. Both exist only in the eventual single Wave 3
aggregate. No raw snapshot, stage, source carrier, pending model, replay context
or provenance operation is public or installed.

`g3r-registry` is one process-lifetime root list. Every entry has exactly fourteen
slots:

| Slot | Value |
|---:|---|
| 0 | exact private shell |
| 1 | `g3r-snapshot`, `g3r-stage`, `g3r-reconstruct`, or `g3r-provenance` |
| 2 | kind-specific lifecycle below |
| 3 | primary parent: generator entry, image bytevector, stage entry, or model entry |
| 4 | exact C2 persistence-policy entry |
| 5 | exact T1 raw-tokenizer registry entry |
| 6 | kind-specific owned payload |
| 7 | fixed metadata projection |
| 8 | exact cleanup/ownership ledger |
| 9 | kind-specific linked-state record or pending model entry |
| 10 | linked pending generator entry or `#f` |
| 11 | preallocated result pair or immutable capsule digest |
| 12 | acquired mask or immutable lane digest |
| 13 | `diagnostic-c4` |

A snapshot is `pending`, `live`, `consuming`, or `dead`. A stage is `live`,
`admitted`, `consuming`, `consumed`, or `dead`. A reconstruction is `open`,
`prepared`, `committed`, or `aborted`. Provenance is `pending`, `live`, or
`dead`. Every lookup authenticates exact shell, kind, lifecycle, registry
membership and all required cross-links before reading a payload slot. A file
path, digest, bytevector, provider string/report, I2 carrier, P1 state, native
pointer or same-shaped vector grants no R authority.

The snapshot payload in slot 6 is a canonical vector of fifteen standard I2
`owned-clone` carriers in C1 logical-path order. The two tied logical paths are
distinct snapshot clones copied from the same pinned C4 owner; no snapshot carrier
identity is duplicated. Slot 8 is their exact release-
envelope ledger. Its projection in slot 7 contains normalized policy, four exact
RNG words, complete history, H, P0 and stop. The stage owns the complete file
image in slot 3, its exact nested C1 bytevector and inspected descriptor graph in
slot 6, and the 256-byte projection below in slot 7. After reconstruction starts,
stage slot 9 is exactly a two-element vector of the reconstruction entry and the
decoded P1 state or `#f`. Before that transition it is `#f`; after decoded-state
release it retains the reconstruction entry and `#f`; final consumption or abort
clears it to `#f`. No slot simultaneously holds two unframed values.

One reconstruction entry owns, directly or through its ledger, the exact stage,
pending C4 model entry, restored native owner, I2/P1 construction identities,
canonical fourteen-handle vector, pending native generator/replay control,
pending generator entry, provenance entry and preallocated two-element result.
Its acquired mask records those authorities in that order; cleanup reads and
clears only recorded bits.

Successful restored provenance stays in a separate fourteen-slot registry entry.
Its slots are exact shell, `g3r-provenance`, lifecycle, model entry, policy entry,
tokenizer entry, immutable tokenizer fingerprint, fixed metadata projection,
empty cleanup ledger, model entry, `#f`, immutable capsule digest, immutable lane
digest and `diagnostic-c4`, in slots 0..13 respectively. It records origin
evidence only and grants no construction, parameter, replay, SAVE or generator
authority. C4 model slot 7 remains `#f`; no existing C4 record shape changes.
Abort tombstones a pending provenance entry. Successful model process lifetime
also retains its immutable provenance entry.

## Fixed file staging and projection

The source-private file bridge has exactly these declarations:

```c
int64_t et_g3r_private_stage_measure_v1(
    void *path_bytes, int64_t max_file, int64_t max_metadata,
    int64_t max_tensor, int64_t max_tensors, void *measure_bytes);
int64_t et_g3r_private_stage_read_v1(
    void *path_bytes, int64_t max_file, int64_t max_metadata,
    int64_t max_tensor, int64_t max_tensors,
    void *measure_bytes, void *image_bytes, void *projection_bytes);
int64_t et_g3r_private_image_validate_v1(
    void *image_bytes, void *projection_bytes);
int64_t et_g3r_private_last_error_domain_v1(void);
int64_t et_g3r_private_last_error_category_v1(void);
int64_t et_g3r_private_last_error_code_v1(void);
```

Their private Eshkol aliases are `g3r-native-stage-measure` arity 6 and
`g3r-native-stage-read` arity 8, `g3r-native-image-validate` arity 2, and the
three `g3r-native-last-error-*` aliases at arity 0. The arguments named
`*_bytes` are authenticated Eshkol bytevector headers. These aliases are
compiler/source-private and grant no public file or bytevector authority.
Measure requires a writable 64-byte output disjoint from the path bytes and
leaves it unchanged on failure. Read requires a readable 64-byte measurement, a
writable image whose length equals the measured physical bytes, and a writable
256-byte projection; path, measurement, image and projection byte ranges must be
pairwise disjoint, and failure leaves both writable outputs unchanged. Image
validation requires a readable exact-length image and a disjoint writable
256-byte projection left unchanged on failure. Every length and range-end
calculation is checked before dereference.

Measure writes exactly 64 bytes, all little-endian:

| Offset | Bytes | Value |
|---:|---:|---|
| 0 / 4 | 4 each | structure bytes, 64 / measurement schema, 1 |
| 8 | 8 | exact physical file bytes |
| 16 | 8 | outer metadata bytes |
| 24 / 32 | 8 each | nested C1 offset / complete nested C1 bytes |
| 40 / 48 | 8 each | history offset / history bytes |
| 56 | 8 | final digest offset, exactly physical bytes minus 32 |

Measure independently performs one bounded regular non-symlink open, uses the
same descriptor for header reads and pre/post `fstat`, requires stable identity,
size and regular-file mode within that call, closes it, and retains no caller
pointer or persistent owner. Read reopens the path and repeats that same-descriptor
pre/post check, recomputes the complete 64-byte measurement, and requires exact
agreement with the supplied measurement before reading exactly the measured bytes
and requiring EOF. It validates outer arithmetic, reserved bytes, digest, fixed
metadata, C4 schema, history/policy/RNG/latch semantics and nested-C1 framing
before publishing either output. Replacement between the two calls is harmless:
the second descriptor, complete measurement and final digest are authoritative.
Failure leaves both output bytevectors unchanged.

`image_validate` runs the identical complete parser over one already built exact
image, writes the same projection only after success and performs no I/O. SAVE
uses this entry point for strict self-parse before publication. Measure, read and
image validation use status returns and push no Eshkol handler.

All `et_g3r_private_*` pointer functions return null on failure and status
functions return 0 on success or the nonzero R category. Their thread-local first
error domain is always 0 R. Categories are 1 invalid-argument, 2 invalid-state,
3 corrupt-data, 4 version-mismatch, 5 unsupported, 6 determinism-unavailable,
7 internal and 8 I/O. Codes are 1 identity, 2 lifecycle/busy,
3 selector/configuration, 4 bounded arithmetic, 5 shape/alias/topology,
6 checksum/digest, 7 allocation, 8 read, 9 publication, 10 capability,
11 stale binding, 12 invariant, 13 floating-point control and 14 readiness.
The trusted source maps categories 1..6 and 8 to the same-named public E1 class;
category 7, including code 7 allocation, maps to public `internal`. P1, C1 and K2
source exceptions remain their canonical E1 values and are captured as values;
they are never invented as numeric R triples. A native R bridge maps a dependency
failure once into this fixed R table before cleanup; malformed native triples map
to R internal/invariant. The I2 restore-copy bridge uses the existing I2 snapshot.
No cleanup overwrites either first snapshot.

Projection is exactly 256 bytes, all little-endian:

| Offset | Bytes | Value |
|---:|---:|---|
| 0 | 4 | structure bytes, 256 |
| 4 | 4 | profile identifier, 4 |
| 8 / 12 | 4 each | H / P0 |
| 16 / 20 / 24 / 28 | 4 each | mode / temperature bits / top-k / top-p bits |
| 32 / 36 / 40 | 4 each | max-new / EOS or `0xffffffff` / stop |
| 44 | 4 | zero |
| 48 / 56 / 64 / 72 | 8 each | RNG version / seed / counter-low / counter-high |
| 80 | 4 | complete history, first H bytes significant and the remainder zero |
| 84 | 4 | zero |
| 88 | 96 | exact raw T1 tokenizer fingerprint |
| 184 | 32 | execution-lane digest |
| 216 | 32 | complete capsule digest |
| 248 | 8 | exact nested C1 byte length |

The dependency-owned private seam
`c1-checkpoint-inspect-g3r-bytes-internal` arity 2 returns one immutable
three-element projection: the existing ten-slot C1 metadata, a vector of exactly
fifteen descriptor records in file order, and the exact alias-group vector. Each
descriptor record has nine slots: path, kind, immutable shape vector, dtype,
device, payload offset, payload bytes, immutable 32-byte checksum and alias
ordinal (`-1` or `0`). It runs the existing complete C1 parser and checksum work,
adds no decoder callback or tensor owner, and uses the same single parser-local
guard as existing inspect. Its descriptor offsets refer only to the rooted input
bytevector and grant no authority. It is not public and no other profile may
consume it without a separate contract.

The trusted stage wrapper authenticates the exact policy and T1 entry/fingerprint,
allocates the cleanup ledger and enters its guard before calling measure. It then
allocates exact measured outputs, calls read, copies the
nested C1 range into an exact bytevector, runs the new private
`c1-checkpoint-inspect-g3r-bytes-internal` parser, and verifies all fifteen logical
entries, fourteen unique storages, one exact head/token tie, shapes, finite bits,
provider identity and checksums. Only after those checks does it enroll the stage
as `live`. No decoder, tensor owner, RNG or model exists yet.

Capability admission creates one genuine current K2 report in the same aggregate,
then one exact deterministic `storage.copy` request for each unique source shape.
Every `capability-require` result must be the current SHARED-R2 verified entry.
The trusted wrapper separately authenticates the current lane digest, revalidates
the exact T1 entry/fingerprint already stored in the live stage, stores the lane
root, and changes `live` to `admitted`. No serialized field selects a provider,
tokenizer or lane.

`g3r-stage-release-internal!` arity 1 is idempotent only for `dead`. It rejects an
active reconstruction cross-link, changes the stage to `dead` before fallible P1
owned-state release, clears the image/nested bytes/descriptors/report/requests and
large projection references, drains the exact decoded-state owner if present, and
preserves the first error. Cleanup defects after exact preflight are fail-stop.
`g3r-stage-consume-internal!` arity 1 accepts only an `admitted` stage linked to
its exact prepared reconstruction after decoded-state release. It preflights and
clears image, nested bytes, descriptors, report and requests without a provider
call, changes the stage to `consumed`, and preserves only the projection and
reconstruction cross-link until the no-failure tail marks the reconstruction
committed.

A snapshot changes `pending` to `live` only after all fifteen clones and metadata
transfer into its ledger. Detached adoption changes `live` to `consuming`, clears
the transferred envelope ledger and stores the exact adopted P1 state in snapshot
slot 9; release clears that cross-link and changes it to `dead`. Stage consumption changes
`admitted` to `consuming` during its complete preflight and to `consumed` only
after transient payload clearing. A failure before that final store restores
`admitted` for ordinary abort cleanup; it never exposes `consuming` outside the
outer guard.

## Exact SAVE snapshot and detached encoding

The only new snapshot-copy C declaration is:

```c
void *et_i2_private_g3r_snapshot_parameter_v1(
    void *ctx, int64_t logical_ordinal, void *exact_handle);
```

The private alias `i2-native-g3r-snapshot-parameter` has arity 3 and is referenced
only by trusted `i2-g3r-snapshot-parameter-internal`. It authenticates the exact
active C4 tuple, FULL pins, logical ordinal 0..14, its fixed C1 path-to-C4-owner
mapping, parameter identity and canonical handle, allocates one standard I2 owned
clone of the exact shape, and invokes the genuine
SHARED-R2 `storage.copy` row from the pinned source into that disjoint clone.
Failure destroys local allocation and publishes nothing. Success returns one
native owned-clone control which the I2 wrapper immediately encloses in the
ordinary authenticated carrier and release envelope. No bare tensor pointer is
returned to a public or cross-aggregate caller.

The two fixed head/token logical ordinals map to C4 unique owner index 10 and its
one canonical handle, but each invocation allocates a physically disjoint owned
clone. Every other ordinal maps one-to-one to its C4 unique owner.

SAVE authenticates idle generator/model/tokenizer, exact binding, populated
cache, policy and lane before mutation. Under `m3-call` it acquires the ordinary
C4 active tuple and copies logical ordinals 0..14 in order into the preallocated
snapshot ledger. It copies policy/RNG/history/H/P0/stop only after their binding
recheck. The head/token tie produces two independent clones of unique owner index
10. A prefix failure releases all recorded clones in reverse, drains pins/tuple,
and leaves model/generator/RNG unchanged. After all copies, ownership transfers
once into the enrolled live
snapshot; SAVE drains the complete active tuple and exits `m3-call` before codec
work or file I/O.

Detached construction forms fifteen fixed C1 records from those fifteen carriers
and the one exact head/token alias group. `state-dict-adopt-owned-internal!`
receives the complete fifteen-envelope release ledger once; before that transfer
the snapshot owns it, and afterward the P1 state does. Existing
`c1-checkpoint-encode-state-internal` produces the nested canonical
C1 bytevector. Source-private `g3r-image-build-internal` arity 3 receives only the
exact authenticated `consuming` snapshot with that adopted-state cross-link, the
nested bytevector and the fixed operation value; it
builds the complete outer bytes without I/O and completes snapshot slot 7 with
the nested length and capsule digest. G3-R allocates a separate 256-byte validation
projection, verifies the image with `g3r-native-image-validate`, and requires exact
agreement with the completed snapshot projection. It then extracts the validated
nested range and runs `c1-checkpoint-inspect-g3r-bytes-internal`, requiring the
same exact schema, tie, finite bits and checksums as LOAD. Neither validator writes
the expected snapshot projection, so the comparison is not self-fulfilling. It
releases the P1 state and snapshot, and only then invokes the existing atomic
writer. Publication and durability status retain the C1
same-directory temporary/fsync/rename/directory-fsync semantics.

`g3r-snapshot-release-internal!` arity 1 is idempotent for `dead`, accepts only
`pending`, `live` or `consuming`, marks dead and clears all model/generator/
tokenizer roots before releasing either its untransferred carrier ledger or its
adopted P1 state. It
continues sibling releases, reports only the first defect, and never destroys the
live model or generator.

## Exact restored-copy route

The restored constructor gains the previously withheld private alias
`g3c4-native-model-owner-create-restored`, arity 0, mapped exactly to
`et_g3c4_private_model_owner_create_restored_v1`. Only
`g3r-reconstruct-prepare-internal` may reference it. There is no public wrapper,
generic restored-model factory or caller-selected origin.

After stage admission and entry to `m3-call`, reconstruction enrolls its cleanup
ledger and pending model/provenance records before native owner creation. It builds
the same fixed P1/I2 module/handle graph as seeded construction, without invoking
an initializer. It then decodes the already-inspected nested C1 bytevector through
the existing owned C1 decoder and binds that exact state to the genuine I2 2.0
provider. The decoded state remains stage-owned and unreleased until every copy
finishes.

The only cross-owner copy addition is:

```c
int64_t et_i2_private_g3c4_restore_copy_one_v1(
    void *exact_owner, int64_t index,
    void *destination_parameter, void *destination_handle,
    void *source_tensor, int64_t source_role, void *source_handle);
```

Its alias `i2-native-g3c4-restore-copy-one` has arity 7 and is referenced only by
`i2-g3c4-restore-copy-one-internal!`. The Eshkol wrapper requires the exact active
R reconstruction/stage, canonical decoded-state entry, scoped P1 state borrow,
restored owner, index, destination I2 parameter carrier and canonical P1 handle.
The wrapper passes the authenticated source tensor pointer, fixed owned-clone role
3 and `source_handle == NULL`; the C bridge rejects any other role or nonnull
source handle. No Eshkol carrier vector crosses the C boundary. The C
bridge repeats C4 owner/index/membership/handle checks and proves the source is a
live I2 owned clone of the exact shape. It then calls the existing I2 copy builder
route exactly as follows: create count 1; set assignment 0 with destination role
1 and source role 3 plus their exact handles; prepare; abort on every create/set/
prepare failure for which a builder exists; and commit once. Prepared commit is
allocation-free, nonraising and full-write, automatically releases the plan and
retires the builder. A nonzero prepared-commit status is a fail-stop I2 defect.
Only after success does one nonraising store mark that restored-owner index filled.
Every precommit failure leaves the destination unchanged. Indices are accepted
exactly once in order 0..13. K2 SHARED-R2 admission proves capability; it is not a
substitute for this I2 mutation route.

The source mapping is the C4 unique-owner order. Index 10 consumes the decoded
carrier for canonical least-path `head/weight` after the decoded state and stage
projection have proved the exact two-member alias group and bit-equal
`token_embedding/weight` payload. The decoder owns fifteen physically disjoint
carriers, so the distinct token-embedding carrier is not copied into a second C4
destination and remains owned by the decoded state until release.
Each scoped state borrow ends in ordinary control immediately after its one copy.
A later failure may leave a copied prefix only inside the unpublished owner;
owner abort destroys the entire fourteen-parameter set. No ordinary
`module-load-state-dict!`, raw tensor copy, C2 owner token or byte-copy shortcut is
admitted. After bit 13 is recorded, the decoded state is released before replay.

## Pending generator and no-draw replay

The replay control is a private native owner with states `OPEN`, `ACTIVE`,
`PREPARED`, `COMMITTED`, and `ABORTED`. It owns the pending generator, restored
typed RNG, cache, logits, history, H/P0/stop, fixed scratch and embedded C4 pins.
It is address-stable and never appears as a public registry kind.

The exact C surface is:

```c
void *et_g3r_private_replay_create_v1(
    void *prepared_owner, void *projection_bytes);
int64_t et_g3r_private_replay_frame_begin_v1(void *replay);
int64_t et_g3r_private_replay_role_step_v1(void *replay, int64_t ordinal);
int64_t et_g3r_private_replay_frame_prepare_v1(void *replay);
int64_t et_g3r_private_replay_frame_commit_v1(void *replay);
int64_t et_g3r_private_replay_frame_rearm_v1(void *replay);
int64_t et_g3r_private_replay_prepare_v1(void *replay);
void *et_g3r_private_replay_generator_v1(void *replay);
int64_t et_g3r_private_replay_commit_v1(void *replay);
int64_t et_g3r_private_replay_abort_v1(void *replay);
```

The exact private aliases are `g3r-native-replay-create` arity 2,
`g3r-native-replay-frame-begin` arity 1, `g3r-native-replay-role-step` arity 2,
`g3r-native-replay-frame-prepare` arity 1,
`g3r-native-replay-frame-commit` arity 1,
`g3r-native-replay-frame-rearm` arity 1, `g3r-native-replay-prepare` arity 1,
`g3r-native-replay-generator` arity 1, `g3r-native-replay-commit` arity 1, and
`g3r-native-replay-abort` arity 1.

For R, `i2-g3c4-construction-prepare-eval-internal!` is the sole construction
prepare compositor. It invokes `module-construction-prepare-eval-internal!`
exactly once while both layers are open, then completes I2 preflight and changes
I2 to prepared. The matching I2 prepared seal and abort similarly invoke the
appropriate P1 prepared seal or open/prepared abort exactly once before changing
the I2 lifecycle and dropping I2 ledgers. R never calls the module operations a
second time.

Before create, the trusted wrapper completes that one I2/P1 prepare compositor,
native owner `prepare_seal`, all
root promotions and every fallible final-publication preflight. Create accepts
only that exact `PREPARED` restored owner linked from the active R context and the
exact stage projection. The trusted wrapper has already
authenticated stage identity; projection bytes alone grant no authority. Create
copies policy, RNG words and history into the native control, creates an
unpublished pending generator entry, and leaves actual cache/history empty.

Replay installs the ordinary exact C4 active tuple against the pending model and
generator, then performs exactly one frame over the first P0 IDs and H-P0 frames
over one successive ID each. `frame_begin` chooses the next fixed slice itself;
the caller supplies no ID, length, frame kind or position. Each frame admits
ordinals 0..20 once in the existing C4 numerical order. Within owner state
`ACTIVE`, each frame moves through `READY`, `OPEN`, `PREPARED`, then either
`NEEDS_REARM` or `COMPLETE`. Begin accepts only `READY`; role steps fill ordinals
in order; prepare accepts only a complete `OPEN` frame; and commit publishes only
into the pending native generator. A nonfinal commit changes to `NEEDS_REARM`.
Rearm clears only frame-local scratch and ordinal state, preserves committed
cache/history/logits/RNG and changes to `READY`; it is allocation-free and
nonraising. The final commit changes to `COMPLETE`, from which rearm rejects.
There is no sample call, G3-S invocation,
output reservation, decoder, public tensor/result or RNG advance. RNG words are
checked unchanged before and after every frame, including the exhausted sentinel.

After exactly H IDs in `COMPLETE`, replay restores the saved terminal latch, checks cache,
history, last logits, H/P0, policy, tokenizer fingerprint, value binding and RNG,
preflights all temporary destruction, ends nested views and pins, and clears the
active tuple. `replay_prepare` changes `ACTIVE` to `PREPARED`; from there
`replay_generator` returns the exact pending native generator. Replay commit is a
no-allocation/nonraising state transfer that detaches that generator and tombstones
the replay control. `replay_abort` is the single abort operation: it first
preflights the entire control, ends any active replay view, drains embedded pins,
clears active owner/model/generator tokens, then destroys every pending generator/
RNG/cache/logits/scratch owner and sets `ABORTED`. It is idempotent only for
`ABORTED` and fail-stops on a corrupt control after complete preflight. Eshkol
clears the pending generator entry only after that one call succeeds. Ordinary C4
`call_acquire` and live-model admission are unchanged.

After `replay_prepare` and all fallible final-publication and cleanup preflights,
one checked store changes the R reconstruction entry from `open` to `prepared`.
The wrapper then calls `g3r-stage-consume-internal!` before the final tail. Replay
and the preallocated provenance record already hold every required projection
value, so no later step reads the cleared image, nested bytes, descriptor graph,
report or requests.

`g3r-replay-run-internal!` is the sole trusted Eshkol loop. It uses the same one
reconstruction cleanup guard for the whole replay and adds no frame, token, role,
decoder or provider guard.

## Abort and no-failure publication tail

The reconstruction guard captures the first error as data. Cleanup occurs after
the protected body has returned its failure outcome; it does not nest an open
copy/replay guard around sibling release guards. Exact abort order is:

1. Preflight every recorded non-replay authority. If replay exists, invoke the one
   `g3r-native-replay-abort`; its internal preflight, active-tuple cleanup and
   pending generator/cache/logits/RNG/scratch destruction are indivisible at this
   layer.
2. Clear the pending generator entry after replay abort succeeds.
3. Invoke the single I2 construction abort compositor for its exact open/prepared
   state; it invokes P1 abort once, restores prior registries/counts and revokes
   all module/handle identities.
4. Abort the native restored owner, destroying all fourteen parameters and
   clearing `staged_c4_owner`.
5. Mark pending model, generator and provenance entries dead; clear their large
   roots and the preallocated result pair.
6. Release decoded C1 state and remaining cleanup ledgers; clear the reconstruction
   cross-link, mark the reconstruction aborted, then release the stage and mark it
   dead.
7. Rethrow the snapshotted first value through the accepted canonical-preserving
   runtime/M3 successor. A cleanup mismatch after preflight is a fail-stop defect,
   never a replacement recoverable error.

Native release operations return status and push no Eshkol handler. Any Eshkol
release guards needed to observe a provider defect execute sequentially after the
reconstruction capture guard has popped. A release status defect may construct the
one bounded E1 shell counted below; cleanup captures it without replacing an
earlier primary error and continues sibling release. A post-preflight invariant
mismatch remains fail-stop.

Successful preparation has already allocated/promoted the result pair, registry
entries, provenance bytes and every cleanup node; released the decoded state and
stage image; prepared P1/I2 and owner; completed replay; and preflighted all
temporary destruction. The final tail is exactly:

1. `i2-g3c4-construction-seal-prepared-internal!`, which invokes the P1 prepared
   seal exactly once;
2. native owner `commit_seal`;
3. `g3r-native-replay-commit`;
4. assignment-only model-entry `live`, then generator-entry `live`, then
   provenance-entry `live`, then reconstruction `committed`, then clear the
   consumed-stage reconstruction cross-link;
5. return the already rooted pair.

This tail allocates nothing, invokes no provider/callback/parser/promotion, takes
no recoverable branch and performs no file I/O. A trusted invariant defect
fail-stops. Neither pending shell is reachable by an untrusted caller before the
tail returns.

## Exact handler reservations

Both public operations first enter their one existing outer `m3t-boundary`, then
call `(g3c4-native-reserve-exception-handlers 3)` before allocating a cleanup
ledger or publishing snapshot/stage authority. This count covers additional
simultaneous pushes after that call:

- the one outer R snapshot/stage owner guard; and
- the two handlers installed by the later `m3-call`.

Stage inspection has at most one C1 parser-local guard nested under the R stage
guard, K2 admission adds none, and native measure/read/write calls add none. SAVE
calls `(g3c4-native-reserve-exception-handlers 3)` as the first `m3-call` body
action before its C4 call-capture guard, tuple, pins or first clone. This exact R
topology would supersede the C4 predecessor's reserve-1 SAVE placeholder upon
root acceptance; an implementation never performs both reservations. A call
failure is captured as data and the call-capture guard pops before reverse clone
release. That release can nest exactly one item guard, one provider-callback guard
and one E1 shell-construction guard. The cumulative outer 3 plus SAVE 3 leaves six
pooled frames; after `m3-call`, only the R owner guard remains active, leaving the
five frames required by public P1 state release. Detached-adoption dual failure
peaks at five total future pushes from the outer call: R owner, P1 adoption
cleanup, item, provider callback and E1 shell. Codec work does not exceed it.

LOAD enters `m3-call` while the admitted stage guard is already active. As the
first body action it calls `(g3c4-native-reserve-exception-handlers 6)` before the
reconstruction guard, pending entries, restored owner, decoded state, copy,
generator or pins. The exact six-frame peak is the decoded-state ownership-
transfer dual-fault path:

1. R reconstruction cleanup;
2. C1 decoded-ownership cleanup;
3. post-transfer P1 adoption cleanup;
4. one release-ledger item guard;
5. the provider callback guard; and
6. E1 shell construction for a release-status defect.

The construction-registration peak remains four: reconstruction, I2 registration,
P1 `construction-call` and one of the two sequential P1 registration guards.
Public decoded-state release while reconstruction remains protected also peaks at
six: reconstruction plus the public, state, item, provider callback and E1 shell
guards. One-at-a-time restore copy and guard-free replay peak below six. Sibling
rollback releases are sequential, but the item/callback/E1 nesting within one
release is included. Final publication pushes no handler. Any change to parser
placement, owned decoder, registration, copy batching, replay or cleanup nesting
requires a contract amendment and a fresh high-water derivation. Revised direct
P1/C2/LOAD counts remain separate evidence and are not reused here.

Reservation failure occurs before authority in its scope. Persistent handler-
allocation-failure tests must exercise outer reserve-3, SAVE reserve-3 and LOAD
reserve-6, every deepest path, cleanup, and unchanged canonical delivery through
all M3 normalizers. This contract uses root's accepted narrowed source-private
function modifier, spelled `:runtime-emergency-rethrow-param <formal>`, for the
canonical check required by the C4 contract: a
top-level fixed-arity definition, one selected unannotated formal, compiler-
proved exact tagged-value type, and an entry spill/store plus the existing
canonical check before any allocating prologue or body action. Nested, internal,
lambda, extern, include-reached, variadic, duplicate and malformed uses reject,
and no function name is privileged. Annotated definitions disable self and mutual
tail-call elimination so every physical entry executes the check. The current
transformer pin still lacks that accepted mechanism and therefore does not satisfy
the implementation dependency.

## Implementation and acceptance hold

Implementation remains blocked until root accepts this contract, the final
compiler/runtime and M3/M3T canonical-error successor, C4 model/transport,
SHARED-R2, the final deterministic-lane manifest and dependency owner changes to
P1/I2/C1 private trusted surfaces. Exact implementation must then freeze the R
header/source/object list, every Eshkol source-closure member, defined/undefined
native symbols, package facade/export/global counts, and supported artifact hashes.

Required focused evidence includes every stage/decoder/copy/replay/abort fault
cut; exact release and tombstone counts; forged/stale/cross-owner tuples; mutated
stage/projection/state/carriers/handles; all reserve high-water and persistent-
failure cases, including cleanup-status defects after the reconstruction capture
has popped; no-draw replay of every P0/H partition; fresh-process exact
continuation; SAVE self-parse; old-destination preservation and publication-
unknown I/O; retained model/provenance growth; supported sanitizers; and two fresh
source-composed AOT artifacts. No numerical, sampler, package, CI or acceptance
claim is made by this design document.
