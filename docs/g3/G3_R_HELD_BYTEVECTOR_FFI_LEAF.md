# G3-R held bytevector copy prerequisite

This feature-gated native leaf adds an Eshkol bytevector destination to the
[held same-owner capture](G3_R_HELD_SAVE_CAPTURE_LEAF.md). It does not create
an Eshkol hold adapter, a P1/O2/control bridge, a C2 owner or a SAVE writer.
The existing typed `hold_begin`, `hold_copy` and `hold_end` APIs are unchanged.

```c
int64_t et_g3t_private_committed_save_hold_copy_bytes_v1(
    int64_t hold, void *destination_header);
```

The source-private entry requires `ET_G3R_COMMITTED_SAVE_HOLD_PRIVATE`. The
header must name one exact 4,872-byte Eshkol bytevector: 136 bytes of policy,
four RNG words, ordered history and partition followed by 4,736 bytes of the
same pinned 14-parameter image. It admits the live monotonic hold token,
checks the complete header and data span against the generator, model, every
live hold and A2/F32/I64 storage, and calls the typed hold copy into local
staging. Only a successful check copies all 4,872 bytes to the caller's
bytevector. Rejection leaves that bytevector unchanged; the hold remains
live until its owner calls `hold_end`. The caller supplies valid, writable,
disjoint bytevector storage and serializes this source-private API.

`scripts/test-g3r-committed-save-snapshot.sh` runs the controlled native
normal, repeat and ASan/UBSan cases plus feature-off symbol closure. It
checks exact-size admission; short, long and stale-token rejection; complete
header overlap with a live hold; A2/F32/I64 overlap; model-bit drift; and
valid RNG drift without partial destination writes. The native gate cannot
prove an Eshkol compiler's FFI bytevector layout, regional lifetime or
aggregate linkage. No supported compiled Eshkol toolchain is present in
this worktree, so no Eshkol adapter or callback was added.

The next bridge must authenticate a live owning aggregate that connects the
exact G3-T generator and M3T model to the P1 capsule, O2 optimizer, T1 root,
resolved controls and dataset cursor. This lineage currently has no such
G3-to-trainer edge. A future adapter must retain the native token through
that capture, recheck all Eshkol registry identities and native baseline
before release, end the hold exactly once on every path, and establish a
compiled aggregate witness before any C2 ownership transfer.
