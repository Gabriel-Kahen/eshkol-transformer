# E3 N1 f32 small OCI metadata acquisition candidate

Status: **SOURCE-only, inert; HOLD**. The reported
`eshkol-f32-quantum-llvm21:20260925` index digest is
`sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
It has not been verified in this work. The
[index request skeleton](eshkol-f32-quantum-llvm21-20260925.small-index-request.json)
leaves source path/identity and limits `null`. The selected manifest/config
descriptors are unknown. No f32 metadata or layer byte was read.

`tests/e3_private/e3_n1_small_metadata.py` implements one read-only small-JSON
stage per invocation. Its public CLI has `REVIEWED_REQUEST_SHA256 = None` and
returns `HOLD: production-request-pin-absent` **before opening any request
source**. No CLI flag, request field, or caller-supplied digest bypasses that
gate. A later source-reviewed exact request SHA-256 must be compiled in before
real acquisition can be considered. Internal tests inject a different,
synthetic index digest and tiny files; that override rejects the actual f32
digest and is not exposed by the CLI.

## Separate stage gates

1. **Index.** A separately reviewed canonical request must freeze the reported
   f32 index SHA-256, expected raw size, absolute source path, exact resolved
   regular-file identity, every ancestor directory identity, positive caps,
   and review pin. The stage checks the raw index SHA-256, version/media type,
   and unique exact `linux/amd64` manifest descriptor. It does not read that
   manifest. The reported digest alone is insufficient to supply the path or
   authorize a read.
2. **Manifest.** Review and externally preserve the complete index evidence
   first. A new exact request must pin the prior evidence file SHA-256, prior
   raw index SHA-256, selected manifest descriptor and its source path and
   identity. The stage checks the prior evidence schema, checksum, raw size and
   hash, and recomputes the selected descriptor from the preserved raw index
   bytes before reading only the pinned manifest JSON. It records
   the config descriptor and ordered layer **claims**; it does not read config
   or layers.
3. **Config.** Review and externally preserve complete manifest evidence
   first. A new exact request must pin that evidence hash, config descriptor
   and source identity. The stage rechecks the manifest evidence/selection,
   then reads only the pinned config JSON. It records `linux/amd64`,
   `rootfs.type=layers`, and ordered diffID **claims** matching layer count.
   It recomputes the config and ordered layer descriptors from the preserved
   raw manifest bytes. It does not verify any compressed layer, expanded diffID, unpacked rootfs,
   selected guest path or loader/provider behavior.

Each source read walks absolute path components with `O_DIRECTORY|O_NOFOLLOW`,
compares caller-frozen file and ancestor device/inode/mode/owner/size/timestamps
to opened fds, caps bytes/time/current Linux resident memory, and rechecks the
named route after reading. The result is a **read-time identity**, not
immutable custody: a later same-inode write or replacement remains possible.
The separate export broker, protected immutable handoff, receipt and seal
contracts still control any later production use.

Each synthetic success emits canonical, checksummed
`e3-n1-small-metadata-v1` **HOLD** evidence containing bounded base64 raw
bytes, exact raw hash/size, read-time identity and stage claims. Its encoded
size is capped at 8 MiB so the next stage can read it; OCI JSON input is capped
at 2 MiB, current RSS at 2 GiB, and stage time at 60 seconds. `/proc/self/statm`
is parsed fail-closed per read chunk for current RSS, avoiding inherited
high-water `ru_maxrss`. These are source limits, not f32 measurements. The
tool publishes no file or snapshot. A later reviewed procedure must capture,
seal and retain raw evidence outside this tool before allowing a successor
stage; a path/hash supplied by a caller is only a claim until independently
reviewed.

This candidate does not modify the full-layer
[OCI verifier](E3_N1_OCI_EVIDENCE_SOURCE.md), the
[f32 export-input/custody plan](E3_N1_F32_EXPORT_INPUT_PREP.md), or any
broker/sealer authority. Its tests use only generated small JSON files and
missing layer paths. No Docker, export, build, marker, key, guest observation
or production acquisition occurred.
