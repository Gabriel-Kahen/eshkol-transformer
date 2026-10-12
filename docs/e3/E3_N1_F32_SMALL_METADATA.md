# E3 N1 f32 small OCI metadata acquisition candidate

Status: **SOURCE-only, inert; HOLD**. The reported
`eshkol-f32-quantum-llvm21:20260925` index digest is
`sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
It has not been verified in this work. The
[index request skeleton](eshkol-f32-quantum-llvm21-20260925.small-index-request.json)
leaves source path/identity and limits `null`. The selected manifest/config
descriptors are unknown. No f32 metadata or layer byte was read.

`tests/e3_private/e3_n1_small_metadata.py` implements one read-only small-JSON
stage per invocation. The exact 2,743-byte
[proposed index request](eshkol-f32-quantum-llvm21-20260925.proposed-index-request.json)
is now compiled as `REVIEWED_REQUEST_SHA256 =
sha256:f0835b216f2d3f2594e31069b70338219865c961fff0ece7a8c3562d8d75316e`.
This is a **request-source pin only**. `REVIEWED_CAPTURE_SHA256 = None` keeps
the production CLI and direct production `acquire` call at
`HOLD: capture-contract-pin-absent` before opening any request source. No CLI
flag, request field, or caller-supplied digest bypasses either gate. Internal
tests inject a different,
synthetic index digest and tiny files; that override rejects the actual f32
digest and is not exposed by the CLI.

## Separate stage gates

1. **Index.** The pinned canonical request freezes the reported
   f32 index SHA-256, expected raw size, absolute source path, exact resolved
   regular-file identity, every ancestor directory identity, positive caps,
   and source review pin. Once a separate capture procedure is reviewed and
   pinned, the stage can check the raw index SHA-256, version/media type,
   and unique exact `linux/amd64` manifest descriptor. It does not read that
   manifest. The reported digest alone is insufficient to supply the path or
   authorize a read.
2. **Manifest.** Review and externally preserve the complete index evidence
   first. A new exact request must pin the prior evidence file SHA-256, prior
   raw index SHA-256, selected manifest descriptor and its source path and
   identity. The stage checks the prior evidence schema, checksum, raw size and
   hash, recorded regular-file size and directory route, and prior byte cap;
   it recomputes the selected descriptor from the preserved raw index
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
contracts still control any later production use. Before enabling a real index
read, a separate procedure must define and independently review bounded
stdout/stderr capture, a 30-second end-to-end deadline (including request
read), a 268,435,456-byte current-RSS cap, an 8-MiB evidence-output ceiling,
failure handling, and external preservation of the complete canonical HOLD
evidence. Its exact reviewed contract SHA-256 must be compiled as the distinct
capture pin. This source writes only stdout after a successful acquisition;
it creates no evidence file, immutable snapshot or custody authority.
The separate [one-use capture candidate](E3_N1_F32_SMALL_CAPTURE.md) now
implements a synthetic-tested HOLD capture procedure, but its source,
destination and custody have not passed that review gate. Its in-process
timer and direct-child RSS observation also leave the external hard deadline
and whole-procedure current-RSS proof open. Its public run gate is false. The
capture pin
therefore remains `None`.

Each synthetic success emits canonical, checksummed
`e3-n1-small-metadata-v1` **HOLD** evidence containing bounded base64 raw
bytes, exact raw hash/size, read-time identity and stage claims. Its encoded
size is capped at 8 MiB so the next stage can read it; OCI JSON input is capped
at 2 MiB, current RSS at 2 GiB, and stage time at 60 seconds as generic source
ceilings. The pinned f32 request narrows those to 4,096 source bytes,
268,435,456 current-RSS bytes and 30 total seconds. `/proc/self/statm`
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
