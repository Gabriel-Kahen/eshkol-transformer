# E3 N1 f32 OCI index acquisition proposal

Status: **PROPOSAL only; real index acquisition HOLD**. Base
`7238fb8fc356a5f23ce480bc249f2a6c7d00e3fc` (tree
`b5b9d5149e7a3231d0f0915f3f42783d329d13f8`). This document and the
[canonical proposed request](eshkol-f32-quantum-llvm21-20260925.proposed-index-request.json)
freeze a metadata-only candidate for the **index stage** of the existing
[small-metadata source](E3_N1_F32_SMALL_METADATA.md). The request's exact
2,743 bytes have SHA-256
`f0835b216f2d3f2594e31069b70338219865c961fff0ece7a8c3562d8d75316e`.
That is the **request hash**, not a verified index hash or an approval pin.
`REVIEWED_REQUEST_SHA256` stays `None`; the CLI remains disabled.

## Frozen candidate

- Reported image: `eshkol-f32-quantum-llvm21:20260925`; expected index digest:
  `sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
  Its 855 bytes were **not opened or hashed** in this proposal.
- Absolute candidate source: `/home/gabe/.local/share/docker/containerd/daemon/io.containerd.content.v1.content/blobs/sha256/f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
  Metadata-only `lstat` observed a regular file: device 55, inode 29376658,
  mode 33060, UID/GID 1000/1000, size 855, mtime_ns
  1790329249210991147, ctime_ns 1790329249210387186. The request contains
  those exact fields and the full `lstat` chain from `/` through `blobs/sha256`;
  every observed ancestor was a directory. No symlink was followed.
- Stage is `index`, with `prior` and `selection` null. The proposed caps are
  4,096 source bytes, 30 elapsed seconds, and 268,435,456 current RSS bytes.
  They are predeclared limits, not measured acquisition usage. The selected
  amd64 manifest and config descriptors remain unknown.

The source-backed reason for this bounded step is that
[E3 section 9](../E3_EVALUATION_PROPOSAL.md#9-required-references-failure-and-resource-proof)
requires supported, exact-input numerical and resource evidence, while the
[roadmap](../ROADMAP.md) records the f32 image only as a reported index digest
and keeps its descriptor chain and real bytes HOLD. The small-metadata source
provides a one-JSON-at-a-time index stage that can verify this reported digest
and discover the unique `linux/amd64` descriptor before any manifest or layer
read. This proposal supplies its missing path, exact source identity and finite
limits for ROOT review; it supplies no numerical proof itself.

## Review gate and limits of evidence

ROOT must review this exact request hash, source path and complete identity
chain, byte/time/RSS caps and current host resource condition **before** any
real index read or source pin change. Revalidate every path component and file
identity immediately before opening under the existing nofollow source path;
reject drift and re-review a new request hash. The same-UID Docker content
store and home ancestors are writable, so an `lstat` snapshot is neither
immutable custody nor proof of the digest. Directory sizes and timestamps may
drift during unrelated host activity. The name's digest is an expected claim
until the capped read independently verifies the 855 bytes. A later read-time
match would still require external preservation and independent review of its
complete canonical evidence before a separately pinned manifest request.

No real index, manifest, config, metadata or layer bytes were read for this
proposal. It creates no evidence seal, receipt, export, marker or custody
authority; it does not enable N1/M1, a supported numerical run, or any
compiled `REVIEWED_REQUEST_SHA256`.
