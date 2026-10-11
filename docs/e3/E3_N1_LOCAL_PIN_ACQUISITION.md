# E3 N1 local image metadata and host pin acquisition

Status: **read-only partial inventory; N1/M1 and import admission HOLD**. The
local containerd content store has a blob named by F's f31 image ID. Its
verified bytes are an OCI **index**, not an image config. The index selects
one linux/amd64 manifest; that manifest selects a distinct config and two
gzip layers. The small index, manifest and config bytes are sealed in the
separate `e3-section9-n1-local-index-host-pins-20261011-a` evidence directory.
The two layer descriptors and local sizes are recorded, but their large bytes
and uncompressed diff IDs were not checked. This content store is not a
reviewed OCI-layout export, and no image-selected guest file has been proven.

The bounded successor corrects `scripts/e3-n1-raw-capture.py` to authenticate
F's ID against the OCI index SHA-256 and select the sole linux/amd64 manifest.
The prior config-digest comparison could not accept this authentic index.
The scanner still deliberately rejects gzip layers; capture remains disabled,
and `REVIEWED_PLAN_SHA256` remains `None`. A focused synthetic test covers an
index whose config has a different digest and rejects a config digest used as
the image identity.

The evidence inventory also seals local host candidate `readelf`, `objdump`,
`addr2line`, Python, loader and libc bytes; modes, resolved paths and sizes;
the five unique F mount roots and all seven historical F host/guest routes;
and four extant Eshkol header copies. These are **host candidates/copies**,
not proof of f31 guest tool, provider, loader or header bytes. The original
F `/out` route is historical and must not be reused for capture.

Still missing: a reviewed full OCI export or bounded gzip streaming scanner;
verified image-selected guest bytes, symlink and provider closure; actual
resolved loader observation; complete system-header trace; exact per-object
compiler argv/environment; final optimized `puts` producer and per-call ABI.
F's launch script gives the outer Docker command and selected environment,
not the compiler's exact per-object argv/env. A later source-reviewed capture
plan must pin those inputs, preserve F object/depfile parity or declare a new
candidate, and keep capture disabled until its executor and limits are
independently reviewed. An accepted raw receipt would still precede separate
tuple-scoped import-policy review and a fresh full Numerical F gate.
