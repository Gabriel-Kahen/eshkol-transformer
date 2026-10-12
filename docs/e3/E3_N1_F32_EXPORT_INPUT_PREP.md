# E3 N1 f32 export input and custody preparation

Status: **SOURCE/PREP only; HOLD**. The sole candidate here is the reported
`eshkol-f32-quantum-llvm21:20260925` OCI **index** digest
`sha256:f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
This is a metadata-only report, not a verified local index read. The
[candidate input JSON](eshkol-f32-quantum-llvm21-20260925.export-input.json)
has explicit `null` for the unknown `oci-layout` size/hash, index size,
selected manifest, config,
ordered layers/diffIDs, source path pins, custody and limits. It is incomplete
and HOLD. No f31 descriptor or failed-F parity claim transfers to this image.

`scripts/e3_n1_export_input.py` accepts one canonical UTF-8
`e3-n1-export-input-v1` JSON file and emits a canonical, checksummed
`e3-n1-export-input-assessment-v1` line to stdout. Every output has
`status: HOLD`. `INCOMPLETE` lists missing top-level slots;
`READY_FOR_REVIEW` means only that supplied fields pass structural checks.
The only accepted incomplete form has every unknown slot `null`; a partly
populated plan is rejected until all supplied claims can be checked together.
Malformed, noncanonical, duplicate-key, inconsistent or over-cap input exits
nonzero with `HOLD:` and no assessment. The CLI reads only its input file. It
opens that input through a nofollow walk of every path component and compares
pre-open, opened and post-read file identity. It does **not** open source paths,
image blobs, output stores, evidence refs or
keys. Capture and seal stdout separately if it is used in a later review.

## Required pre-read input

The complete plan must freeze exact `oci-layout` expected SHA-256 and size,
the index's raw digest and size, a unique
`linux/amd64` selected manifest descriptor, its config descriptor, ordered
layer descriptors, and ordered config diffID claims. Descriptors carry exact
media type, `sha256:` digest and size; uncompressed tar's descriptor digest
must equal its diffID. These are **expected OCI claims** until reviewed raw
index/manifest/config bytes and later full layer bytes prove them. The
validator cannot discover a descriptor from an image tag or infer a config
ImageID from the index digest. It does not derive or report a verified
ChainID; use the separate [OCI byte verifier](E3_N1_OCI_EVIDENCE_SOURCE.md)
after an independently admitted export.

`source_pins` lists the proposed `oci-layout`, `index.json`, manifest blob,
config blob and ordered layer blobs in that exact export order. Each item has
an export-relative name, source absolute/resolved path, regular-file mode,
UID/GID, device/inode, full absolute ancestor directory chain with modes,
owners, device/inode and ACL evidence digests, and **pre-read `lstat_size`**.
Each source also carries an expected SHA-256 equal to the `oci-layout`, index
or blob digest in the frozen chain. The reported lstat size must match the
expected size. No layer
source hash is recorded as measured before the read; descriptor digests are
expected values that a future broker must check while streaming. A matching
path, inode or size does not authenticate source bytes or prevent later drift.
The plan rejects duplicate blob names and source paths.

Positive bounded inputs include per-blob and combined byte caps, output cap,
seconds, RSS, and measured free space of at least twice the output cap. The
proposed ceilings are 512 MiB per blob and combined, 2 GiB output/RSS and
1,800 seconds. These are structural ceilings, not measured workload results.
An exact complete plan needs separately reviewed positive limits chosen for
the actual f32 descriptors and source filesystem.

`custody` requires distinct caller, broker and sealer UIDs; broker-owned
protected output, intent, receipt and one-use stores; separately sealer-owned
seal store and immutable-target parent; named ancestor identities and ACL
evidence; exact target basename; a unique broker-owned stage and durable
pre-stage HOLD policy. It requires two distinct role-specific PureEd25519-v3
**public** verification anchors, their raw 32-byte public keys/digests,
verifier-binary path pins with expected size and SHA-256, and separate
trust-evidence references. It also requires
evidence references for UID/ACL, mount namespace, `/proc` and fd isolation,
one-use protocol, receipt issuance, seal custody and immutable handoff. These
references are supplied claims, not checked evidence; no private key, real
nonce, prepared marker or production trust anchor is stored in this candidate
file. The [synthetic broker](E3_N1_BROKER_CANDIDATE.md) uses HMAC and cannot
satisfy this production authority contract by itself.

## Later acquisition and review sequence

1. Independently acquire and review the exact small OCI index, selected
   manifest and config bytes through an approved read-only procedure; freeze
   their descriptor chain without reading a real layer under this source task.
   Record finite source `lstat` identities, protected destinations, authority
   evidence and limits in a new complete canonical input. Review its digest.
2. Separately review and deploy the [distinct authority custody contract](E3_N1_SEAL_AUTHORITY_CONTRACT.md):
   actual UID/ACL/ancestor, namespace, `/proc` and fd isolation; independent
   receipt/seal verification anchors; broker and sealer binaries; immutable
   target handoff; and one-use path. The plan validator cannot prove these.
3. Only under that reviewed release and broker procedure may a broker durably
   write the protected pre-stage HOLD intent and stage marker **before** its
   first layer read. It must stream each source once under caps into private
   files, compare ordered size/hash to expected descriptors, reject source
   identity drift, sync, and use no-replace publication. Any uncertainty stays
   broker-owned HOLD/INDETERMINATE. Readable target or VERIFIED bytes do not
   release it.
4. An independently checked durable broker receipt and returned one-use token,
   immutable target custody and separate seal authority are required before
   a consumer can use the export. Then run the source-reviewed offline OCI
   byte verifier on the exact admitted target, verify expanded diffIDs and
   derived ChainID, and separately review finite guest path discovery and
   observation/capture inputs. N1/M1 and F parity still require their own
   later gates.

The only tests here create **synthetic metadata** and malformed-plan
mutants. They do not create an OCI export, read f32 blobs, launch Docker or a
build, issue a receipt/seal, use keys, or create a one-use marker.
