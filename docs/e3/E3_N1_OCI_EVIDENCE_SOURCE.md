# E3 N1 offline OCI byte-evidence candidate

Status: **SOURCE only; HOLD**. `tests/e3_private/e3_n1_oci_evidence.py` is a generic,
read-only offline verifier. It does not pull an image, run Docker, extract a
layer, launch a build, create a marker/key/release, or grant N1/M1 admission.
Its tests generate tiny OCI layouts. Neither of the two local f31 gzip layer
candidates has been read or verified by this work; their existing inventory is
metadata only. The full f31 export, guest observations, complete input manifest,
independent seal and custody gate remain HOLD.

The source follows [OCI image-spec v1.1.1 image layout](https://github.com/opencontainers/image-spec/blob/v1.1.1/image-layout.md)
and [config](https://github.com/opencontainers/image-spec/blob/v1.1.1/config.md).
The CLI requires a canonical UTF-8 JSON input with one trailing newline,
unique keys, and exactly these fields:

```json
{"index_digest":"sha256:<64 lowercase hex>","layout":"/absolute/nofollow/layout","limits":{"max_blob_bytes":1,"max_evidence_bytes":1,"max_expanded_layer_bytes":1,"max_rss_bytes":1,"max_seconds":1,"max_total_expanded_bytes":1},"observations":{"environment":{"records":[],"state":"UNMEASURED/HOLD"},"headers":{"records":[],"state":"UNMEASURED/HOLD"},"os":{"records":[],"state":"UNMEASURED/HOLD"},"packages":{"records":[],"state":"UNMEASURED/HOLD"},"paths":{"records":[],"state":"UNMEASURED/HOLD"},"tools":{"records":[],"state":"UNMEASURED/HOLD"}},"platform":{"architecture":"amd64","os":"linux"},"schema":"e3-n1-oci-input-v1"}
```

The numeric `1` values above show field shape only; they are not approved
resource settings or an executable f31 pin. The input is incomplete for N1
because every observation is explicitly `UNMEASURED/HOLD`. A category can
instead use `external-unverified` with nonempty records. Every such record has
exactly `subject`, `value`, and `evidence`; `evidence` has an absolute regular
file `path`, exact `size`, `sha256:...`, and declared `producer` and `method`.
The source hashes that file using nofollow path walks, then preserves the
claim and provenance without treating their semantics as measured. Missing
categories, records without their file, duplicate subjects, changed bytes,
symlinks, noncanonical input and unsupported media types HOLD. The input
manifest must be separately reviewed and sealed before any production use.

Run `python tests/e3_private/e3_n1_oci_evidence.py /absolute/input.json` against a
separately obtained local OCI layout. On success it prints one canonical
`e3-n1-oci-evidence-v1` JSON line to stdout, with `status: HOLD`, the exact
input SHA-256, descriptor chain, ordered compressed size/hash and expanded
diffID checks, and a checksum over the canonical payload excluding the
checksum field. Capture and independently seal stdout elsewhere; this CLI
does not publish an evidence file. Failure prints `HOLD:` to stderr, emits no
evidence and returns nonzero.

The verifier requires a unique exact requested OS/architecture descriptor
from `index.json`, a version-2 OCI manifest, a matching OCI config with
`rootfs.type=layers`, and exact ordered layer/diffID count. It streams regular,
nofollow gzip or uncompressed tar blobs under positive capped time, RSS,
compressed and expanded byte limits; the per-layer and combined expanded caps
are enforced on every output chunk. It requires UTF-8 OCI JSON and rejects
nonfinite numbers or malformed Unicode. It rejects short, extra-member and
trailing gzip input. The config digest is the OCI ImageID; the index digest
identifies the index and is kept separately. The output's `chain_id` applies
the v1.1.1 recursion to the **verified ordered diffIDs**; it is labelled
`derived-from-verified-diffids-only`. It does not prove unpacked filesystem
state, overlay effects, selected guest paths, runtime loader/provider
resolution, or the truth of external observations.

This source has no production pin, reviewed export custody, immutable snapshot
or authenticated seal authority. A caller-supplied layout or evidence path
alone is not an admission authority. The next gate is independent source/test
review and a separately reviewed, complete f31 input and export procedure
before any real layer read.
