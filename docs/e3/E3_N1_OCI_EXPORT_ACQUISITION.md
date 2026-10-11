# E3 N1 bounded OCI export acquisition candidate

Status: **inert source candidate; no production release pin, layer read, or
approved export**. `scripts/e3-n1-oci-export-acquisition.py` hardcodes the F
index, manifest, config, ordered layer descriptors and config diff IDs from
the accepted exact-input plan. `RELEASED_PIN_SHA256` remains `None`, so its CLI
returns HOLD before opening a pin or any layer. Installing that constant and
running a real acquisition require a separate immutable metadata-only pin and
independent plus ROOT review of the exact source, pin, destination and limits.

The release pin is a JSON object with exactly `schema: 1`, `small`, `layers`,
`sources`, `destination` and `limits`. `small` has `index`, `manifest`, and
`config` records. Each records absolute `path` and `resolved_path`, full
`lstat` mode, byte size, device, inode, nanosecond mtime/ctime and SHA-256.
`layers` is the two ordered OCI descriptor objects. `sources` is the two
ordered candidate records with the same path/identity metadata but **no
asserted byte hash**; their digest is verified only during acquisition.
`destination` records a fresh absolute target, its absolute parent, parent
mode/device/inode and measured free bytes. `limits` contains positive
`seconds`, `max_rss_bytes`, `max_disk_bytes`, `max_blob_bytes` and
`max_combined_bytes`. The pin file itself is at most 64 KiB, exact-byte
SHA-256 matched to the source release constant, and cannot have a symlink in
its path.

Before a layer read, the source checks the small file identities and exact
bytes, parses their OCI index-to-manifest-to-config chain, checks the sole
`linux/amd64` descriptor and the two ordered gzip descriptors, checks source
candidate identities, destination freshness and parent/free-space facts, and
rejects symlinks in every opened source path component. The hard ceilings are
512 MiB per compressed blob, 512 MiB combined, 520 MiB staged disk bytes,
1,800 seconds and 2 GiB peak process RSS. The pin must budget at least the
expected files plus 64 KiB for staging and failure records. No gzip bytes are
decompressed or executed.

Each candidate is opened with `O_NOFOLLOW`; pre/post `lstat` and `fstat`
identity, size and timestamps must match the pin. The source copies each blob
once into a private `.part` regular file while counting and SHA-256 hashing
every written byte. Only after both exact compressed sizes and hashes pass
does it create the OCI layout and rename the private directory to a fresh
target using Linux `renameat2(RENAME_NOREPLACE)`. Any staged failure writes
`HOLD.json` and `SHA256SUMS-ACQUISITION` beside the retained partial files.
A successful directory contains `ACQUISITION.json` and the checksum list.
This is an acquisition artifact only; its read-only evidence seal and guest
selection are separate gates in the exact-input plan.

Generated tiny-blob tests exercise exact publication, release absence,
malformed pins, byte and disk caps, symlink and inode swaps, mutation during
read, late time HOLD and second-blob hash failure. They never open an f31
layer, invoke Docker, run a guest tool, scan gzip or authorize capture.
