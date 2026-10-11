# E3 N1 bounded gzip selected-byte scanner

Status: **source-only candidate; capture disabled; N1/M1 and import admission
HOLD**. This successor extends the inert preflight's OCI reader for
`application/vnd.oci.image.layer.v1.tar+gzip`. It streams one gzip member into
the tar selected-file scanner without writing an expanded layer. The pinned
compressed descriptor SHA-256 and size are checked before scanning and again
as the stream is consumed. The expanded byte SHA-256 must equal the config's
uncompressed diffID. Missing footer, corrupt checksum, concatenated members
and trailing compressed bytes fail closed.

The reviewed plan must add positive `max_uncompressed_layer_bytes` and
`max_total_uncompressed_bytes` caps, bounded by 8 and 16 GiB respectively.
Decompression releases at most one fixed chunk at a time and rejects output
past the smaller remaining cap. Existing selected-byte, time, RSS, whiteout,
symlink, mode and path checks apply to gzip tar members. Zstd remains
unsupported. All requested guest entries must still be selected and pinned;
the scanner does not prove a full image overlay or dynamic-loader resolution.

Ten focused synthetic tests pass, including an OCI index→manifest→gzip layer
path and compressed hash, diffID, short, corrupt, trailing, concatenated,
decompression-cap, time, selected mode, whiteout and symlink negatives. These
tests use tiny generated layers. **Neither real f31 gzip layer was opened or
hashed for this change.** No reviewed plan SHA has been installed; the CLI
still rejects `--capture`, emits HOLD and never runs Docker or a target.

Next gate: independent and root source review of this scanner, then an exact
reviewed OCI export and selected guest input plan. Only after separate capture
runner/resource review may a bounded real layer scan be considered. A raw
receipt still needs semantic producer, ABI, provider, loader and header review
before any import-policy change or fresh full Numerical F gate.
