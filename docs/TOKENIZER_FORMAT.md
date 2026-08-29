# Byte tokenizer format

Status: proposed T1 format contract, version 1.0. The public tokenizer runtime is
blocked on the dependencies listed below; this document does not provide substitute
runtime values, configuration objects, persistence policies, or errors.

## Semantics

The byte vocabulary is fixed: byte value `b` has token ID `b` for every
`0 <= b <= 255`. Byte tokens therefore occupy IDs `0..255` in that order.

Special tokens are explicit records with a name, ID, and decode policy. Their IDs
must uniquely and contiguously cover `256..(255 + special-count)`. Their names are
unique ASCII strings matching `[a-z][a-z0-9._-]{0,63}`. Records are serialized by
ascending ID, independent of their construction order.

The only version-1 decode policies are:

- `omit`: decoding the special token emits no bytes.
- `error`: decoding the special token is an `invalid-argument` error.

Encoding never recognizes a special-token spelling inside input. Special IDs enter
an encoding only through the configured prefix and suffix lists. Those lists are
ordered, may intentionally repeat an ID, and may reference only configured `omit`
specials. No BOS, EOS, padding, or other special is inserted by default.

Normalization is always `none`. The UTF-8 policy is one of:

- `raw`: accept and return byte strings without UTF-8 validation. Every byte
  sequence, including malformed UTF-8 and embedded NULs, is preserved exactly.
- `strict`: require valid UTF-8 on encode and after byte emission on decode. Invalid
  input is rejected; it is never replaced, ignored, normalized, or repaired.

For valid UTF-8, encoding operates on its original UTF-8 bytes and decoding returns
those bytes exactly. The format does not authorize Unicode normalization.

## Canonical bytes

An artifact is strict 7-bit ASCII TSV. Every field separator is one byte `09`, every
record ends with one byte `0a`, and no byte follows the checksum record. A BOM,
carriage return, trailing space, blank line, or extra final LF is noncanonical.

Version 1.0 has the following records in exactly this order. Metavariables in angle
brackets describe fields and are not literal file bytes.

```text
format<HT>eshkol-byte-tokenizer<LF>
version<HT>1<HT>0<LF>
required-features<HT>0<LF>
optional-fields<HT>0<LF>
limit-file-bytes<HT>1048576<LF>
limit-specials<HT>4096<LF>
limit-name-bytes<HT>64<LF>
limit-header-items<HT>4096<LF>
limit-optional-value-bytes<HT>64<LF>
limit-prefix<HT>4096<LF>
limit-suffix<HT>4096<LF>
payload-bytes<HT><PAYLOAD_BYTES><LF>
kind<HT>byte<LF>
normalization<HT>none<LF>
utf8-policy<HT><raw-or-strict><LF>
byte-ids<HT>0<HT>255<LF>
special-count<HT><N><LF>
special<HT><ID><HT><NAME><HT><omit-or-error><LF>  (N records)
prefix-count<HT><P><LF>
prefix<HT><INDEX><HT><ID><LF>                    (P records)
suffix-count<HT><S><LF>
suffix<HT><INDEX><HT><ID><LF>                    (S records)
checksum-algorithm<HT>sha256<LF>
checksum<HT><DIGEST><LF>
```

`<HT>` and `<LF>` above mean the single bytes `09` and `0a`. All integers use
unsigned base-10 ASCII with no sign and no leading zero, except that zero is exactly
`0`. Special records are strictly ascending by ID. Prefix and suffix indices are
exactly `0..count-1` in ascending order. The ID order of prefix and suffix records is
semantic and is not sorted.

`PAYLOAD_BYTES` is the byte count beginning with the `k` of the `kind` record and
ending with the LF of the last suffix record, or the LF of `suffix-count` when `S`
is zero. It excludes the envelope header, `checksum-algorithm`, and `checksum`. The
declared value must equal the observed count.

The count in `required-features` is followed by that many canonical
`required-feature<HT><NAME><LF>` records, sorted uniquely by name. The count in
`optional-fields` follows them and is followed by that many canonical
`optional-field<HT><NAME><HT><VALUE><LF>` records, also sorted uniquely by name.
Feature/field names use the special-name grammar above. An optional value is an even,
nonempty sequence of at most 128 lowercase hexadecimal characters and is
semantically inert to an older reader.

Both counts are zero in version 1.0, so none of those variable records is present in
a version-1.0 artifact. A version-1 reader rejects a major other than 1 with
`version-mismatch`. It may read a higher minor only after parsing the generic
records: any required feature is `unsupported`, while declared optional fields are
ignored. An extension that changes tokenizer semantics is required, not optional.
The fixed limit records are part of every version-1 envelope and may change only
with a major version decision.

## Checksum and fingerprint

Let `A` be the exact artifact bytes from the `format` record through the LF ending
the `checksum-algorithm` record. Let `P` be the exact payload bytes counted by
`payload-bytes`, and let `H` be the exact envelope header bytes from `format` through
the LF ending `limit-suffix`. `H` includes all declared required/optional records and
limits but excludes `payload-bytes`.

`DIGEST` is the 64-character lowercase hexadecimal SHA-256 digest of the exact byte
concatenation:

```text
"eshkol-byte-tokenizer-checksum-v1\n" || A
```

The final file bytes are exactly:

```text
A || "checksum\t" || DIGEST || "\n"
```

The tokenizer identity digest is domain-separated from the artifact checksum. It is
the lowercase SHA-256 digest of these exact bytes:

```text
"sha256:eshkol-byte-tokenizer-v1\n" || H || P
```

The public fingerprint has exactly this lexical form:

```text
sha256:eshkol-byte-tokenizer-v1:<64 lowercase hex>
```

Thus the identity digest input and output both identify the SHA-256/canonicalization
version. The digest covers the format identifier, major/minor version,
required/optional declarations, hard limits, normalization and UTF-8 policies,
fixed byte mapping, complete special table, and ordered prefix/suffix behavior. The
separately domain-separated artifact checksum also covers the declared payload
length and checksum algorithm. Loading and reserializing a valid version-1.0 artifact
must reproduce the original bytes.

## Limits and validation

Version 1.0 imposes these hard limits before construction:

- total file size: at most 1 MiB (`1,048,576` bytes);
- payload: at most 1 MiB and exactly `payload-bytes` bytes;
- special tokens: at most 4,096;
- special-token name: 1 to 64 ASCII bytes;
- required-feature and optional-field records: at most 4,096 each;
- optional-field value: 1 to 64 bytes encoded as 2 to 128 lowercase hex digits;
- prefix and suffix: at most 4,096 entries each;
- token IDs: nonnegative signed-64-bit values, additionally constrained by the
  fixed byte range and contiguous-special rule above.

A persistence policy may impose smaller limits but may not enlarge these version-1
limits. Readers bound the file and individual line lengths before allocation, then
validate ASCII/canonical bytes, record order and counts, version/features, checksum,
and finally semantic invariants. No partially validated tokenizer is exposed.

Duplicate or unknown records, duplicate names or IDs, gaps, byte/special collisions,
bad list references, invalid decode policies, noncanonical numbers/order, and
declared-size mismatches are rejected. Checksum mismatch, truncation, and malformed
artifact structure are `corrupt-data`; unsupported versions/features follow the
categories above. Files are inert data and are never evaluated as Eshkol code.

## Save visibility and durability boundary

A conforming save writes the complete canonical bytes to a uniquely named temporary
file in the target directory, closes it, and only then renames it over the target.
The replacement is atomically visible only where the host provides same-filesystem
atomic rename semantics. Before that rename the target is unchanged; a failed save
may leave a separate temporary file.

This is the required save direction, not current runtime acceptance evidence. Source
inspection of the pinned runtime found same-directory temporary-creation and rename
entry points, but T1 has not proved their behavior or atomic visibility through the
public save path. Current evidence also does not prove file or directory `fsync`,
crash durability, or persistence across power loss. Version 1.0 therefore makes no
current atomicity or durability claim. A later implementation must prove its
requested guarantee or report it unavailable.

## Runtime dependencies

The public T1 runtime remains blocked; this format document does not work around the
following contracts:

- E1, issue #23: shared structured public errors and accessors;
- I1, issue #24: dense contiguous `i64` token storage required by A0;
- X1, issue #20: validated tokenizer configuration values;
- C1, issue #19: persistence-policy values, bounded I/O, and save policy behavior.

Until those dependencies merge, there is no production `tokenizer-byte`,
`tokenizer-load`, `tokenizer-save!`, `tokenizer-encode`, or `tokenizer-decode`
implementation and no tagged-vector, alist, raw-list, or unstructured-error stand-in.
