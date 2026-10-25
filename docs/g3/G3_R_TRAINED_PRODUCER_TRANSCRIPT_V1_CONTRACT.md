# G3-R trained producer transcript V1

**Proposed private data contract. Capture, decoder, source union, and compiled
comparison are not yet accepted.** The accepted budget-one replay gate emits a
pass marker, not a transcript. This separate fixture must observe the same
trained CLI3 model and its two existing budget-one generator paths before
their release. Its pair is
`replay.c2` SHA-256 `9ec2a1d40ebd43e57d1e8b75fa8f6d96cd4d17abc1002f70da04d20144ee78df`
and `replay.g3rcv1` SHA-256
`75fec71697cf50556228481b0a2a30f634ce8991a7f6db4ad7fbdc700f1cfbf2`.
The historical replay receipt authenticating that pair is
`g3r-trained-replay-01a584c6-attempt3/SHA256SUMS-RUN`, SHA-256
`18b9577bf6651a6df7d74f54a6ddefbaf483f1cfe8fefc46aa2dca2c40d5960f`.
It contains no transcript. A future separately sealed producer receipt must
authenticate the exact transcript bytes and source closure before consumption.

## Fixed binary layout

All fields are little-endian. Counts, dimensions, lengths, ID and raw binary32
bits use unsigned 32-bit words; the four RNG fields at each trace point use
the exact signed 64-bit G3-S word bit patterns. Binary32 values are serialized
without numeric conversion. V1 is exactly 6,292
bytes; a decoder rejects any other length, version, magic, nonzero reserved
word, dimension, count, digest, or trailing data before exposing a result.

| Offset | Bytes | Meaning |
| ---: | ---: | --- |
| 0 | 8 | ASCII `G3RTV1` followed by two zero bytes |
| 8 | 4 | version = 1 |
| 12 | 4 | total length = 6,292 |
| 16 | 32 | exact saved C2 SHA-256 |
| 48 | 32 | exact G3RCV1 SHA-256 |
| 80 | 4 | raw V256 prompt byte, zero extended |
| 84 | 4 | context = 2 |
| 88 | 4 | unique primary parameter count = 14 |
| 92 | 4 | reserved = 0 |
| 96 | 4,960 | fourteen ordered parameter records |
| 5,056 | 1,204 | fixed P1/G1 trace |
| 6,260 | 32 | SHA-256 of bytes 0 through 6,259 |

Each parameter record is four words `(index, rank, dim0, dim1)`, then its
row-major binary32 words. `dim1=0` for rank one. The exact ordered path and
shape table is the accepted TR3 fixed roster in
`internal/p1/lib/transformer/module.esk` (`tr3-fixed-path?` and
`tr3-fixed-shape?`): indices 0–3 attention key/output/query/value weight
`[4,4]`; 4 FFN down `[4,8]`; 5 FFN up `[8,4]`; 6–7 norm1 bias/weight `[4]`;
8–9 norm2 bias/weight `[4]`; 10 head weight `[256,4]`; 11–12 final norm
bias/weight `[4]`; 13 position embedding weight `[2,4]`. This is 1,184
binary32 words. All are CPU f32. The tokenizer embedding is tied to head;
its live alias identity must be checked separately and is never serialized as
a second payload. Reject path order, rank, shape, dtype/device or tie drift.

The trace fields occur in this exact order: 256 last-logit binary32 bits from
the existing **captured** no-draw P1 owner; 4 key and 4 value binary32 cache
words at its length one; its 4 RNG signed 64-bit words; its logical length and
cache length (both one); emitted G1 token ID from the separate genuine
**reference** generator; 8 key and 8 value binary32 cache words from that
reference after G1; its 4 successor RNG signed 64-bit words; its G1 logical
length and cache length (both two). The 4 captured P1 RNG words must equal the
authenticated companion state: P1 consumes no draw. The ID is 0–255 and the
companion's policy is exactly budget one. No text, pointer, registry identity,
object header, or executable term is present.

## Producer and consumer obligations

Capture parameter bits directly from the live trained model after authentic
CLI3 restore and exact C2 SAVE, while the owner is retained; do not derive
them by decoding C2. Capture P1 from the producer's `captured` generator and
G1 from its separate genuine `reference` generator. The accepted
`g3r-trained-replay-compare!` first checks captured P1 against the decoded
no-draw `replayed` owner, then checks genuine reference G1 against replayed
G1, including ID, lengths, K/V and successor RNG. The transcript must preserve
these owner labels; it must not claim that P1 and G1 were observed on one
generator. Getter reads and serialization must add no draws or state changes.
The existing reference G1 call advances RNG as part of ordinary generation;
model, tokenizer, checkpoint, and companion bytes remain unchanged. Publish to a distinct
path with atomic no-overwrite semantics after all fields and checksum are
complete. A failure may leave an orphaned pair and must not overwrite it.

The fresh consumer authenticates both the historical pair receipt and the
future transcript-producer receipt, the exact pair bytes and the sealed
transcript bytes. It checks both header digests and full checksum, then compares live restored
parameter carriers bit for bit and the P1/G1 trace word for word. Header
digests and checksum detect corruption; the new sealed receipt supplies
transcript provenance.
Normal, fresh repeat and native ASan+UBSan must produce byte-identical
transcripts. Corrupt a byte without recomputing the checksum and require
checksum rejection. Separately, mutate each semantic field class and
recompute the checksum: wrong fixed shape/count/version/C2 or pair digest,
out-of-range ID, wrong RNG or length, and wrong parameter or trace bit must
be rejected by exact expected-value comparison with unchanged destination;
one-bit live parameter, logit, K/V, ID and RNG mutations must cause mismatch.
This V1 profile remains context two and one G1; it proves no later EOS or
multi-token continuation.

## Source union preflight

`tests/g3r/preflight_trained_transcript_union.py` pins the accepted seam
`bba97271` and replay `01a584c6` closures. Their 92 shared source paths have
identical Git blobs. The planned fresh consumer has one seam root plus only
`g3r_rng_import_extension.esk` and `g3r_replayed_p1_g1_extension.esk`, for
103 unique sources. It must **not** load the replay producer root, which
would load `cli3_generate_root.esk` a second time. The distinct producer
transcript image starts from the accepted replay root and its 97-source
closure, plus a future reviewed capture leaf. The preflight is source-only;
it does not prove generated IR, native link, runtime registry identity or
capture behavior.
