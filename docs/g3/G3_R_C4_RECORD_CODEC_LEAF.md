# Private C4 data-only continuation record codec

Status: source/local candidate only. The C4 record is a proposed wire choice,
not a live SAVE point, restored owner, or resumed generation. The accepted C2
`G3RCV1` source and its 288-byte encoding are unchanged. This leaf loads that
codec only for checked byte, integer, RNG and T1 fingerprint helpers; the new
functions remain source-private.

The record is exactly 288 bytes. It uses the accepted C1 SHA-256 and preserves
the original field offsets outside the expanded history and phase word:

| Offset | Bytes | Canonical C4 field |
|---:|---:|---|
| 0 | 8 | ASCII `G3RCV2` and two zero bytes |
| 8 | 2+2 | LE major 2, minor 0 |
| 12 | 4 | LE total length 288 |
| 16 | 4 | LE profile 2, fixed `diagnostic-c4` P2/G2 data profile |
| 20 | 4 | LE G3-S/Philox algorithm 1 |
| 24–36 | 16 | LE history length, prompt count, generated count, manual count |
| 40 | 48 | six signed LE i64 policy words: mode, f32 temperature bits, top-k, f32 top-p bits, requested budget 2, EOS (`-1` or byte ID) |
| 88 | 32 | four signed LE i64 RNG words: version 1, nonnegative seed, two complete counter words |
| 120 | 4 | ordered raw-V256 history IDs; unused slots zero |
| 124 | 4 | LE phase 0–3 |
| 128 | 96 | canonical raw-V256 T1 fingerprint text |
| 224 | 32 | full SHA-256 digest of one separately authenticated immutable C2 image |
| 256 | 32 | SHA-256 of bytes 0–255 |

Only a two-ID origin prompt and zero manual appends are encoded. Length is
`2 + generated`, within 2–4. Phase 0 is prompt-ready with generated count
zero. Phase 1 is a **proposed stable one-token non-EOS history** with count
one and a first generated ID unequal to configured EOS. Phase 2 is terminal
first-token EOS, with count one and an explicit matching EOS. Phase 3 is a
terminal two-token result, with count two and a non-EOS first generated ID;
the second may equal EOS or finish by budget. EOS in the prompt has no
terminal meaning. Invalid phase/count/EOS combinations and nonzero unused
history slots reject even when the record checksum is repaired.

The policy projection follows the accepted C4 private budget-two gate:
greedy or categorical, positive finite f32 temperature, top-k `1..256`,
positive finite f32 top-p at most one, requested budget exactly two, and EOS
disabled or `0..255`. Greedy retains exact default controls. The header
identifies the C4 profile; the six numeric words contain no live profile
symbol. The source-private encoder and decoder cannot establish that these
words came from a same-registry generator. The T1 fingerprint is checked for
canonical syntax, not against a live tokenizer. The C2 digest argument is an
already authenticated whole-image precondition; the codec opens no file.

`g3r-c4-record-encode!` validates, constructs and hashes private staged bytes
before its sole destination copy. `g3r-c4-record-decode!` validates exact
extent, header, pair digest, checksum, zero unused history, and field domains;
it then creates newly owned policy, RNG, history, fingerprint and digest
objects and publishes nine data-only slots with one checked whole-vector
copy. Rejected inputs preserve destination bytes or slot identities under
serialized caller access. C2 and C4 decoders reject each other's magic and
version. The independent Python oracle compares six complete byte records
and checks their SHA-256 values; malformed and cross-profile fixtures cover
failure atomicity.

Local Eshkol 1.3.4-evolve with Clang 22.1.8 passed normal, repeat and
ASan+UBSan fixture runs: C4 50 checks, unchanged C2 49 checks, and
feature-off 2 checks in each run, with byte-identical output and empty
compile/runtime stderr. Independent Python oracles matched six C4 and five
C2 full records. This is unpinned local evidence, not a reviewed compiled
admission gate.

**Phase 1 has no admitted live producer.** The current P2/G2 non-EOS prefix
remains inside one active `m3-call`; the accepted G3-R SAVE contract rejects
active calls. A completed P2/G1 output uses requested budget one and cannot
be relabeled as cumulative budget two. Before a C4 companion may be saved
between tokens, a reviewed closed-call stable one-token snapshot/hold
transaction must authenticate the exact C4 model, P2 IDs, first selected ID,
T1 byte, policy, RNG, cache and retained next logits. A separate reviewed
second-token entry must resume from that idle P3 state without resampling or
re-prefilling token one. Neither API exists here. A fresh-process replay and
trained C4 checkpoint binding are further gates; no phase code authorizes
generation or changes terminal behavior.
