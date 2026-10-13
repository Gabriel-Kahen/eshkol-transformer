# G3-R current-main same-owner capture source port

**Source-only candidate; integration HOLD.** This branch now includes post-E3
remote main `bd81379c2141ff28ffa847ce579a658d98d78b85` (PR #205),
merged over its original `6f6ef9ada4c586573d495c40283ad4b9410d0957`
(PR #204) base. It ports the private D2, M3/X1, G3 creation,
training-control and O2 adoption roots and
the fixed diagnostic-C2 capture caller. The nine existing-file deltas add
private lineage/control routes and O2 held-state hooks. Current-main G3-T,
F32, and M3 owner implementations remain authoritative. There is no public
SAVE, pair publication, restore, replay or resumed-continuation admission.

The native coordinator begins a registered G3/A2 held owner, pins the same
fourteen M3 handles and F32 bit witnesses, and holds the original O2 owner.
It captures only the current-main 136-byte, **17 signed 64-bit word** image:

| Word | Owner field |
|---:|---|
| 0-5 | sampling policy |
| 6-9 | G3 RNG state |
| 10-11 | ordered raw history IDs |
| 12 | committed history length, 1 or 2 |
| 13-15 | prompt, generated, manual counts |
| 16 | generated EOS terminal provenance |

The Eshkol caller compares all six policy words to the live generator policy,
requires exact signed 64-bit words, checks the history partition and byte
range, and recomputes the terminal bit from the generated ID and EOS policy.
The old 18-word version-prefixed ledger image is rejected by shape, and its
feature macro is a compile-time error in the coordinator header. The current
G3 owner supplies the history and rejects pending G1; the capture caller does
not reconstruct either from a detached output.

Both private O2 hold and held-state issue reject any overlap between their
output slot and error record before reading or writing the slot or changing
owner state. This denial returns invalid-argument status without touching the
overlapping caller bytes; the registered hold/state and optimizer busy bit
remain unchanged.

The successful private return is `[detached C2 owner, copied 17-word G3 image,
training-control epoch]`. It grants no file writer or model/tokenizer identity
to a later consumer. The closure list under
`native/g3r_same_owner_o2_adoption_source_closure.txt` identifies the
recursive Eshkol sources; the adjacent current-main SHA-256 manifest pins the
reviewed bytes.

The post-E3 source/closure/tests at `929199c` passed independent review.
Offline source predicates, native O2 hold cuts, header/offset compile checks,
and native syntax checks remain preparation evidence only. Real compiled
Eshkol post-adoption longjmp and partial O2 claim cuts, current-main native
union, full supported toolchain, and ownership/failure matrix remain required
before integration. The old-base post-adoption fixture
cannot certify this branch because its ancestry and source hashes are pinned
to the divergent 18-word implementation.
