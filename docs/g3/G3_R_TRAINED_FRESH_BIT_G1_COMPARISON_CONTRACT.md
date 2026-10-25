# G3-R trained C2 fresh-process bit and P1/G1 comparison

**Proposed bounded private evidence contract; no source aggregate or compiled
comparison is accepted by this document.** This is a context-two, one-byte
prerequisite for the same trained checkpoint. It does not establish resumed
saved-policy generation, multi-token decoding, later EOS, or public resume.

## Exact inputs and independent oracle

The accepted budget-one pair is
`g3r-trained-replay-01a584c6-attempt3/normal/replay.c2` (SHA-256
`9ec2a1d40ebd43e57d1e8b75fa8f6d96cd4d17abc1002f70da04d20144ee78df`)
and `normal/replay.g3rcv1` (SHA-256
`75fec71697cf50556228481b0a2a30f634ce8991a7f6db4ad7fbdc700f1cfbf2`).
Its 519-file receipt `SHA256SUMS-RUN` has SHA-256
`18b9577bf6651a6df7d74f54a6ddefbaf483f1cfe8fefc46aa2dca2c40d5960f`
and zero launcher/Docker exits. This C2 image is byte-identical to the fa9
trained C2 (SHA-256 `9ec2a1d4…`), while this companion has `max_new=1` and
the fa9 companion has `max_new=0`. All later comparisons must authenticate
both the replay receipt and pair bytes; a hash-equivalent or equal-seed model
must not replace the exact saved image.

The accepted `01a584c6` same-registry producer compares its captured trained
owner to its decoded no-draw P1 replay: all 256 last-logit bits, first A2 K/V
row and four RNG words, then genuine reference/replayed G1 ID, logical and
cache lengths (one and two), both K/V rows and successor RNG. Its sealed stdout contains
only a pass marker. **It contains no reusable bit or G1 trace transcript.**
A separate reviewed, data-only oracle fixture must capture the fixed ordered
14 live parameter payloads from that trained model and both no-draw P1 and
private replayed G1 from the decoded `replayed` generator in the producer
process before owner release. The accepted gate compares that replayed P1 to
the independent `captured` P1 and its G1 to the genuine `reference` generated
call; the future transcript must identify `replayed` as its sole trace owner
and retain those two independent controls. It must
version, bound and checksum
its output, bind it to the exact C2 and G3RCV1 SHA values, and seal it with
normal/repeat/native sanitizer byte parity. No pointer, registry identity or
executable term may be serialized. Its capture must not alter training,
model parameters or saved pair bytes; observational reads add no RNG draws or
state changes beyond the existing replayed and reference G1 calls. The historical replay
receipt authenticates the pair, while a new receipt must authenticate the
transcript bytes and source closure.

## Fresh consumer and parameter proof

A distinct process loads one private root containing the accepted reader and
`bba97271` receipt-to-TR3 seam exactly once. It admits the budget-one pair
through the exact same-image C2/SHA receipt and live canonical T1, constructs
one fresh model/optimizer/D2/trainer candidate, restores only that trainer
from the retained owner, and leaves the receipt and trainer idle. It then
compares the fresh model's fixed 14 primary paths, shapes, dtype/device and
tied token-embedding/head alias against the C2 topology and sealed producer
oracle. `parameter-tree-paths`, `parameter-tree-handle`,
`parameter-handle-tensor-identity-internal`, and `i2-value-equal?` already
expose exact live same-registry parameter comparison; the latter delegates
to native bit equality. The accepted C2 borrow/projection and
`state-dict-c2-owned-tensor-borrow-{begin,end}-internal` can expose each
fixed source carrier **only while its exact receipt borrow is active**. A
private status-only comparator may compare each source carrier directly to
the fresh model carrier, ending every subordinate P1 access before the C2
borrow and returning no carrier. The sealed producer payload supplies an
independent pre-save bit oracle; a second load of the same C2 alone does not.
Compare raw binary32 bits for every element, including signed zero and NaN
patterns, rather than numeric approximate equality or a model digest.
Reject one-bit payload mutation, path/shape/tie drift and a seeded but
unrestored candidate. All comparisons precede receipt/trainer release; all
borrows close on success and recoverable failure.

## P1/G1 trace and gate

The fresh candidate releases its idle trainer lease, enters eval mode, and
imports the companion's authenticated four-word G3-S state through the
accepted typed RNG owner. Under the exact recorded budget-one policy and
raw-V256 prompt, it executes genuine no-draw `g3m-prefill-p1!`, checks the
four RNG words unchanged, then executes one admitted G1 path. The accepted
`g3r_replayed_p1_g1_extension.esk` and its test-only native getters show the
source shape for 256 last-logit bits, A2 K/V rows, emitted ID, G1 output
logical length one/cache length two and successor RNG; they need an exact
reviewed union with the reader
root, not a second CLI3/T1/G3 registry. The fresh trace is compared element
for element against the sealed producer transcript. Include a one-bit logit,
K/V, output-ID and RNG mutation control, plus full-context no-draw rejection
after G1. The old fa9 budget-zero companion is a wrong-policy control.

Use one immutable source tree/package closure per mode, normal fresh process
and repeat plus native ASan+UBSan, empty relevant stderr, exact symbol/export
closure, authenticated producer and oracle receipts, and one-use output seals.
A successful gate proves exact fresh restored parameter bits and this
context-two P1/G1 trace for the same trained C2. It does not prove that a
saved generated history resumes, repeated decode, later EOS, or an installed
public continuation. No implementation should start until the producer
transcript schema and exact source union are independently reviewed.
