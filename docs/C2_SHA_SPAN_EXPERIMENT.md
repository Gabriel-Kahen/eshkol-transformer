# C2-owned C1 SHA span experiment

Status: test-only source candidate. The ordinary C1 and C2 checkpoint paths,
public ABIs, callback, and digest bytes are unchanged.

The accepted C1 encoder allocates a fresh contiguous preimage for each tensor
digest (`domain + zero-digest record + payload`) and another for the container
digest (`domain + unsigned container`). C1 also copies the tensor payload into
its descriptor and assembles the final container; this experiment addresses
only the two preimage concatenations. The C2 format module already processes
SHA-256 in successive spans. Under `ET_C2_SHA_SPANS_EXPERIMENT`, a private
helper hashes up to three owned bytevectors without constructing a preimage.
It validates lengths, cumulative SHA bit length, output size, and output/input
overlap before writing the 32-byte digest. No default object exports the helper.

Run `bash scripts/test-c2-sha-spans-experiment.sh` for optimized and
ASan/UBSan/LSan digest-equivalence and malformed-input checks. The tests cover
C1 tensor and container domain bytes, zero and SHA block-boundary payloads,
the independent SHA-256 `abc` known answer, unchanged inputs, and no digest
write on rejection. The C1 encoder is not wired to this helper; a later
feature-private callback change would have to preserve the existing record
zero slot, domain ordering, payload ownership, callback failure behavior,
source/symbol manifests, and byte-identical save/load gates.

The sealed sequential native-first C2 pair at
`/home/gabe/.codex/evidence/eshkol-transformer/c2-sha-matched-pair-e3280e3-20261126/RESULTS.md`
reported 140.908 versus 880.920 seconds, with native runtime peak RSS 47,744
KiB higher. It was one fixture on an uncontrolled host. This test-only helper
has no measured RSS or throughput effect and does not attribute that delta to
preimage buffers. The next gate is independent source/test review, followed by
an isolated Eshkol callback integration and exact-byte operational replay.
