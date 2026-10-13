# G3-R committed snapshot to record adapter contract

Status: the current-main combined committed-save snapshot gate is accepted.
This contract now has a source-private implementation and feature-local tests;
its exact-head runtime and independent-review gates remain separate. It adds
no public name, package export, file I/O, or new upstream API.

## Exact inputs and authority boundary

The feature-local adapter may compose only the existing
`(g3r-committed-save-snapshot! generator destination)` and
`(g3r-record-encode! destination policy rng history prompt generated manual
fingerprint c2-digest)` entries. The first requires an exact live, idle,
same-aggregate G3-T/M3T/raw-V256 T1 owner and a caller-owned bytevector of at
least 136 bytes. It returns detached seven-field data in this order:
`policy[6]`, `rng[4]`, ordered raw history, origin prompt length, generated
count, manual count, generated-EOS terminal bit. Its 136-byte copy is staging
data, **not** the G3RCV1 record. The second writes an exact 288-byte record
only after its own validation and SHA-256 work succeed.

The adapter maps the first six returned fields directly into the existing
encoder signature. It derives the 96-byte fingerprint from the exact raw-V256
tokenizer identity authenticated by the snapshot, verifies its byte length and
equality to that identity, and passes it as the existing encoder argument.
It must validate the seventh field before encoding: terminal is one exactly
when `generated=1`, EOS is configured, and the last generated history byte
equals EOS. A prompt or manual byte equal to EOS does not make the record
terminal. The wire format stores this classification through its policy,
history and partition fields; it has no terminal-bit field. Any mismatch
rejects before the 288-byte destination changes.

The encoder's 32-byte `c2-digest` argument is a precondition, not C2
authority. For this leaf's tests, derive it independently from one immutable
C2 fixture image and call it **test-authenticated**. In production, a later
writer must obtain a whole-image SHA from the authoritative C2 SAVE image and
prove that exact model/tokenizer image belongs to the snapshotted generator.
No such SAVE receipt or file-pair publication is established here. A caller's
arbitrary 32 bytes, or equality with a digest recomputed from a mutable path,
cannot authorize publication. The in-memory record may be decoded only with
an independently authenticated expected digest, as required by the existing
codec and pair-decode contract.

## Publication and failure order

Keep the 136-byte snapshot scratch and 288-byte record scratch private. Take
the same-owner snapshot first; verify the detached shape, exact fingerprint,
partition and terminal consistency; then call the existing staged encoder.
Only a successful encoder result may copy 288 bytes once to a caller-owned
destination. Before that final copy, authenticate its exact extent and
disjointness from source scratch. Snapshot, validation, allocation, SHA, or
encoder failure leaves the caller's 288-byte destination unchanged. A failed
snapshot also leaves its caller-owned 136-byte destination unchanged under the
existing snapshot contract. No pointer, registry identity, cache, lease, or
live owner enters the record. Detached output bytes are never substituted for
committed history.

## Bounded test matrix and next gate

The accepted current-main snapshot aggregate passed its combined compiled
runtime gate. A feature-local same-aggregate test now exercises genuine
P1/G1 commit and compare the adapter's 288 bytes against an independent
Python `struct`/`hashlib` oracle: exact header/version/profile/algorithm,
six policy and four RNG signed-i64 words, length/partition, ordered IDs,
zero unused byte and padding, exact 96-byte fingerprint, C2 digest at
`224..255`, and SHA-256 of `0..255` at `256..287`. Decode with the fixture's
independently computed expected digest and compare all eight detached fields.
Use the existing controlled native history tests for P1/P2, generated/manual
partitions and first generated EOS; add an aggregate witness for each route
only after its real Eshkol path is admitted. Do not promote controlled stubs
to a real same-owner claim.

Negative cases must cover short/aliased destination, wrong or foreign owner,
held model/A2 lease, busy call, idle post-P1 G1 abort, changed tokenizer or
policy, impossible partition, forged terminal bit, invalid fingerprint,
malformed digest extent, failed allocation and SHA/encoder cuts. Fill the
caller destination with a sentinel and compare all bytes after each rejection;
verify old outputs, generator/cache/RNG and retry where those owners exist.
An arbitrary but 32-byte digest is **not** an encoder rejection case: pair
mismatch belongs to authenticated decode/load tests. Run normal, repeat,
ASan+UBSan+LSan, feature-off symbol and source-closure checks for the eventual
source leaf. This contract-only commit runs no such gates.

File-pair writer/reader, atomic pair publication, fresh-process live-model
restore, no-draw cache replay, resumed public generation, and a larger
repeated-decode/later-EOS profile remain separate dependencies for full G3-R.

## Implementation status on this branch

`native/g3r_snapshot_codec_adapter_extension.esk` composes the admitted
snapshot and codec, validates the detached partition, terminal bit and exact
raw-V256 fingerprint, and publishes one staged 288-byte copy. The feature-local
aggregate fixture uses the deterministic valid M3 C2 test image produced by
`tests.c2.prepare_checkpoint_operational_fixtures.make_m3_artifact((2, 4))`
for its independent whole-image digest. This digest is test-authenticated only.
The compiled normal/repeat/sanitizer gate and independent review remain open;
no SAVE writer or C2-to-generator ownership proof is added.
