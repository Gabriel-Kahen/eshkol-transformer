# G3-R trained C2 fresh consumer

**Bounded compiled fresh-process comparison accepted.** Source
`f4214f6a`/tree `c2456cce` passed independent source review. The one-use
normal/repeat/native ASan+UBSan consumer gate is sealed at
`g3r-trained-fresh-consumer-f4214f6a-attempt2/SHA256SUMS-RUN` (SHA-256
`a1407b14da03523b019165a1d6eae92f416fa00280122e44aba8c7669f092b70`)
with independent review
`g3r-trained-fresh-consumer-v2-independent-result-review-20261025/SHA256SUMS-REVIEW`
(SHA-256 `0fa1a88faea087621236293708200f5b9f8070fc931eabe3eea1e620a51bb323`).
The first attempt remains sealed nonaccepting: SHA-256
`b1c7db388a7c10b042b25a1f73b0aff472a57ec5b3ea931abf76b08c68330775`,
missing the private C2 SHA-stage native symbol at thin executable link.
The producer transcript is accepted
separately at `c04f7017` with sealed run
`g3r-trained-transcript-c04f7017-attempt3/SHA256SUMS-RUN` (SHA-256
`4bcedbf176296accf8afa8699cc710a552cc892e263ed731fd03ab81bb3cebc3`).
The exact input pair and V1 layout are specified by the [fresh comparison contract](G3_R_TRAINED_FRESH_BIT_G1_COMPARISON_CONTRACT.md)
and [producer transcript contract](G3_R_TRAINED_PRODUCER_TRANSCRIPT_V1_CONTRACT.md).

`native/g3r_trained_fresh_consumer_root.esk` loads the reader, receipt-to-TR3
seam, typed RNG import, and private replayed G1 leaf in that order. Its closure
replaces the seam root and adds only the two existing replay imports: 103
unique sources, one C2/TR3/G3 registry. The producer root is absent. The
entry `g3r-trained-fresh-consume-internal!` receives a live same-image C2/SHA
receipt, fresh idle trainer, live canonical tokenizer, exact companion bytes,
and sealed V1 transcript bytes. The caller must construct and own these inputs
in this same root and independently authenticate both the historical pair
receipt and the new producer transcript receipt before calling it. This entry
does not open paths or authenticate an external seal.

Before restore it checks the receipt digest, exact pair SHA, companion policy,
tokenizer fingerprint, V1 header, ordered 14-record roster, lengths, P1 RNG,
and checksum. It stages decoded data without publishing on a malformed
transcript. It then restores that one trainer through the accepted seam. Under
the exact receipt/C2/P1 borrow it compares every primary source carrier with
the corresponding fresh model carrier using native `i2-value-equal?` and
closes subordinate P1 access before the receipt borrow. Separately, it checks
the 14 live model paths, shapes, CPU f32 role, tied embedding/head identity,
and physical binary32 bytes against the independent transcript payload.
Borrow cleanup uses stable cells under the pinned guarded-failure behavior;
a failed cleanup is fail-stop, not a recoverable parity result.

After idle lease unenrollment and receipt release, it evaluates the model,
imports the companion's four RNG words through the typed G3 owner, executes
one no-draw P1 and one genuine private G1, and compares all 256 P1 last-logit
bits, P1 and G1 K/V words, P1 and successor RNG words, emitted ID, and
logical/cache lengths. It requires the full-context G2 preflight to reject.
The bridge only reports host byte order; it does not expose a pointer.

`tests/g3r/test_trained_fresh_consumer_source.py` checks one-root order and
103-source closure, source mutation controls, synthetic V1 decode rejection
with unchanged destination, and one-bit semantic mismatches in parameter,
logit, K/V, ID, and RNG fields. These Python checks are schema/source checks,
not native behavior. A new private test wrapper constructs a canonical tokenizer,
fresh trainer and exact C2/SHA receipt, reads the host-sealed companion and
transcript files, and calls the consumer within one package. A separate thin
executable invokes it. The linked gate passed this executable in normal,
repeat and native sanitizer processes, with exact parameter-bit and P1/G1
trace comparisons. A separate thin negative caller required the exact typed
consumer failure and unchanged destination for checksum, pair, parameter,
logit, P1/G1 K/V, ID and RNG mutations in normal and sanitizer modes.
No persisted generated-history resume, multi-token decoding, EOS, public API,
or full G3 continuation is claimed.
