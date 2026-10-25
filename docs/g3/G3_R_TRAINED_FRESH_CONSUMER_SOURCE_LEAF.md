# G3-R trained C2 fresh consumer source candidate

**Source-only candidate.** No producer transcript receipt, linked consumer,
compiled test, or cross-process parity is accepted here. The exact input pair
and V1 layout are specified by the [fresh comparison contract](G3_R_TRAINED_FRESH_BIT_G1_COMPARISON_CONTRACT.md)
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
not native behavior. The next gate requires an independently accepted producer
transcript and receipt, source/test review, an exact linked one-root harness,
fresh-process normal/repeat and native sanitizer runs with the pinned source
closure and feature-off controls. No persisted generated-history resume,
multi-token decoding, EOS, public API, or full G3 continuation is claimed.
