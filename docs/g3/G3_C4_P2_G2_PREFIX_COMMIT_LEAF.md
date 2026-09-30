# Private P2/G2 first-token prefix commit

`ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE` extends the already gated first-frame
native build. Its only added symbol is
`et_g3c4_private_p2g2_prefix_commit_v1(context, pending_output, raw_header)`.
The caller must first reserve the budget-two output, prefill an authentic owned
P2 prompt, sample from its `[1,256]` last-logit row, and run the position-two
token forward. `raw_header` is an unaliased nine-byte carrier: an `i64` byte
count of one followed by the caller's V256 raw byte. The entry checks that
byte against the selected ID and rejects a matching first-token EOS before
mutation. Genuine T1 decode provenance is a later Eshkol gate.

The forward retains its `[1,256]` next logits and the staged `[1,2,4,2]`
K/V words in the preallocated context. Commit validates the output's unused
`I1[2]` carrier, staged K/V, length `3`, mask `1110`, pin binding, raw byte and
view-free A2 transaction. Its closed tail commits position-two K/V, advances
the four-word RNG, and retains selected ID, byte and logits in private prefix
state. The requested budget remains two; the pending output has no numeric,
ID or text readiness. No shell, public accessor or terminal result is made
live. An impossible A2 commit failure aborts the process.

Finish rejects the intermediate state. Abort preflights pending output and A2
leases, discards any pending output, scrubs the private prefix data and drains
the call while retaining committed cache `3/1110`, binding and successor RNG.
Held leases or malformed carriers reject before mutation. The next gate must
provide second-token sampling/forward, EOS-aware terminal publication and
preallocated logical `G=1/2` output forms; this leaf exposes none of them.
