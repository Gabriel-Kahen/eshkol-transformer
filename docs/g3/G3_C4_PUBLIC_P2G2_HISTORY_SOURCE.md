# Public P2/G1–G2 history source prerequisite

The installed manual C4 aggregate privately enables
`ET_G3C4_P2_ORIGINAL_SNAPSHOT_PRIVATE`. Only the genuine sealed-T1 P2
constructor marks an input as exact P2 and retains its two original V256 IDs.
The generic length-two constructor has no marker. A private 24-byte C ABI
destination comprises the existing 8-byte length header (`16`) followed by
two little-endian i64 IDs. The getter authenticates the live input address,
owned dense CPU-i64 `[1,2]` tensor, original IDs and absence of an active I1
borrow before writing. Certain release clears the marker and IDs.

The existing public input constructor enrolls its T1 shell and native input
in a lexical issuer. The private capture entry
`g3c4-public-p2g2-history-capture-internal!` joins that issuer to one exact
public G2 output, ticket, model, tokenizer and creation policy. Admission
through `g3c4-public-p2g2-history-admit-once-internal!` rechecks the owners,
terminal IDs, count, cache and terminal-output RNG, and spends the issuer
once. The returned detached vector is mutable and is **not** independent
authority for a later writer. There is no generator-current-RNG getter or
same-handle model parameter-bit witness in this leaf.

The history issuer is finite (64 inputs and 64 histories per aggregate),
retains canonical roots, and has no retire/consume path yet. This is a
source-only candidate: compiled native/Eshkol fault cuts, public EOS/G2
runtime paths, resource limits, model-bit capture and the versioned paired
SAVE/LOAD format require separate acceptance.
