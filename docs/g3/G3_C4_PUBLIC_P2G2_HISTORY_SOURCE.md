# Public P2/G1–G2 history source prerequisite

The installed manual C4 aggregate privately enables
`ET_G3C4_P2_ORIGINAL_SNAPSHOT_PRIVATE`. Only the genuine sealed-T1 P2
constructor marks an input as exact P2 and retains its two original V256 IDs.
The generic length-two constructor has no marker. A private 24-byte C ABI
destination comprises the existing 8-byte length header (`16`) followed by
two little-endian i64 IDs. The getter authenticates the live input address,
owned dense CPU-i64 `[1,2]` tensor, original IDs and absence of an active I1
borrow before writing. Certain release clears the marker and IDs.

The existing public input constructor enrolls its T1 shell identity and
native input in a lexical issuer. The original T1 shell may then be released;
capture and admission use the native exact-P2 getter, without rereading T1.
The manual private aggregate now loads the accepted
`t1_published_release_private.esk` seam after its canonical T1 root and
compiles its T1 shell with `ET_T1_PUBLISHED_RELEASE_PRIVATE`. The staged
`tests/g3c4/p2g2_history_release_private.esk` gate creates the genuine P2
input, returns from the copying and issuer-enrolling constructor, physically
releases that exact T1 shell, checks dead native length/read and stale decode
rejection, and then runs the EOS and two-ID continuations. Both histories are
captured and admitted once from independent native input IDs after release.
The fixture is source checked; it has not yet run in a compiled aggregate.
The private capture entry
`g3c4-public-p2g2-history-capture-internal!` joins that issuer to one exact
public G2 output, ticket, model, tokenizer and creation policy. Admission
through `g3c4-public-p2g2-history-admit-once-internal!` rechecks the owners,
terminal IDs, count, cache and terminal-output RNG, and spends the issuer
once. Both routes perform input-provider work before their final terminal
snapshot, then check captured origin and ticket cells without callback calls
before the fixed publication tail. The public source/ticket count, first ID,
and second ID are checked against terminal snapshot slots 2, 0, and 1;
snapshot slot 3 is cache length. The returned detached vector is mutable and is **not** independent
authority for a later writer. There is no generator-current-RNG getter or
same-handle model parameter-bit witness in this leaf.

The history issuer is finite (64 inputs and 64 histories per aggregate),
retains canonical roots, and has no retire/consume path yet. This is a
source-only candidate: compiled native/Eshkol fault cuts, public EOS/G2
runtime paths, resource limits, model-bit capture and the versioned paired
SAVE/LOAD format require separate acceptance.
