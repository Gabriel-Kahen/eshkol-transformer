# G3-T private owned scalar-token input

This leaf completes the accepted source-private `input_from_token(i64)` and
`g3t-generation-token-input-create/1` contract for an exact byte ID. Under
`ET_G3T_OWNED_TOKEN_INPUT_PRIVATE`, the native constructor validates 0..255
before allocation, creates a distinct rank-two CPU I1[1,1], copies its one
word, and enrolls the kind-2 input only after the I1 is complete. Its inline
ID is validated staging for the existing frame ABI; the I1 is canonical
owned storage. The historical inline-only witness remains selected when the
new private feature is off.

The Eshkol wrapper preallocates the shell, pending registry entry, rooted
ledger and list before native construction. On a native failure or any later
wrapper exception, it releases any new native owner and leaves an inert
tombstone. Typed release checks an active I1 borrow before destruction,
destroys the I1 once, and remains idempotent for the exact dead owner.

The focused P2 aggregate witness covers range/identity rejection, a native
header failure and four I1 allocation cuts, wrapper failure tombstones,
borrow-guarded release, exact owned singleton word and I1 count, and P1/G0
and P1/G1 prefill through the owned scalar input. It preserves the separate
T1-backed P1/P2 input witnesses. No public G3-G facade, export, package or
CLI is added; P2/G1 still requires a capacity-three contract.

The supported network-disabled image
`eshkol-checked-promotion-llvm21:20260922`
(`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`)
strictly compiled the P2 test and ran its normal, repeat and
ASan+UBSan+LSan gate against source commit
`6bbc14a4e58afbae6112e750846db52486daa2ff` (tree
`6a2c76b156285ea6c2a0705c545bdd7c6f259808`). All three reported
46,586 checks with matching output and empty compiler/runtime stderr.
Ten source contracts and Q0 4/4 passed. The sealed external evidence is
`g3t-owned-token-input-6bbc14a`.
